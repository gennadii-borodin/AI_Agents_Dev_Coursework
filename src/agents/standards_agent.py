import functools
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import json_repair
import yaml

from src.config import Settings, get_settings
from src.llm_provider import RouterAIProvider
from src.models import StandardsReport, TestCase
from src.prompts import build_agent_system_prompt, build_json_schema
from src.tools.sql_tool import get_all_test_cases, get_test_cases_by_reqs
from src.tracing import OTEL_AVAILABLE
from src.tracing import otel_trace as _otel_trace

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def load_standards_rules() -> dict[str, Any]:
    """Загружает реестр правил QA-TEST из data/standards_rules.yaml."""
    path = Path(__file__).resolve().parent.parent.parent / "data" / "standards_rules.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=None)
def rule_classification() -> tuple[frozenset, frozenset]:
    """Возвращает (blocking_rule_ids, auto_fix_rule_ids) из реестра правил."""
    rules = load_standards_rules().get("rules", [])
    blocking = frozenset(r["id"] for r in rules if r.get("blocking"))
    auto_fix = frozenset(r["id"] for r in rules if r.get("auto_fixable"))
    return blocking, auto_fix


@functools.lru_cache(maxsize=None)
def num_active_rules() -> int:
    """Число активных правил QA-TEST из реестра (data/standards_rules.yaml).

    Используется как знаменатель метрики compliance вместо магической
    константы (M3).
    """
    return len(load_standards_rules().get("rules", []))

STANDARDS_SYSTEM_PROMPT = build_agent_system_prompt("standards_agent")

# Единый источник правил QA-TEST — data/standards_rules.yaml (T7, §1.3 ревью).
# Инжектируем актуальные правила в системный промпт, чтобы исключить
# дублирование/рассинхрон с hardcoded-текстом в prompts/standards_agent.yaml.
_STANDARDS_RULES_TEXT = yaml.safe_dump(
    load_standards_rules().get("rules", []), allow_unicode=True, sort_keys=False
)
STANDARDS_SYSTEM_PROMPT = (
    STANDARDS_SYSTEM_PROMPT
    + "\n\n## Действующие правила QA-TEST (источник: data/standards_rules.yaml):\n"
    + _STANDARDS_RULES_TEXT
)


def _prepare_test_cases_data(test_cases: list[dict]) -> list[dict]:
    result = []
    for tc in test_cases:
        result.append({
            "id": tc["test_case_id"],
            "req": tc["req"],
            "title": tc["title"][:60],
            "desc": tc["description"][:40] if tc.get("description") else "",
            "expected": tc["expected_result"][:40] if tc.get("expected_result") else "",
            "type": tc["test_type"],
            "quality": tc["design_quality"],
            "review": tc["qa_review"],
            "precond": tc["preconditions"][:30] if tc.get("preconditions") else "",
            "data": tc["test_data"][:30] if tc.get("test_data") else "",
            "steps": tc["steps"][:60] if tc.get("steps") else "",
        })
    return result


def _fix_json(text: str) -> str:
    text = text.strip()
    if not text.startswith("{"):
        idx = text.find("{")
        if idx >= 0:
            text = text[idx:]
    brace_count = text.count("{") - text.count("}")
    if brace_count > 0:
        text = text + "}" * brace_count
    bracket_count = text.count("[") - text.count("]")
    if bracket_count > 0:
        text = text + "]" * bracket_count
    if text.endswith(","):
        text = text[:-1]
    if not text.endswith("}"):
        text = text + "}"
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r":\s*,", ": null,", text)
    return text


def _parse_llm_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            repaired = json_repair.repair_json(response, return_objects=True)
            if isinstance(repaired, str):
                return json.loads(repaired)
            elif isinstance(repaired, dict):
                return repaired
            else:
                raise ValueError(f"Unexpected type: {type(repaired)}")
        except Exception:
            if OTEL_AVAILABLE and _otel_trace is not None:
                s = _otel_trace.get_current_span()
                if s is not None and s.is_recording():
                    s.add_event("json_repaired", {"agent": "standards", "tool": "json_repair"})
            fixed = _fix_json(response)
            return json.loads(fixed)


def _normalize_violations(raw: Any) -> list[dict]:
    """Приводит ответ LLM к плоскому списку dict-нарушений.

    Модель может вернуть ``violations`` как вложенный список
    (напр. ``[[{...}, {...}]]``) или с не-dict элементами — без этой
    нормализации последующий ``v.get("rule_id")`` падает (adversarial #3,
    реальный сбой прогона). Рекурсивно «разворачиваем» списки и оставляем
    только dict-элементы.
    """
    result: list[dict] = []
    if not isinstance(raw, list):
        return result
    for item in raw:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, list):
            result.extend(_normalize_violations(item))
    return result


def _analyze_chunk(chunk_data: list[dict], llm: RouterAIProvider, settings: Settings) -> list[dict]:
    tc_json = json.dumps(chunk_data, ensure_ascii=True)
    user_msg = f"Проверь тест-кейсы на соответствие стандартам QA:\n{tc_json}\nВерни JSON."

    logger.info(f"Sending {len(user_msg)} chars to LLM for standards analysis")
    response = llm.chat_completion(
        messages=[
            {"role": "system", "content": STANDARDS_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        model=settings.model_senior,
        max_tokens=settings.standards_max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "standards_chunk",
                "schema": build_json_schema("standards_agent"),
            },
        },
    )
    logger.info(f"LLM response length: {len(response) if response else 0}")

    if not response or not response.strip():
        logger.warning("Empty LLM response for standards chunk, returning empty violations")
        return []

    data = _parse_llm_response(response)
    return _normalize_violations(data.get("violations", []))


def run_standards_agent(
    test_cases: Optional[list[TestCase]] = None,
    requirement_ids: Optional[list[str]] = None,
    settings: Optional[Settings] = None,
    llm: Optional[RouterAIProvider] = None,
) -> StandardsReport:
    if settings is None:
        settings = get_settings()
    if llm is None:
        llm = RouterAIProvider(settings)

    from src.tracing import set_span_output, trace_agent

    chunk_size = settings.agents_chunk_size

    with trace_agent(
        "Standards Agent",
        **{"agent.type": "standards"},
    ) as span:

        if test_cases is None:
            if requirement_ids:
                tc_data = get_test_cases_by_reqs(requirement_ids)
            else:
                tc_data = get_all_test_cases()
            test_cases = [TestCase(**tc) for tc in tc_data]

        if requirement_ids:
            req_set = set(requirement_ids)
            test_cases = [tc for tc in test_cases if tc.req in req_set]

        tc_dicts = _prepare_test_cases_data([tc.model_dump() for tc in test_cases])

        if span is not None:
            span.set_attribute("test_cases.count", len(test_cases))
            span.set_attribute("chunk_size", chunk_size)

        all_violations = []
        chunk_total = (len(tc_dicts) + chunk_size - 1) // chunk_size
        max_iter = settings.standards_max_iterations
        iteration = 0

        for i in range(0, len(tc_dicts), chunk_size):
            # Защита от runaway-цикла (revью T4): жёсткий потолок итераций.
            if max_iter and max_iter > 0 and iteration >= max_iter:
                logger.warning(
                    f"Standards analysis reached max_iterations={max_iter}, stopping early"
                )
                break
            chunk = tc_dicts[i:i + chunk_size]
            logger.info(f"Analyzing standards chunk {iteration + 1}/{chunk_total}")
            chunk_violations = _analyze_chunk(chunk, llm, settings)
            all_violations.extend(chunk_violations)
            iteration += 1

        num_rules = num_active_rules()
        if span is not None:
            span.set_attribute("chunks.total", chunk_total)
            span.set_attribute("violations.total", len(all_violations))
            if num_rules > 0 and test_cases:
                denom = len(test_cases) * num_rules
                compliance_pct = round((denom - len(all_violations)) / denom * 100, 1)
            else:
                compliance_pct = 100.0
            set_span_output(span, {"compliance_percentage": compliance_pct}, mime_type="application/json")

    num_rules = num_active_rules()
    if num_rules > 0 and test_cases:
        total_checks = len(test_cases) * num_rules
        passed_checks = total_checks - len(all_violations)
        compliance = (passed_checks / total_checks * 100) if total_checks > 0 else 100.0
    else:
        if num_rules == 0:
            logger.warning("standards_rules.yaml is empty; compliance set to 100.0")
        compliance = 100.0
    compliance = max(0.0, min(100.0, compliance))

    blocking_rule_ids, auto_fix_rule_ids = rule_classification()
    blocking = [
        f"{v.get('rule_id', '?')}: {v.get('description', '')} in {v.get('test_case_id', '?')}"
        for v in all_violations
        if v.get("rule_id") in blocking_rule_ids
    ]

    auto_fix = list(set(
        f"{v.get('rule_id', '?')}: {v.get('description', '')}"
        for v in all_violations
        if v.get("rule_id") in auto_fix_rule_ids
    ))

    human_review = list(set(
        f"{v.get('rule_id', '?')}: {v.get('description', '')} in {v.get('test_case_id', '?')}"
        for v in all_violations
        if v.get("rule_id") not in auto_fix_rule_ids
    ))

    return StandardsReport(
        compliance_percentage=round(compliance, 1),
        violations=all_violations,
        blocking_violations=blocking,
        auto_fix_available=auto_fix,
        human_review_required=human_review,
    )
