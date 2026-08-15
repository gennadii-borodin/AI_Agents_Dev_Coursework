import json
import logging
import re
from typing import Optional

import json_repair

from src.config import Settings, get_settings
from src.llm_provider import RouterAIProvider
from src.models import StandardsReport, TestCase
from src.prompts import build_agent_system_prompt
from src.tools.sql_tool import get_all_test_cases, get_test_cases_by_reqs
from src.tracing import OTEL_AVAILABLE
from src.tracing import otel_trace as _otel_trace

logger = logging.getLogger(__name__)

STANDARDS_SYSTEM_PROMPT = build_agent_system_prompt("standards_agent")


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
        temperature=0.1,
        json_mode=True,
    )
    logger.info(f"LLM response length: {len(response) if response else 0}")

    if not response or not response.strip():
        logger.warning("Empty LLM response for standards chunk, returning empty violations")
        return []

    data = _parse_llm_response(response)
    return data.get("violations", [])


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

    CHUNK_SIZE = 50

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
            span.set_attribute("chunk_size", CHUNK_SIZE)

        all_violations = []

        for i in range(0, len(tc_dicts), CHUNK_SIZE):
            chunk = tc_dicts[i:i + CHUNK_SIZE]
            chunk_total = (len(tc_dicts) + CHUNK_SIZE - 1) // CHUNK_SIZE
            logger.info(f"Analyzing standards chunk {i // CHUNK_SIZE + 1}/{chunk_total}")
            chunk_violations = _analyze_chunk(chunk, llm, settings)
            all_violations.extend(chunk_violations)

        if span is not None:
            span.set_attribute("chunks.total", (len(tc_dicts) + CHUNK_SIZE - 1) // CHUNK_SIZE)
            span.set_attribute("violations.total", len(all_violations))
            set_span_output(span, {"compliance_percentage": round(
                (len(test_cases) * 9 - len(all_violations)) / (len(test_cases) * 9) * 100
                if test_cases else 100.0, 1)}, mime_type="application/json")

    total_checks = len(test_cases) * 9
    passed_checks = total_checks - len(all_violations)
    compliance = (passed_checks / total_checks * 100) if total_checks > 0 else 100.0

    blocking = [
        f"{v['rule_id']}: {v['description']} in {v['test_case_id']}"
        for v in all_violations
        if v["rule_id"] == "QA-TEST-010"
    ]

    _auto_fix_rules = {"QA-TEST-002", "QA-TEST-004", "QA-TEST-005", "QA-TEST-006", "QA-TEST-009"}
    auto_fix = list(set(
        f"{v['rule_id']}: {v['description']}"
        for v in all_violations
        if v["rule_id"] in _auto_fix_rules
    ))

    human_review = list(set(
        f"{v['rule_id']}: {v['description']} in {v['test_case_id']}"
        for v in all_violations
        if v["rule_id"] not in _auto_fix_rules
    ))

    return StandardsReport(
        compliance_percentage=round(compliance, 1),
        violations=all_violations,
        blocking_violations=blocking,
        auto_fix_available=auto_fix,
        human_review_required=human_review,
    )
