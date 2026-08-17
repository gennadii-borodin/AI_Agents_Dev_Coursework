import json
import logging
from typing import Optional

from src.config import Settings, get_settings
from src.json_utils import parse_json_response
from src.llm_provider import RouterAIProvider
from src.models import DesignReport, Requirement, TestCase
from src.prompts import build_agent_system_prompt, build_json_schema
from src.tools.sql_tool import (
    get_all_requirements,
    get_all_test_cases,
    get_requirements_by_ids,
    get_test_cases_by_reqs,
)

logger = logging.getLogger(__name__)


DESIGN_SYSTEM_PROMPT = build_agent_system_prompt("design_agent")


def _prepare_requirements_data(requirements: list[dict]) -> list[dict]:
    return [
        {
            "requirement_id": r["requirement_id"],
            "title": r["title"],
            "requirement_text": r["requirement_text"][:150],
            "category": r["category"],
            "priority": r["priority"],
        }
        for r in requirements
    ]


def _prepare_test_cases_data(test_cases: list[dict]) -> list[dict]:
    result = []
    for tc in test_cases:
        result.append({
            "id": tc["test_case_id"],
            "req": tc["req"],
            "title": tc["title"][:50],
            "type": tc["test_type"],
            "quality": tc["design_quality"],
            "review": tc["qa_review"],
            "precond_empty": (
                not tc.get("preconditions")
                or tc["preconditions"].strip() in ("", "Не определены")
            ),
            "data_empty": (
                not tc.get("test_data")
                or tc["test_data"].strip() in ("", "Не определены")
            ),
        })
    return result


def run_design_agent(
    requirements: Optional[list[Requirement]] = None,
    test_cases: Optional[list[TestCase]] = None,
    requirement_ids: Optional[list[str]] = None,
    settings: Optional[Settings] = None,
    llm: Optional[RouterAIProvider] = None,
) -> DesignReport:
    if settings is None:
        settings = get_settings()
    if llm is None:
        llm = RouterAIProvider(settings)

    from src.tracing import set_span_output, trace_agent

    with trace_agent(
        "Design Agent",
        **{"agent.type": "design"},
    ) as span:

        if requirements is None:
            if requirement_ids:
                req_data = get_requirements_by_ids(requirement_ids)
            else:
                req_data = get_all_requirements()
            requirements = [Requirement(**r) for r in req_data]

        if test_cases is None:
            if requirement_ids:
                tc_data = get_test_cases_by_reqs(requirement_ids)
            else:
                tc_data = get_all_test_cases()
            test_cases = [TestCase(**tc) for tc in tc_data]

        if requirement_ids:
            req_set = set(requirement_ids)
            test_cases = [tc for tc in test_cases if tc.req in req_set]

        if span is not None:
            span.set_attribute("requirements.count", len(requirements))
            span.set_attribute("test_cases.count", len(test_cases))

        req_data = _prepare_requirements_data([r.model_dump() for r in requirements])
        tc_data = _prepare_test_cases_data([tc.model_dump() for tc in test_cases])

        req_json = json.dumps(req_data, ensure_ascii=False)
        tc_json = json.dumps(tc_data, ensure_ascii=False)

        # Детерминированная статическая валидация (revью T6, Этап 5): вместо
        # LLM-гипотез о «валидности» тестов — реальные находки по структуре.
        # Передаём их модели как достоверные факты, снижая долю галлюцинаций.
        from src.tools.code_validator import validate_test_cases

        known_req_ids = {r["requirement_id"].upper() for r in req_data}
        static = validate_test_cases([tc.model_dump() for tc in test_cases], known_req_ids)
        static_lines = [
            f"- {f['test_case_id']}: {', '.join(f['issues'])}"
            for f in static["findings"]
        ]
        static_block = "\n".join(static_lines) if static_lines else "(структурных проблем не найдено)"

        user_message = f"""Оцени качество тест-дизайна.

## Требования ({len(req_data)}):
{req_json}

## Тест-кейсы ({len(tc_data)}):
{tc_json}

## Детерминированные находки статического валидатора (достоверны, не гадай):
{static_block}

Верни отчёт о качестве тест-дизайна в формате JSON."""

        logger.info("Calling LLM for design analysis...")

        response = llm.chat_completion(
            messages=[
                {"role": "system", "content": DESIGN_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=settings.model_senior,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "design_report",
                    "schema": build_json_schema("design_agent"),
                },
            },
        )

        if span is not None:
            span.set_attribute("llm.response.length", len(response) if response else 0)

        def _on_repair() -> None:
            if span is not None:
                span.add_event("json_repaired", {"agent": "design", "tool": "json_repair"})

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = parse_json_response(response, on_repair=_on_repair)

        data = _normalize_design(data)
        set_span_output(
            span,
            {"overall_score": data.get("overall_score")},
            mime_type="application/json",
        )
        return DesignReport(**data)


def _normalize_design(data: dict) -> dict:
    """Дозаполняет обязательные ключи, чтобы отчёт рендерился даже при
    обрезанном/неполном JSON-ответе модели (без KeyError в report.py)."""
    if not isinstance(data, dict):
        # LLM вернул не объект (напр. JSON-массив) — бросаем, чтобы узел
        # поймал ошибку и quality_gate повторил агента, вместо тихой
        # генерации пустого отчёта с score=0.
        raise ValueError(
            f"Design LLM returned {type(data).__name__}, expected object"
        )
    data.setdefault("overall_score", 0.0)
    data["techniques_applied"] = [
        {
            "technique": str(t.get("technique", "?")) if isinstance(t, dict) else str(t),
            "coverage": (t.get("coverage", "n/a") if isinstance(t, dict) else "n/a"),
        }
        for t in data.get("techniques_applied", []) or []
        if isinstance(t, dict) or isinstance(t, str)
    ]
    data["weak_tests"] = [
        {
            "test_case_id": (t.get("test_case_id", "?") if isinstance(t, dict) else "?"),
            "reason": (t.get("reason", "") if isinstance(t, dict) else ""),
        }
        for t in data.get("weak_tests", []) or []
        if isinstance(t, dict) or isinstance(t, str)
    ]
    data["test_scores"] = [
        {
            "test_case_id": (t.get("test_case_id", "?") if isinstance(t, dict) else "?"),
            "score": (t.get("score", 0.0) if isinstance(t, dict) else 0.0),
        }
        for t in data.get("test_scores", []) or []
        if isinstance(t, dict) and "test_case_id" in t
    ]
    data.setdefault("missing_techniques", [])
    data.setdefault("duplicate_tests", [])
    data.setdefault("recommendations", [])
    return data
