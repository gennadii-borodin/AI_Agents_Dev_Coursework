import json
import logging
import re
from typing import Any, Optional

import json_repair

from src.config import Settings, get_settings
from src.llm_provider import RouterAIProvider
from src.models import DesignReport, Requirement, TestCase
from src.tools.sql_tool import get_all_requirements, get_all_test_cases, get_requirements_by_ids

logger = logging.getLogger(__name__)


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

DESIGN_SYSTEM_PROMPT = """Ты — старший QA-инженер, эксперт по тест-дизайну.

Входные данные уже переданы. Проанализируй их и верни СТРОГО валидный JSON:
{
  "overall_score": float (0-100),
  "techniques_applied": [{"technique": str, "coverage": "partial"}],
  "missing_techniques": [str],
  "weak_tests": [{"test_case_id": str, "reason": str}],
  "duplicate_tests": [{"original": str, "duplicates": [str]}],
  "recommendations": [str],
  "test_scores": [{"test_case_id": str, "score": float}]
}

Вывод должен содержать ТОЛЬКО эти поля. Не включай сырые данные из входных требований или тестов в значения полей.
"""


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
            "precond_empty": not tc.get("preconditions") or tc["preconditions"].strip() in ("", "Не определены"),
            "data_empty": not tc.get("test_data") or tc["test_data"].strip() in ("", "Не определены"),
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

    if requirements is None:
        req_data = get_requirements_by_ids(requirement_ids) if requirement_ids else get_all_requirements()
        requirements = [Requirement(**r) for r in req_data]

    if test_cases is None:
        tc_data = get_all_test_cases()
        test_cases = [TestCase(**tc) for tc in tc_data]

    if requirement_ids:
        req_set = set(requirement_ids)
        test_cases = [tc for tc in test_cases if tc.req in req_set]

    req_data = _prepare_requirements_data([r.model_dump() for r in requirements])
    tc_data = _prepare_test_cases_data([tc.model_dump() for tc in test_cases])

    req_json = json.dumps(req_data, ensure_ascii=False)
    tc_json = json.dumps(tc_data, ensure_ascii=False)

    user_message = f"""Оцени качество тест-дизайна.

## Требования ({len(req_data)}):
{req_json}

## Тест-кейсы ({len(tc_data)}):
{tc_json}

Верни отчёт о качестве тест-дизайна в формате JSON."""

    logger.info("Calling LLM for design analysis...")
    response = llm.chat_completion(
        messages=[
            {"role": "system", "content": DESIGN_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=settings.model_senior,
        temperature=0.1,
        json_mode=True,
    )

    try:
        data = json.loads(response)
        return DesignReport(**data)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error, attempting repair: {e}")
        try:
            repaired = json_repair.repair_json(response, return_objects=True)
            if isinstance(repaired, str):
                data = json.loads(repaired)
            elif isinstance(repaired, dict):
                data = repaired
            else:
                raise ValueError(f"Unexpected type: {type(repaired)}")
            return DesignReport(**data)
        except Exception as e2:
            logger.error(f"Failed to repair JSON: {e2}")
            logger.error(f"Raw response: {response[:500]}")
            raise
