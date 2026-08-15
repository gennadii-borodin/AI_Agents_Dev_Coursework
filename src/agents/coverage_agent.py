import json
import logging
import re
from typing import Any, Optional

import json_repair

from src.config import Settings, get_settings
from src.llm_provider import RouterAIProvider
from src.models import CoverageReport, Requirement, TestCase
from src.tools.rag_tool import rag_search_by_requirement
from src.tools.sql_tool import (
    get_all_requirements,
    get_all_test_cases,
    get_requirements_by_ids,
    get_tests_without_requirements,
)

logger = logging.getLogger(__name__)


def _fix_json(text: str) -> str:
    text = text.strip()
    if not text.startswith("{"):
        idx = text.find("{")
        if idx >= 0:
            text = text[idx:]
    if text.endswith(","):
        text = text[:-1]
    if not text.endswith("}"):
        text = text + "}"
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text

PRIORITY_WEIGHTS = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0.5}

COVERAGE_SYSTEM_PROMPT = """Ты — старший QA-инженер, эксперт по покрытию требований.

Верни СТРОГО валидный JSON со следующей структурой:
{
  "total_coverage": float,
  "critical_coverage": float,
  "matrix": [{"requirement_id": str, "title": str, "category": str, "priority": str, "weight": float, "covered": bool, "test_count": int, "test_types": [str]}],
  "uncovered_requirements": [str],
  "tests_without_requirements": [str],
  "indirect_coverage": [{"requirement_id": str, "reason": str}],
  "gaps": [str],
  "recommendations": [str],
  "residual_risk": "low" | "medium" | "high"
}

Вес требования: Critical=3, High=2, Medium=1, Low=0.5
total_coverage = (сумма весов покрытых требований / сумма весов всех требований) * 100
critical_coverage = (покрытые Critical / все Critical) * 100
indirect_coverage: требования с >6 тестами
gaps: требования без негативных или без позитивных тестов
residual_risk: high если total_coverage<80 или critical_coverage<100, medium если total_coverage<95, иначе low
"""


def _prepare_requirements_data(requirements: list[dict]) -> list[dict]:
    return [
        {
            "requirement_id": r["requirement_id"],
            "title": r["title"],
            "requirement_text": r["requirement_text"][:200],
            "category": r["category"],
            "priority": r["priority"],
        }
        for r in requirements
    ]


def _prepare_test_cases_data(test_cases: list[dict]) -> list[dict]:
    return [
        {
            "test_case_id": tc["test_case_id"],
            "req": tc["req"],
            "title": tc["title"],
            "test_type": tc["test_type"],
            "priority": tc["priority"],
        }
        for tc in test_cases
    ]


def run_coverage_agent(
    requirements: Optional[list[Requirement]] = None,
    test_cases: Optional[list[TestCase]] = None,
    requirement_ids: Optional[list[str]] = None,
    settings: Optional[Settings] = None,
    llm: Optional[RouterAIProvider] = None,
) -> CoverageReport:
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
        requirements = [r for r in requirements if r.requirement_id in req_set]
        test_cases = [tc for tc in test_cases if tc.req in req_set]

    unlinked = get_tests_without_requirements()
    tests_without_requirements = [tc["test_case_id"] for tc in unlinked]

    req_data = _prepare_requirements_data([r.model_dump() for r in requirements])
    tc_data = _prepare_test_cases_data([tc.model_dump() for tc in test_cases])

    req_json = json.dumps(req_data, ensure_ascii=False)
    tc_json = json.dumps(tc_data, ensure_ascii=False)

    user_message = f"""Проанализируй требования и тест-кейсы.

## Требования ({len(req_data)}):
{req_json}

## Тест-кейсы ({len(tc_data)}):
{tc_json}

## Тесты без требований (req пуст или null):
{json.dumps(tests_without_requirements, ensure_ascii=False)}

Верни отчёт о покрытии в формате JSON."""

    logger.info("Calling LLM for coverage analysis...")
    response = llm.chat_completion(
        messages=[
            {"role": "system", "content": COVERAGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=settings.model_senior,
        temperature=0.1,
        json_mode=True,
    )

    try:
        data = json.loads(response)
        return CoverageReport(**data)
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
            return CoverageReport(**data)
        except Exception as e2:
            logger.error(f"Failed to repair JSON: {e2}")
            logger.error(f"Raw response: {response[:500]}")
            raise
