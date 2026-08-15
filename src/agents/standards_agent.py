import json
import logging
import re
from typing import Any, Optional

import json_repair

from src.config import Settings, get_settings
from src.llm_provider import RouterAIProvider
from src.models import StandardsReport, TestCase
from src.tools.sql_tool import get_all_test_cases

logger = logging.getLogger(__name__)

STANDARDS_SYSTEM_PROMPT = """Ты — QA lead, проверяющий соответствие стандартам.

Верни СТРОГО валидный JSON:
{
  "violations": [{"rule_id": str, "severity": "major"|"minor"|"critical", "test_case_id": str, "description": str, "auto_fixable": bool}],
  "blocking_violations": [str],
  "auto_fix_available": [str],
  "human_review_required": [str]
}

Правила:
- QA-TEST-001: ID начинается с "TC-"
- QA-TEST-002: title <= 120 символов, не "Проверить..."
- QA-TEST-003: req не пуст
- QA-TEST-004: description >= 20 символов
- QA-TEST-005: steps без "проверить", "убедиться"
- QA-TEST-006: expected_result без шаблонных фраз
- QA-TEST-008: test_type не пуст
- QA-TEST-009: preconditions не пустые
- QA-TEST-010: нет секретов (token=, password=, user@example.com, abc123)

Блокирующие: QA-TEST-010
Auto-fix: QA-TEST-002, QA-TEST-004, QA-TEST-005, QA-TEST-006, QA-TEST-009
"""


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

    if test_cases is None:
        tc_data = get_all_test_cases()
        test_cases = [TestCase(**tc) for tc in tc_data]

    if requirement_ids:
        req_set = set(requirement_ids)
        test_cases = [tc for tc in test_cases if tc.req in req_set]

    tc_dicts = _prepare_test_cases_data([tc.model_dump() for tc in test_cases])

    CHUNK_SIZE = 10
    all_violations = []

    for i in range(0, len(tc_dicts), CHUNK_SIZE):
        chunk = tc_dicts[i:i + CHUNK_SIZE]
        logger.info(f"Analyzing standards chunk {i // CHUNK_SIZE + 1}/{(len(tc_dicts) + CHUNK_SIZE - 1) // CHUNK_SIZE}")
        chunk_violations = _analyze_chunk(chunk, llm, settings)
        all_violations.extend(chunk_violations)

    total_checks = len(test_cases) * 9
    passed_checks = total_checks - len(all_violations)
    compliance = (passed_checks / total_checks * 100) if total_checks > 0 else 100.0

    blocking = [
        f"{v['rule_id']}: {v['description']} in {v['test_case_id']}"
        for v in all_violations
        if v["rule_id"] == "QA-TEST-010"
    ]

    auto_fix = list(set(
        f"{v['rule_id']}: {v['description']}"
        for v in all_violations
        if v["rule_id"] in {"QA-TEST-002", "QA-TEST-004", "QA-TEST-005", "QA-TEST-006", "QA-TEST-009"}
    ))

    human_review = list(set(
        f"{v['rule_id']}: {v['description']} in {v['test_case_id']}"
        for v in all_violations
        if v["rule_id"] not in {"QA-TEST-002", "QA-TEST-004", "QA-TEST-005", "QA-TEST-006", "QA-TEST-009"}
    ))

    return StandardsReport(
        compliance_percentage=round(compliance, 1),
        violations=all_violations,
        blocking_violations=blocking,
        auto_fix_available=auto_fix,
        human_review_required=human_review,
    )
