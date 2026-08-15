"""Детерминированный статический валидатор тест-кейсов (revью T6, Этап 5).

Заменяет опору design-агента на LLM-гипотезы о «валидности/запускаемости»
тестов реальной статической проверкой структуры и заполненности полей.
Работает без LLM и БД, поэтому дешёв и воспроизводим.
"""

from typing import Optional

_PLACEHOLDERS = {"", "не определены", "не определено", "none", "null", "todo", "tbd", "xxx", "-"}


def _is_empty(value: Optional[str]) -> bool:
    return value is None or value.strip().lower() in _PLACEHOLDERS


def _unbalanced_brackets(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in text or "":
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return True
            stack.pop()
    return bool(stack)


def validate_test_cases(
    test_cases: list[dict],
    known_requirement_ids: Optional[set[str]] = None,
) -> dict:
    """Возвращает детерминированные находки по каждому тест-кейсу.

    findings: list[dict] с ключами test_case_id / issues (список строк-кодов).
    checked: число проверенных ТК.
    """
    findings: list[dict] = []

    for tc in test_cases or []:
        issues: list[str] = []
        tc_id = tc.get("test_case_id", "?")

        if _is_empty(tc.get("preconditions")):
            issues.append("empty_preconditions")
        if _is_empty(tc.get("test_data")):
            issues.append("empty_test_data")
        if _is_empty(tc.get("expected_result")):
            issues.append("empty_expected_result")
        if _is_empty(tc.get("steps")):
            issues.append("empty_steps")
        if _unbalanced_brackets(tc.get("steps", "")):
            issues.append("unbalanced_brackets_in_steps")
        if _is_empty(tc.get("req")) and _is_empty(tc.get("requirement_id")):
            issues.append("missing_requirement_link")

        if known_requirement_ids:
            req = (tc.get("req") or tc.get("requirement_id") or "").strip().upper()
            if req and req not in known_requirement_ids:
                issues.append("unknown_requirement_link")

        for field in ("preconditions", "test_data", "expected_result", "steps", "description"):
            if (tc.get(field) or "").strip().lower() in _PLACEHOLDERS:
                issues.append(f"placeholder_in_{field}")

        if issues:
            findings.append({"test_case_id": tc_id, "issues": issues})

    # Детекция дубликатов по (req, title).
    groups: dict[tuple, list[str]] = {}
    for tc in test_cases or []:
        key = (tc.get("req"), tc.get("title"))
        groups.setdefault(key, []).append(tc.get("test_case_id", "?"))

    for ids in groups.values():
        if len(ids) > 1:
            for tid in ids:
                existing = next((f for f in findings if f["test_case_id"] == tid), None)
                if existing is None:
                    findings.append({"test_case_id": tid, "issues": ["duplicate_test_case"]})
                elif "duplicate_test_case" not in existing["issues"]:
                    existing["issues"].append("duplicate_test_case")

    return {"findings": findings, "checked": len(test_cases or [])}
