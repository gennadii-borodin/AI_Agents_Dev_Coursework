"""Тесты детерминированного статического валидатора тест-кейсов (revью T6)."""

from src.tools.code_validator import validate_test_cases


def _tc(**kw) -> dict:
    base = {
        "test_case_id": "TC-1",
        "req": "REQ-001",
        "title": "Базовый",
        "description": "d",
        "preconditions": "Пользователь авторизован",
        "test_data": "login=admin",
        "steps": "1. Открыть (2. Нажать)",
        "expected_result": "Успех",
        "priority": "high",
        "test_type": "functional",
        "design_quality": "ok",
        "qa_review": "ok",
        "review_comment": "",
    }
    base.update(kw)
    return base


def test_detects_empty_fields():
    tc = _tc(
        preconditions="Не определены",
        test_data="",
        expected_result="не определено",
        steps="",
    )
    res = validate_test_cases([tc])
    issues = res["findings"][0]["issues"]
    assert "empty_preconditions" in issues
    assert "empty_test_data" in issues
    assert "empty_expected_result" in issues
    assert "empty_steps" in issues


def test_detects_unbalanced_brackets():
    tc = _tc(steps="1. Открыть (нажать")
    res = validate_test_cases([tc])
    assert "unbalanced_brackets_in_steps" in res["findings"][0]["issues"]


def test_detects_duplicate_by_req_and_title():
    a = _tc(test_case_id="TC-A", req="REQ-001", title="Одинаковый")
    b = _tc(test_case_id="TC-B", req="REQ-001", title="Одинаковый")
    res = validate_test_cases([a, b])
    ids = {f["test_case_id"] for f in res["findings"]}
    assert {"TC-A", "TC-B"} <= ids
    assert any(
        "duplicate_test_case" in f["issues"]
        for f in res["findings"]
    )


def test_detects_unknown_requirement_link():
    tc = _tc(req="REQ-999")
    res = validate_test_cases([tc], known_requirement_ids={"REQ-001"})
    assert "unknown_requirement_link" in res["findings"][0]["issues"]


def test_clean_test_case_has_no_findings():
    res = validate_test_cases([_tc()])
    assert res["findings"] == []
    assert res["checked"] == 1
