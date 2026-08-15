"""Тесты метрики compliance (M3): знаменатель из числа правил, а не магическая 9."""

import json

from src.agents.standards_agent import num_active_rules, run_standards_agent
from src.models import TestCase as QATestCase

from tests.integration.helpers import ScriptedLLM, apply_llm_patch


def _make_tcs(n: int) -> list[QATestCase]:
    return [
        QATestCase(
            test_case_id=f"TC-{i}",
            req="REQ-001",
            title="t",
            description="",
            preconditions="",
            test_data="",
            steps="",
            expected_result="",
            priority="Low",
            test_type="Functional",
            design_quality="Good",
            qa_review="Passed",
            review_comment="",
        )
        for i in range(n)
    ]


def test_num_active_rules_returns_rule_count(monkeypatch):
    import src.agents.standards_agent as sa

    rules = [{"id": f"R-{i}"} for i in range(5)]
    monkeypatch.setattr(sa, "load_standards_rules", lambda: {"rules": rules})
    sa.num_active_rules.cache_clear()
    try:
        assert num_active_rules() == 5
    finally:
        sa.num_active_rules.cache_clear()


def test_compliance_denominator_uses_rule_count(monkeypatch):
    import src.agents.standards_agent as sa

    n_rules = 7
    rules = [{"id": f"R-{i}", "blocking": False, "auto_fixable": False} for i in range(n_rules)]
    monkeypatch.setattr(sa, "load_standards_rules", lambda: {"rules": rules})
    sa.num_active_rules.cache_clear()
    sa.rule_classification.cache_clear()

    stub = ScriptedLLM()  # DEFAULT_STANDARDS => 2 нарушения
    apply_llm_patch(monkeypatch, stub)

    tcs = _make_tcs(3)
    report = run_standards_agent(test_cases=tcs)

    total = len(tcs) * n_rules
    expected = round((total - 2) / total * 100, 1)
    assert report.compliance_percentage == expected


def test_compliance_empty_ruleset_is_100(monkeypatch):
    import src.agents.standards_agent as sa

    monkeypatch.setattr(sa, "load_standards_rules", lambda: {"rules": []})
    sa.num_active_rules.cache_clear()
    sa.rule_classification.cache_clear()

    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)

    report = run_standards_agent(test_cases=_make_tcs(3))
    assert report.compliance_percentage == 100.0


def test_compliance_clamped_to_zero(monkeypatch):
    import src.agents.standards_agent as sa

    rules = [{"id": "R-0"}]
    monkeypatch.setattr(sa, "load_standards_rules", lambda: {"rules": rules})
    sa.num_active_rules.cache_clear()
    sa.rule_classification.cache_clear()

    many_violations = json.dumps(
        {
            "violations": [
                {"rule_id": "R-0", "test_case_id": f"TC-{i}", "description": "x"}
                for i in range(10)
            ]
        }
    )
    stub = ScriptedLLM(standards_response=many_violations)
    apply_llm_patch(monkeypatch, stub)

    report = run_standards_agent(test_cases=_make_tcs(3))
    assert 0.0 <= report.compliance_percentage <= 100.0
