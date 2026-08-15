"""Интеграционные тесты валидации структуры выходных данных (JSON schema).

Вместо сравнения текста проверяем, что выходные данные агентов и итоговое
состояние удовлетворяют контракту (pydantic-модели = JSON Schema вывода).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.agents.standards_agent import run_standards_agent
from src.models import CoverageReport, DesignReport, StandardsReport
from src.report import (
    generate_coverage_markdown,
    generate_design_markdown,
    generate_standards_markdown,
)

from .helpers import (
    DEFAULT_COVERAGE,
    DEFAULT_DESIGN,
    DEFAULT_STANDARDS,
    ScriptedLLM,
    validate_model,
)

pytestmark = [pytest.mark.integration]


def test_coverage_output_matches_json_schema_contract():
    report = validate_model(CoverageReport, json.loads(DEFAULT_COVERAGE))
    assert isinstance(report, CoverageReport)
    assert 0.0 <= report.total_coverage <= 100.0
    assert report.residual_risk in ("low", "medium", "high")


def test_design_output_matches_json_schema_contract():
    report = validate_model(DesignReport, json.loads(DEFAULT_DESIGN))
    assert isinstance(report, DesignReport)
    assert 0.0 <= report.overall_score <= 100.0
    assert isinstance(report.test_scores, list)


def test_standards_llm_output_structure():
    """LLM возвращает нарушения; compliance_percentage добавляется в коде."""
    data = json.loads(DEFAULT_STANDARDS)
    assert "violations" in data
    required = {"rule_id", "severity", "test_case_id", "description", "auto_fixable"}
    for v in data["violations"]:
        assert required.issubset(v.keys())


def test_standards_full_report_contract():
    """Итоговый отчёт (LLM-вывод + вычисленный compliance) валиден."""
    report = validate_model(
        StandardsReport,
        {**json.loads(DEFAULT_STANDARDS), "compliance_percentage": 95.0},
    )
    assert isinstance(report, StandardsReport)
    assert 0.0 <= report.compliance_percentage <= 100.0


def test_missing_required_field_fails_validation():
    bad = {"matrix": []}  # нет total_coverage / critical_coverage / ...
    with pytest.raises(ValidationError):
        CoverageReport(**bad)


async def test_end_to_end_reports_conform_to_schema(app_graph, scripted_llm):
    from src.models import ReviewState

    from .helpers import to_review_state

    raw = await app_graph.ainvoke(ReviewState(user_query="провести полное ревью"))
    result = to_review_state(raw)

    validate_model(CoverageReport, result.coverage_report.model_dump())
    validate_model(DesignReport, result.design_report.model_dump())
    validate_model(StandardsReport, result.standards_report.model_dump())


def test_markdown_reports_have_expected_structure():
    cov = validate_model(CoverageReport, json.loads(DEFAULT_COVERAGE))
    md = generate_coverage_markdown(cov)
    assert "# Отчёт по покрытию требований" in md
    assert "Общее покрытие" in md

    des = validate_model(DesignReport, json.loads(DEFAULT_DESIGN))
    assert "# Отчёт по качеству тест-дизайна" in generate_design_markdown(des)

    std = validate_model(
        StandardsReport,
        {**json.loads(DEFAULT_STANDARDS), "compliance_percentage": 90.0},
    )
    assert "# Отчёт по соответствию стандартам QA" in generate_standards_markdown(std)


async def test_standards_classifies_blocking_and_auto_fix(isolate_services):
    """Бизнес-логика классификации нарушений воспроизводима и валидна."""
    stub = ScriptedLLM()
    report = run_standards_agent(requirement_ids=["REQ-001"], llm=stub)
    # QA-TEST-010 — blocking & не auto_fix из data/standards_rules.yaml.
    blocking = " ".join(report.blocking_violations)
    assert "QA-TEST-010" in blocking
    # QA-TEST-002 — auto_fixable.
    assert any("QA-TEST-002" in item for item in report.auto_fix_available)
    assert 0.0 <= report.compliance_percentage <= 100.0
