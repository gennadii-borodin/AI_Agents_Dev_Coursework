"""
Интеграционные тесты для QA Review Agent.

Тесты проверяют:
1. SQL-инструменты и безопасность
2. Детерминированную логику (без LLM)
3. Генерацию отчётов
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.sql_tool import (
    execute_sql,
    get_all_requirements,
    get_all_test_cases,
    get_tests_without_requirements,
)


@pytest.fixture(scope="session")
def db_url() -> str:
    return os.environ.get("DATABASE_URL", "postgresql://qa_user:qa_password@localhost:5432/qa_review")


@pytest.fixture(scope="session", autouse=True)
def setup_db(db_url: str):
    os.environ["DATABASE_URL"] = db_url
    yield


class TestSQLTool:
    def test_get_all_requirements(self):
        reqs = get_all_requirements()
        assert len(reqs) == 20
        req_ids = {r["requirement_id"] for r in reqs}
        assert "REQ-001" in req_ids
        assert "REQ-020" in req_ids

    def test_get_all_test_cases(self):
        tcs = get_all_test_cases()
        assert len(tcs) == 125
        tc_ids = {tc["test_case_id"] for tc in tcs}
        assert "TC-STORE-0001" in tc_ids
        assert "TC-STORE-0125" in tc_ids

    def test_get_tests_without_requirements(self):
        unlinked = get_tests_without_requirements()
        assert isinstance(unlinked, list)

    def test_sql_security_forbidden_keywords(self):
        result = execute_sql("DROP TABLE requirements")
        assert result["error"] is not None
        assert "Forbidden" in result["error"]

    def test_sql_security_valid_select(self):
        result = execute_sql("SELECT COUNT(*) as cnt FROM requirements")
        assert result["error"] is None
        assert result["results"][0]["cnt"] == 20


class TestReportGeneration:
    def test_reports_structure(self):
        from src.models import CoverageReport, DesignReport, StandardsReport

        coverage = CoverageReport(
            total_coverage=85.0,
            critical_coverage=100.0,
            matrix=[],
            uncovered_requirements=["REQ-017"],
            tests_without_requirements=[],
            indirect_coverage=[],
            gaps=["REQ-017: нет негативных тестов"],
            recommendations=["Добавить тесты для REQ-017"],
            residual_risk="medium",
        )
        assert coverage.total_coverage == 85.0

        design = DesignReport(
            overall_score=70.0,
            techniques_applied=[],
            missing_techniques=["Pairwise Testing"],
            weak_tests=[{"test_case_id": "TC-001", "reason": "no oracle"}],
            duplicate_tests=[],
            recommendations=["Добавить pairwise тесты"],
            test_scores=[],
        )
        assert design.overall_score == 70.0
        assert len(design.missing_techniques) == 1

        standards = StandardsReport(
            compliance_percentage=90.0,
            violations=[{"rule_id": "QA-TEST-003", "severity": "major", "test_case_id": "TC-001", "description": "no req link", "auto_fixable": False}],
            blocking_violations=[],
            auto_fix_available=[],
            human_review_required=[],
        )
        assert standards.compliance_percentage == 90.0
        assert len(standards.violations) == 1

    def test_markdown_generation(self):
        from src.models import CoverageReport, DesignReport, ReviewState, StandardsReport
        from src.report import (
            generate_coverage_markdown,
            generate_design_markdown,
            generate_standards_markdown,
            generate_summary_markdown,
        )

        coverage = CoverageReport(
            total_coverage=85.0,
            critical_coverage=100.0,
            matrix=[{"requirement_id": "REQ-001", "title": "Test", "category": "functional", "priority": "High", "weight": 2, "covered": True, "test_count": 3, "test_types": ["Functional"]}],
            uncovered_requirements=[],
            tests_without_requirements=[],
            indirect_coverage=[],
            gaps=[],
            recommendations=[],
            residual_risk="low",
        )
        md = generate_coverage_markdown(coverage)
        assert "# Отчёт по покрытию требований" in md
        assert "85.0%" in md

        design = DesignReport(
            overall_score=70.0,
            techniques_applied=[{"technique": "Boundary Value Analysis", "coverage": "partial"}],
            missing_techniques=[],
            weak_tests=[],
            duplicate_tests=[],
            recommendations=[],
            test_scores=[],
        )
        md = generate_design_markdown(design)
        assert "# Отчёт по качеству тест-дизайна" in md
        assert "70.0" in md

        standards = StandardsReport(
            compliance_percentage=90.0,
            violations=[],
            blocking_violations=[],
            auto_fix_available=[],
            human_review_required=[],
        )
        md = generate_standards_markdown(standards)
        assert "# Отчёт по соответствию стандартам QA" in md
        assert "90.0%" in md

        state = ReviewState(
            user_query="test",
            coverage_report=coverage,
            design_report=design,
            standards_report=standards,
        )
        md = generate_summary_markdown(state)
        assert "# Сводный отчёт" in md


class TestGraphRouting:
    def test_router_scenarios(self):
        from src.graph import route_request
        from src.models import ReviewState

        test_cases = [
            ("провести полное ревью", "full_review", ["coverage", "design", "standards"]),
            ("проверить покрытие требований", "coverage_review", ["coverage"]),
            ("оценить дизайн тестов", "design_review", ["design"]),
            ("проверить соответствие стандартам", "standards_review", ["standards"]),
            ("найти тесты без требований", "find_unlinked_tests", []),
            ("проверить покрытие REQ-001", "requirement_coverage", ["coverage", "design", "standards"]),
        ]

        for query, expected_scenario, expected_agents in test_cases:
            state = ReviewState(user_query=query)
            result = route_request(state)
            assert result.scenario == expected_scenario, f"Failed for query: {query}"
            assert result.agents_to_run == expected_agents, f"Failed agents for query: {query}"
