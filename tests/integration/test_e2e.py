"""E2E-тесты: полный workflow агента с мокированным LLM.

Прогоняем скомпилированный LangGraph целиком (router -> агенты -> отчёты)
через разные сценарии, не обращаясь к реальным LLM/БД.
"""

from __future__ import annotations

import pytest

from src.models import ReviewState

from .helpers import assert_review_state_schema, to_review_state

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


async def test_e2e_full_review_runs_all_agents(app_graph, scripted_llm):
    state = ReviewState(user_query="провести полное ревью всех тестов")
    raw = await app_graph.ainvoke(state)
    result = to_review_state(raw)

    assert result.scenario == "full_review"
    assert set(result.agents_to_run) == {"coverage", "design", "standards"}
    assert result.coverage_report is not None
    assert result.design_report is not None
    assert result.standards_report is not None
    assert_review_state_schema(result)

    # Router + coverage(RAG+analysis) + design + standards(chunks) вызвали LLM.
    chat_calls = [c for c in scripted_llm.calls if c[0] == "chat"]
    tool_calls = [c for c in scripted_llm.calls if c[0] == "tools"]
    assert len(chat_calls) == 4  # router, coverage, design, standards
    assert len(tool_calls) == 1  # RAG в coverage-агенте


async def test_e2e_coverage_review_runs_only_coverage(app_graph):
    state = ReviewState(user_query="проверить покрытие требований")
    raw = await app_graph.ainvoke(state)
    result = to_review_state(raw)

    assert result.scenario == "coverage_review"
    assert result.agents_to_run == ["coverage"]
    assert result.coverage_report is not None
    assert result.design_report is None
    assert result.standards_report is None
    assert_review_state_schema(result)


async def test_e2e_design_review_runs_only_design(app_graph):
    state = ReviewState(user_query="оценить дизайн тестов")
    raw = await app_graph.ainvoke(state)
    result = to_review_state(raw)

    assert result.scenario == "design_review"
    assert result.agents_to_run == ["design"]
    assert result.design_report is not None
    assert result.coverage_report is None
    assert result.standards_report is None


async def test_e2e_standards_review_runs_only_standards(app_graph):
    state = ReviewState(user_query="проверить соответствие стандартам")
    raw = await app_graph.ainvoke(state)
    result = to_review_state(raw)

    assert result.scenario == "standards_review"
    assert result.agents_to_run == ["standards"]
    assert result.standards_report is not None
    assert result.coverage_report is None
    assert result.design_report is None


async def test_e2e_requirement_coverage_filters_by_req_id(app_graph, scripted_llm):
    state = ReviewState(user_query="проверить покрытие REQ-001")
    raw = await app_graph.ainvoke(state)
    result = to_review_state(raw)

    assert result.scenario == "requirement_coverage"
    assert result.requirement_ids == ["REQ-001"]
    assert set(result.agents_to_run) == {"coverage", "design", "standards"}
    assert result.coverage_report is not None
    # Агрегаты пересчитываются в коде, а не берутся из LLM.
    assert result.coverage_report.critical_coverage == 100.0


async def test_e2e_find_unlinked_tests_collects_unlinked(app_graph):
    state = ReviewState(user_query="найти тесты без требований unlinked")
    raw = await app_graph.ainvoke(state)
    result = to_review_state(raw)

    assert result.scenario == "find_unlinked_tests"
    assert result.agents_to_run == []
    unlinked = result.sql_results.get("unlinked_tests", [])
    ids = {row["test_case_id"] for row in unlinked}
    assert "TC-STORE-0005" in ids
