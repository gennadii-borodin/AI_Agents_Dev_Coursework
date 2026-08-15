"""Интеграционные тесты взаимодействия узлов графа (state transitions).

Проверяем детерминированные переходы между узлами: как router выбирает
сценарий/набор агентов и как ``_route_next`` последовательно передаёт
управление следующему агенту из ``agents_to_run``.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END as END_MARKER

from src.graph import _route_next, route_request
from src.models import ReviewState

from .helpers import to_review_state

pytestmark = [pytest.mark.integration]


def _state(**kw) -> ReviewState:
    return ReviewState(user_query="q", **kw)


def test_route_next_cycles_through_agents_in_canonical_order():
    plan = ["coverage", "design", "standards"]
    # Пустой current_step не в плане -> сразу END (defensive).
    assert _route_next(_state(current_step="", agents_to_run=plan)) is END_MARKER
    assert _route_next(_state(current_step="coverage", agents_to_run=plan)) == "design"
    assert _route_next(_state(current_step="design", agents_to_run=plan)) == "standards"
    assert _route_next(_state(current_step="standards", agents_to_run=plan)) is END_MARKER


def test_route_next_respects_agents_to_run_subset():
    # Только design в плане -> после design сразу END.
    assert _route_next(_state(current_step="design", agents_to_run=["design"])) is END_MARKER
    # coverage -> design пропущен -> после coverage сразу standards.
    plan = ["coverage", "standards"]
    assert _route_next(_state(current_step="coverage", agents_to_run=plan)) == "standards"


def test_route_request_maps_query_to_scenario_and_agents(patch_llm, isolate_services):
    cases = [
        ("провести полное ревью", "full_review", ["coverage", "design", "standards"]),
        ("проверить покрытие требований", "coverage_review", ["coverage"]),
        ("оценить дизайн тестов", "design_review", ["design"]),
        ("проверить соответствие стандартам", "standards_review", ["standards"]),
        ("найти тесты без требований unlinked", "find_unlinked_tests", []),
        ("проверить покрытие REQ-001", "requirement_coverage", ["coverage", "design", "standards"]),
    ]
    for query, scenario, agents in cases:
        result = route_request(ReviewState(user_query=query))
        assert result.scenario == scenario, query
        assert result.agents_to_run == agents, query


async def test_graph_executes_only_requested_branch(app_graph):
    # design_review -> единственный узел design, coverage/standards не исполняются.
    raw = await app_graph.ainvoke(ReviewState(user_query="оценить дизайн тестов"))
    result = to_review_state(raw)
    assert result.current_step == "design"
    assert result.design_report is not None
    assert result.coverage_report is None
    assert result.standards_report is None


async def test_graph_reaches_end_without_infinite_loop(app_graph):
    raw = await app_graph.ainvoke(ReviewState(user_query="провести полное ревью"))
    result = to_review_state(raw)
    # После standards _route_next возвращает END -> current_step остаётся standards.
    assert result.current_step == "standards"
    assert result.scenario == "full_review"
