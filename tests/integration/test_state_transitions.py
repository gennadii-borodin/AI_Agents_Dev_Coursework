"""Интеграционные тесты взаимодействия узлов графа (state transitions).

Проверяем детерминированные переходы между узлами: как router выбирает
сценарий/набор агентов и как ``_route_next`` последовательно передаёт
управление следующему агенту из ``agents_to_run``.
"""

from __future__ import annotations

import pytest

from src.graph import _fan_out, route_request
from src.models import ReviewState

from .helpers import to_review_state

pytestmark = [pytest.mark.integration]


def _state(**kw) -> ReviewState:
    return ReviewState(user_query="q", **kw)


def test_fan_out_returns_sends_for_requested_agents():
    # Параллельный запуск всех трёх независимых агентов (revью §1/T2).
    state = _state(scenario="full_review", agents_to_run=["coverage", "design", "standards"])
    sends = _fan_out(state)
    assert [s.node for s in sends] == ["coverage", "design", "standards"]


def test_fan_out_respects_agents_to_run_subset():
    state = _state(scenario="coverage_review", agents_to_run=["coverage"])
    sends = _fan_out(state)
    assert [s.node for s in sends] == ["coverage"]


def test_fan_out_find_unlinked_routes_to_find_unlinked():
    # find_unlinked не имеет агентов в плане -> отдельный узел.
    state = _state(scenario="find_unlinked_tests", agents_to_run=[])
    sends = _fan_out(state)
    assert [s.node for s in sends] == ["find_unlinked"]


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
    assert result.design_report is not None
    assert result.coverage_report is None
    assert result.standards_report is None


async def test_graph_reaches_end_without_infinite_loop(app_graph):
    raw = await app_graph.ainvoke(ReviewState(user_query="провести полное ревью"))
    result = to_review_state(raw)
    assert result.scenario == "full_review"
    # Все три независимых агента исполнены параллельно.
    assert result.coverage_report is not None
    assert result.design_report is not None
    assert result.standards_report is not None
