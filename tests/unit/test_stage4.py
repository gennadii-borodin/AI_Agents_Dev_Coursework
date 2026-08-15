"""Tests for Stage 4: parallel graph fan-out via Send API (revью §1/T2).

Независимые агенты (coverage/design/standards) запускаются одновременно через
Send, а не последовательно. Каждый узел пишет только свой report-ключ,
поэтому параллельные записи не конфликтуют.
"""

from src.config import Settings
from src.graph import _fan_out, build_graph
from src.models import ReviewState
from tests.integration.helpers import ScriptedLLM, apply_llm_patch


def test_fan_out_returns_one_send_per_requested_agent():
    state = ReviewState(
        user_query="q",
        scenario="full_review",
        agents_to_run=["coverage", "design", "standards"],
    )
    sends = _fan_out(state)
    assert [s.node for s in sends] == ["coverage", "design", "standards"]


def test_fan_out_find_unlinked_is_single_branch():
    state = ReviewState(user_query="q", scenario="find_unlinked_tests", agents_to_run=[])
    sends = _fan_out(state)
    assert [s.node for s in sends] == ["find_unlinked"]


def test_graph_full_review_runs_all_agents(monkeypatch, isolate_services):
    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)
    g = build_graph()
    result = g.invoke(ReviewState(user_query="провести полное ревью всех тестов"))
    assert result["scenario"] == "full_review"
    # Все три независимых агента исполнены параллельно через fan-out.
    assert result["coverage_report"] is not None
    assert result["design_report"] is not None
    assert result["standards_report"] is not None


def test_graph_respects_agents_to_run_subset(monkeypatch, isolate_services):
    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)
    g = build_graph()
    result = g.invoke(ReviewState(user_query="оценить дизайн тестов"))
    assert result["design_report"] is not None
    assert result.get("coverage_report") is None
    assert result.get("standards_report") is None
