"""Tests for quality_gate + targeted retry (revью §4, Этап 4 — завершение дизайна графа)."""

from src.config import Settings
from src.graph import build_graph, get_settings, run_standards_agent
from src.models import ReviewState
from tests.integration.helpers import ScriptedLLM, apply_llm_patch


def _graph(monkeypatch, retry_enabled: bool = True):
    apply_llm_patch(monkeypatch, ScriptedLLM())
    cfg = Settings(
        router_ai_api_key="k", database_url="d",
        targeted_retry_enabled=retry_enabled, max_retry_attempts=2,
    )
    monkeypatch.setattr("src.graph.get_settings", lambda: cfg)
    return build_graph()


def test_quality_gate_retries_failing_agent(monkeypatch, isolate_services):
    calls = {"n": 0}
    real = run_standards_agent

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 1:
            raise RuntimeError("standards boom (simulated)")
        return real(*args, **kwargs)

    monkeypatch.setattr("src.graph.run_standards_agent", flaky)

    g = _graph(monkeypatch, retry_enabled=True)
    result = g.invoke(ReviewState(user_query="провести полное ревью"))

    # targeted retry поднял упавший standards — отчёт в итоге есть
    assert result.get("standards_report") is not None
    assert calls["n"] == 2
    assert result["final_answer"]


def test_quality_gate_no_retry_when_disabled(monkeypatch, isolate_services):
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("standards boom (simulated)")

    monkeypatch.setattr("src.graph.run_standards_agent", boom)

    g = _graph(monkeypatch, retry_enabled=False)
    result = g.invoke(ReviewState(user_query="провести полное ревью"))

    # retry выключен: standards остаётся partial (None), вызван ровно 1 раз
    assert result.get("standards_report") is None
    assert calls["n"] == 1
    assert any("standards" in q for q in (result.get("unresolved_questions") or []))
    # coverage/design всё равно готовы, final_answer агрегирован
    assert result.get("coverage_report") is not None
    assert result.get("design_report") is not None
    assert result["final_answer"]


def test_quality_gate_populates_final_answer_on_success(monkeypatch, isolate_services):
    g = _graph(monkeypatch, retry_enabled=True)
    result = g.invoke(ReviewState(user_query="провести полное ревью"))
    assert result.get("coverage_report") is not None
    assert result.get("standards_report") is not None
    assert "Coverage:" in result["final_answer"]
