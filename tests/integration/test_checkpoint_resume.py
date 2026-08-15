"""Тесты checkpoint + сохранения частичных отчётов (C2, adversarial #2)."""

from __future__ import annotations

import pytest

from src.models import ReviewState

from .helpers import ScriptedLLM, apply_llm_patch

pytestmark = [pytest.mark.integration]


def test_resume_skips_already_done_agents(monkeypatch, isolate_services):
    """Сбой на Standards не теряет прогон: resume по thread_id не перезапускает
    уже готовые агенты и не делает лишних LLM-вызовов (adversarial #2)."""
    from langgraph.checkpoint.memory import MemorySaver

    from src.graph import build_graph, run_standards_agent

    real_standards = run_standards_agent
    calls = {"n": 0}

    def flaky_standards(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("standards boom (simulated)")
        return real_standards(*args, **kwargs)

    monkeypatch.setattr("src.graph.run_standards_agent", flaky_standards)

    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)

    ckpt = MemorySaver()
    g = build_graph(ckpt)
    config = {"configurable": {"thread_id": "resume-1"}}

    try:
        g.invoke(ReviewState(user_query="провести полное ревью"), config=config)
    except RuntimeError:
        pass

    # из чекпоинта: coverage и design уже посчитаны, standards — нет
    snap = g.get_state(config)
    partial = ReviewState.model_validate(snap.values)
    assert partial.coverage_report is not None
    assert partial.design_report is not None
    assert partial.standards_report is None

    calls_before_resume = len(stub.calls)

    final = g.invoke(None, config=config)
    final_state = ReviewState.model_validate(final)
    assert final_state.standards_report is not None

    # при resume новые LLM-вызовы — только standards (роутер/coverage/design пропущены)
    new_calls = stub.calls[calls_before_resume:]
    new_systems = [c[2][0].get("content", "") for c in new_calls]
    assert not any("покрытия" in s for s in new_systems)
    assert not any("дизайн" in s for s in new_systems)


def test_run_review_saves_partial_reports_on_failure(monkeypatch, isolate_services):
    """При сбое агента частично готовые отчёты всё равно сохраняются (finally)."""
    import src.report as report_mod
    from src.graph import run_standards_agent
    from src.report import run_review

    def boom(*args, **kwargs):
        raise RuntimeError("standards boom")

    monkeypatch.setattr("src.graph.run_standards_agent", boom)

    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)

    before = set(p.name for p in report_mod.REPORTS_DIR.glob("report_coverage_*.md"))

    try:
        run_review("провести полное ревью", thread_id="rv-fail")
    except RuntimeError:
        pass

    after = set(p.name for p in report_mod.REPORTS_DIR.glob("report_coverage_*.md"))
    # coverage-отчёт сохранён, несмотря на падение standards
    assert after - before, "partial coverage report must be saved even on agent failure"
