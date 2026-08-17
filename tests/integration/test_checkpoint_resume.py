"""Тесты checkpoint + сохранения частичных отчётов (C2, adversarial #2)."""

from __future__ import annotations

import pytest

from src.models import ReviewState

from .helpers import ScriptedLLM, apply_llm_patch

pytestmark = [pytest.mark.integration]


def test_agent_failure_does_not_abort_review(monkeypatch, isolate_services):
    """Сбой одного агента не прерывает весь прогон (revью T6/C2, Этап 5).

    Падающий Standards оставляет partial-отчёт (None), но coverage/design
    считаются, ошибка фиксируется в state.errors, и граф завершается без
    исключения (adversarial #2 в новой модели устойчивости).
    """
    from src.graph import build_graph, run_standards_agent

    def boom(*args, **kwargs):
        raise RuntimeError("standards boom (simulated)")

    monkeypatch.setattr("src.graph.run_standards_agent", boom)

    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)

    g = build_graph()
    result = g.invoke(ReviewState(user_query="провести полное ревью"))

    # прогон завершён, несмотря на падение standards
    assert result["coverage_report"] is not None
    assert result["design_report"] is not None
    assert result.get("standards_report") is None
    assert any("standards" in e for e in (result.get("errors") or []))


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
