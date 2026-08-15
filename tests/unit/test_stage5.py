"""Tests for Stage 5: code-validator integration + agent-failure resilience.

E5.1: code_validator скилл детерминированно валидирует ТК (проверяется в
test_code_validator.py). E5.2: сбой одного агента не прерывает прогон —
partial-отчёты сохраняются, ошибка фиксируется в state.errors. E5.3: design-
агент передаёт находки code_validator в промпт вместо LLM-гипотез.
"""

from src.graph import build_graph
from src.models import ReviewState
from tests.integration.helpers import ScriptedLLM, apply_llm_patch


def test_agent_failure_keeps_partial_results(monkeypatch, isolate_services):
    from src.graph import run_standards_agent

    def boom(*args, **kwargs):
        raise RuntimeError("standards boom (simulated)")

    monkeypatch.setattr("src.graph.run_standards_agent", boom)

    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)

    g = build_graph()
    result = g.invoke(ReviewState(user_query="провести полное ревью"))

    # прогон завершён без исключения; coverage/design готовы
    assert result["coverage_report"] is not None
    assert result["design_report"] is not None
    # упавший standards оставляет partial (None), но ошибка зафиксирована
    assert result.get("standards_report") is None
    assert any("standards" in e for e in (result.get("errors") or []))


def test_design_agent_receives_static_findings(monkeypatch, isolate_services):
    captured = {}

    orig = ScriptedLLM.chat_completion

    def spy(self, messages, *args, **kwargs):
        captured["user"] = messages[-1]["content"]
        return orig(self, messages, *args, **kwargs)

    # apply_llm_patch подменяет RouterAIProvider на экземпляр ScriptedLLM,
    # поэтому шпионим сам класс ScriptedLLM, а не RouterAIProvider.
    monkeypatch.setattr(ScriptedLLM, "chat_completion", spy)
    apply_llm_patch(monkeypatch, ScriptedLLM())

    g = build_graph()
    g.invoke(ReviewState(user_query="оценить дизайн тестов"))

    # находки code_validator попадают в промпт design-агента (E5.3)
    assert "статического валидатора" in captured.get("user", "")
