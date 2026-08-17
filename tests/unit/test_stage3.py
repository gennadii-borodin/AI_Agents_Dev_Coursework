"""Tests for Stage 3: removal of redundant LLM calls (revью §2, §3, T4).

router-LLM и RAG делаем ОПЦИОНАЛЬНЫМИ (не удаляем полностью) через флаги
конфигурации; для Standards снижаем max_tokens вывода.
"""

from src.agents.coverage_agent import run_coverage_agent
from src.agents.standards_agent import run_standards_agent
from src.config import Settings
from src.graph import ReviewState, route_request
from tests.integration.helpers import (
    ScriptedLLM,
    apply_llm_patch,
    make_fake_execute_sql,
    SAMPLE_REQUIREMENTS,
    SAMPLE_TEST_CASES,
)


def _settings(**overrides) -> Settings:
    base = {"router_ai_api_key": "test"}
    base.update(overrides)
    return Settings(**base)


def test_router_llm_disabled_skips_llm(monkeypatch):
    import src.graph as graph_mod

    calls = {"n": 0}
    orig = graph_mod._llm_route

    def spy(query, llm, settings):
        calls["n"] += 1
        return orig(query, llm, settings)

    monkeypatch.setattr(graph_mod, "_llm_route", spy)
    settings = _settings(router_llm_enabled=False)
    out = route_request(ReviewState(user_query="покрытие REQ-001"), settings)
    assert calls["n"] == 0, "при router_llm_enabled=False LLM-роутер не вызывается"
    assert out.agents_to_run, "regex-роутинг сохраняет работоспособность"


def test_router_llm_enabled_calls_llm(monkeypatch):
    import src.graph as graph_mod

    calls = {"n": 0}
    orig = graph_mod._llm_route

    def spy(query, llm, settings):
        calls["n"] += 1
        return orig(query, llm, settings)

    monkeypatch.setattr(graph_mod, "_llm_route", spy)
    settings = _settings(router_llm_enabled=True)
    route_request(ReviewState(user_query="покрытие REQ-001"), settings)
    assert calls["n"] == 1


def test_rag_disabled_skips_invoke_with_tools(monkeypatch):
    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)
    monkeypatch.setattr(
        "src.tools.sql_tool.execute_sql",
        make_fake_execute_sql(SAMPLE_REQUIREMENTS, SAMPLE_TEST_CASES),
    )
    run_coverage_agent(requirement_ids=["REQ-001"], settings=_settings(rag_enabled=False))
    tool_calls = [c for c in stub.calls if c[0] == "tools"]
    assert tool_calls == [], "при rag_enabled=False RAG-вызовы не должны выполняться"


def test_rag_enabled_keeps_invoke_with_tools(monkeypatch):
    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)
    monkeypatch.setattr(
        "src.tools.sql_tool.execute_sql",
        make_fake_execute_sql(SAMPLE_REQUIREMENTS, SAMPLE_TEST_CASES),
    )
    run_coverage_agent(requirement_ids=["REQ-001"], settings=_settings(rag_enabled=True))
    tool_calls = [c for c in stub.calls if c[0] == "tools"]
    assert tool_calls, "при rag_enabled=True RAG-вызовы сохраняются (поведение по умолчанию)"


class _SpyLLM(ScriptedLLM):
    def chat_completion(self, messages, model=None, temperature=0.1, max_tokens=None,
                        json_mode=False, response_format=None):
        self.last_max_tokens = max_tokens
        return super().chat_completion(
            messages, model, temperature, max_tokens, json_mode, response_format
        )


def test_standards_uses_standards_max_tokens(monkeypatch):
    stub = _SpyLLM()
    apply_llm_patch(monkeypatch, stub)
    monkeypatch.setattr(
        "src.tools.sql_tool.execute_sql",
        make_fake_execute_sql(SAMPLE_REQUIREMENTS, SAMPLE_TEST_CASES),
    )
    settings = _settings()
    run_standards_agent(requirement_ids=["REQ-001"], settings=settings)
    assert getattr(stub, "last_max_tokens", None) == settings.standards_max_tokens
    assert settings.standards_max_tokens < settings.llm_max_tokens, "T4: снижен лимит вывода"
