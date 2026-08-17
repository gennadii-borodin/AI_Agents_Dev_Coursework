"""Tests for Stage 2: context/tool-output reduction (revью T2, T3, §4, T7, T9)."""

import src.agents.standards_agent as standards_agent
import src.embedding as embedding_mod
import src.graph as graph_mod
import src.tools.sql_tool as sql_tool
from src.models import ReviewState


def test_get_all_test_cases_uses_projection(monkeypatch):
    captured = {}

    def fake_execute_sql(query, params=None, settings=None):
        captured["query"] = query
        return {"results": [], "row_count": 0, "truncated": False, "error": None}

    monkeypatch.setattr(sql_tool, "execute_sql", fake_execute_sql)
    sql_tool.get_all_test_cases()
    q = captured["query"].upper()
    assert "SELECT *" not in q, "SELECT * должен быть заменён проекцией колонок (T2)"
    assert "EMBEDDING" not in q, "embedding-вектор не должен попадать в выборку (T2)"
    assert "TEST_CASE_ID" in q


def test_get_all_requirements_uses_projection(monkeypatch):
    captured = {}

    def fake_execute_sql(query, params=None, settings=None):
        captured["query"] = query
        return {"results": [], "row_count": 0, "truncated": False, "error": None}

    monkeypatch.setattr(sql_tool, "execute_sql", fake_execute_sql)
    sql_tool.get_all_requirements()
    q = captured["query"].upper()
    assert "SELECT *" not in q
    assert "EMBEDDING" not in q


def test_load_data_once_populates_state(monkeypatch):
    monkeypatch.setattr(
        sql_tool,
        "get_all_requirements",
        lambda: [{
            "requirement_id": "REQ-1", "title": "t", "requirement_text": "rt",
            "category": "c", "priority": "High", "qa_requirements_review": "", "rejection_reason": "",
        }],
    )
    monkeypatch.setattr(
        sql_tool,
        "get_all_test_cases",
        lambda: [{
            "test_case_id": "TC-1", "req": "REQ-1", "title": "t", "description": "",
            "preconditions": "", "test_data": "", "steps": "", "expected_result": "",
            "priority": "High", "test_type": "functional", "design_quality": "good",
            "qa_review": "ok", "review_comment": "",
        }],
    )
    state = ReviewState(user_query="cover REQ-1")
    out = graph_mod.load_data_once(state)
    assert len(out.requirements) == 1
    assert len(out.test_cases) == 1


def test_embed_text_is_cached(monkeypatch):
    calls = {"n": 0}

    def fake_uncached(text, model, settings):
        calls["n"] += 1
        return [0.1, 0.2]

    monkeypatch.setattr(embedding_mod, "_embed_uncached", fake_uncached)
    embedding_mod._embed_cache.clear()
    prov = embedding_mod.EmbeddingProvider()
    a = prov.embed_text("hello world")
    b = prov.embed_text("hello world")
    assert a == b
    assert calls["n"] == 1, "повторный вызов должен браться из кэша (§4 ревью)"


def test_standards_prompt_uses_single_source_rules():
    # Единый источник — data/standards_rules.yaml, инжектируется в промпт (T7).
    assert "QA-TEST-010" in standards_agent.STANDARDS_SYSTEM_PROMPT
    assert "data/standards_rules.yaml" in standards_agent.STANDARDS_SYSTEM_PROMPT
    # Старый hardcoded-блок правил удалён из prompts/standards_agent.yaml.
    assert "Должен начинаться с \"TC-\"" not in standards_agent.STANDARDS_SYSTEM_PROMPT
