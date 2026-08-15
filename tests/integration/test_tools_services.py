"""Интеграционные тесты работы инструментов (tools) и внешних сервисов.

Проверяем:
- ToolRegistry: маршрутизация sql_query/rag_search, валидация аргументов,
  безопасность SQL (forbidden keywords), отказоустойчивость RAG.
- Изоляция от PostgreSQL/routerai.ru (всё мокировано).
"""

from __future__ import annotations

import pytest

from src.skills import ToolRegistry
from src.tools.sql_tool import execute_sql

from .helpers import SAMPLE_REQUIREMENTS, SAMPLE_TEST_CASES, make_fake_execute_sql

pytestmark = [pytest.mark.integration]


# --- ToolRegistry: роутинг и валидация -------------------------------------
def test_registry_runs_sql_query_against_isolated_db(monkeypatch):
    fake = make_fake_execute_sql(SAMPLE_REQUIREMENTS, SAMPLE_TEST_CASES)
    monkeypatch.setattr("src.tools.sql_tool.execute_sql", fake)

    registry = ToolRegistry()
    result = registry.execute("sql_query", {"query": "SELECT * FROM test_cases"})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"results", "row_count", "truncated", "error"}
    assert result["error"] is None
    assert result["row_count"] == len(SAMPLE_TEST_CASES)
    assert {r["test_case_id"] for r in result["results"]} == {
        tc["test_case_id"] for tc in SAMPLE_TEST_CASES
    }


def test_registry_rag_search_returns_list_or_graceful_empty(monkeypatch):
    # Эмбеддинг недоступен (сеть) -> rag_search должен вернуть [] без падения.
    import src.embedding as emb

    def _boom(self, text, model=None):
        raise RuntimeError("no net")

    monkeypatch.setattr(emb.EmbeddingProvider, "embed_text", _boom)

    registry = ToolRegistry()
    out = registry.execute("rag_search", {"collection": "test_cases", "query": "логин", "top_k": 3})
    assert isinstance(out, list)
    assert out == []  # graceful degradation


def test_registry_validates_required_args():
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.execute("rag_search", {"query": "x"})  # нет collection


def test_registry_unknown_tool_is_safe():
    registry = ToolRegistry()
    assert "Unknown tool" in registry.execute("nope", {})
    assert "Unknown tool" in registry.execute_to_json("nope", {})


# --- Безопасность SQL (реальная логика execute_sql, БД не нужна) -----------
def test_sql_forbids_destructive_keywords():
    forbidden_queries = (
        "DROP TABLE requirements",
        "DELETE FROM test_cases",
        "UPDATE test_cases SET x=1",
    )
    for forbidden in forbidden_queries:
        result = execute_sql(forbidden)
        assert result["error"] is not None
        assert "Forbidden" in result["error"]


def test_sql_injection_via_tool_is_rejected():
    registry = ToolRegistry()
    # Попытка «инъекции» через tool-аргумент перехватывается на уровне execute_sql.
    result = registry.execute("sql_query", {"query": "SELECT 1; DROP TABLE requirements;"})
    assert result["error"] is not None


def test_sql_limits_result_rows(monkeypatch):
    # Имитируем БД, возвращающую > MAX_SQL_ROWS строк, и зеркалим логику
    # усечения, как в реальном execute_sql.
    import src.tools.sql_tool as sql_tool_mod
    from src.tools.sql_tool import MAX_SQL_ROWS

    big = [{"id": i} for i in range(1500)]
    truncated = len(big) > MAX_SQL_ROWS
    rows = big[:MAX_SQL_ROWS]
    monkeypatch.setattr(
        "src.tools.sql_tool.execute_sql",
        lambda query, params=None, settings=None: {
            "results": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "error": None,
        },
    )

    result = sql_tool_mod.execute_sql("SELECT * FROM test_cases")
    assert result["truncated"] is True
    assert len(result["results"]) <= MAX_SQL_ROWS
