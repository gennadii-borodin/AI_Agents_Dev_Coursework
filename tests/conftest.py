"""Переиспользуемые фикстуры для интеграционных тестов.

Изоляция от внешних сервисов обеспечивается двумя фикстурами:
- ``isolate_services`` — подменяет ``execute_sql`` in-memory датасетом и
  заглушкой эмбеддинга (нет обращений к PostgreSQL / routerai.ru).
- ``patch_llm`` — подменяет ``RouterAIProvider`` детерминированным ``ScriptedLLM``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    SAMPLE_REQUIREMENTS,
    SAMPLE_TEST_CASES,
    ScriptedLLM,
    apply_llm_patch,
    make_fake_execute_sql,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Дамми-окружение + сброс кэша Settings (без обращений к сети/БД)."""
    monkeypatch.setenv("ROUTER_AI_API_KEY", "test-router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    get_settings.cache_clear()


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def isolate_services(monkeypatch):
    """Изолирует SQL и эмбеддинг от внешних сервисов."""
    fake = make_fake_execute_sql(SAMPLE_REQUIREMENTS, SAMPLE_TEST_CASES)
    monkeypatch.setattr("src.tools.sql_tool.execute_sql", fake)
    import src.embedding as emb

    def _fake_embed(self, text, model=None):
        return [0.0] * 8

    monkeypatch.setattr(emb.EmbeddingProvider, "embed_text", _fake_embed)
    return fake


@pytest.fixture
def patch_llm(monkeypatch, scripted_llm: ScriptedLLM) -> ScriptedLLM:
    apply_llm_patch(monkeypatch, scripted_llm)
    return scripted_llm


@pytest.fixture
def app_graph(patch_llm, isolate_services):
    from src.graph import build_graph

    return build_graph()


@pytest.fixture
def app_graph_ckpt(patch_llm, isolate_services):
    from langgraph.checkpoint.memory import MemorySaver

    from src.graph import build_graph

    return build_graph(checkpointer=MemorySaver())
