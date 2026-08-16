"""Интеграционные тесты обработки ошибок и восстановления (error recovery).

Проверяем, что система устойчива к:
- невалидному JSON от LLM (json_repair),
- недоступности LLM-роутера (fallback на regex),
- ошибкам SQL (агенты не падают, возвращают отчёт),
- сбоям RAG-инструмента (graceful degradation).
"""

from __future__ import annotations

import pytest

from src.agents.coverage_agent import run_coverage_agent
from src.agents.design_agent import run_design_agent
from src.graph import route_request
from src.models import CoverageReport, ReviewState

from .helpers import (
    SAMPLE_REQUIREMENTS,
    SAMPLE_TEST_CASES,
    ScriptedLLM,
    apply_llm_patch,
    make_fake_execute_sql,
)

pytestmark = [pytest.mark.integration]


# --- Невалидный JSON от LLM -> json_repair ---------------------------------
MALFORMED_COVERAGE = (
    '{"total_coverage": 80.0, "critical_coverage": 100.0, "matrix": [], '
    '"uncovered_requirements": [], "tests_without_requirements": [], '
    '"indirect_coverage": [], "gaps": [], "recommendations": [], "residual_risk": "low"'
    # намеренно без закрывающей скобки
)


def test_coverage_recovers_from_malformed_json(monkeypatch, isolate_services):
    stub = ScriptedLLM(coverage_response=MALFORMED_COVERAGE)
    report = run_coverage_agent(requirement_ids=["REQ-001"], llm=stub)
    assert isinstance(report, CoverageReport)
    # Агрегаты пересчитываются в коде; матрица пуста -> total_coverage из LLM (80.0).
    assert report.total_coverage == 80.0
    assert report.residual_risk == "low"


def test_design_recovers_from_malformed_json(monkeypatch, isolate_services):
    malformed = (
        '{"overall_score": 60.0, "techniques_applied": [], "missing_techniques": [], '
        '"weak_tests": [], "duplicate_tests": [], "recommendations": [], "test_scores": []'
    )
    stub = ScriptedLLM(design_response=malformed)
    report = run_design_agent(requirement_ids=["REQ-001"], llm=stub)
    assert report.overall_score == 60.0


# --- Недоступность LLM-роутера -> regex fallback ---------------------------
def test_router_asks_rephrase_on_llm_outage(monkeypatch, isolate_services):
    # LLM-роутер недоступен (fail_router) → сценарий не определяется,
    # просим пользователя переформулировать, а не молча подменяем regex.
    stub = ScriptedLLM(fail_router=True)
    apply_llm_patch(monkeypatch, stub)
    state = route_request(ReviewState(user_query="проверь покрытие требований"))
    assert state.scenario == ""
    assert state.agents_to_run == []
    assert any("routing_failed" in e for e in state.errors)
    assert state.unresolved_questions


# --- Ошибки SQL -> агент не падает -----------------------------------------
def test_coverage_resilient_to_sql_error(monkeypatch, scripted_llm):
    def err_sql(query, params=None, settings=None):
        return {"results": [], "row_count": 0, "truncated": False, "error": "connection refused"}

    monkeypatch.setattr("src.tools.sql_tool.execute_sql", err_sql)
    report = run_coverage_agent(requirement_ids=["REQ-001"], llm=scripted_llm)
    # Отчёт построен по данным LLM, БД «упала» -> агент не упал и вернул отчёт.
    assert isinstance(report, CoverageReport)
    assert report.total_coverage is not None


# --- Сбой RAG-инструмента -> graceful degradation --------------------------
class _RagFailingLLM(ScriptedLLM):
    def invoke_with_tools(self, *args, **kwargs):
        raise RuntimeError("rag service down")


def test_coverage_degrades_gracefully_when_rag_fails(monkeypatch, isolate_services):
    stub = _RagFailingLLM()
    report = run_coverage_agent(requirement_ids=["REQ-001"], llm=stub)
    assert isinstance(report, CoverageReport)
    # Несмотря на сбой RAG, анализ покрытия завершён.
    assert report.total_coverage is not None


# --- Полная изоляция: ни БД, ни LLM не должны вызываться по сети -----------
def test_no_network_calls_when_isolated(monkeypatch):
    """Жёсткая гарантия изоляции: любой реальный HTTP/SQL перехватывается."""
    import src.embedding as emb
    import src.llm_provider as lp

    def _block(*a, **k):
        raise AssertionError("network call attempted despite isolation")

    monkeypatch.setattr(emb.EmbeddingProvider, "embed_text", _block)
    monkeypatch.setattr(lp.RouterAIProvider, "_do_request", _block)
    monkeypatch.setattr(
        "src.tools.sql_tool.execute_sql",
        make_fake_execute_sql(SAMPLE_REQUIREMENTS, SAMPLE_TEST_CASES),
    )

    stub = ScriptedLLM()
    apply_llm_patch(monkeypatch, stub)
    report = run_coverage_agent(requirement_ids=["REQ-001"], llm=stub)
    assert isinstance(report, CoverageReport)
