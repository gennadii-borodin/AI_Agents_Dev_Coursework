"""Общие инструменты для интеграционных тестов QA Review Agent.

Важное замечание по мокированию LLM
------------------------------------
В проекте LLM — это собственный провайдер ``RouterAIProvider`` (обёртка над
HTTP-клиентом routerai.ru), а **не** langchain-чат-модель. Поэтому стандартные
``langchain-core`` test utilities (``GenericFakeChatModel`` и т.п.) здесь не
применимы напрямую. Детерминированный «scripted LLM» реализован классом
``ScriptedLLM``, имитирующим тот же интерфейс, что и ``RouterAIProvider``
(``chat_completion`` / ``invoke_with_tools``). Это даёт полную изоляцию от
внешних сервисов и воспроизводимость (scripted responses).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from src.models import (
    CoverageReport,
    DesignReport,
    ReviewState,
    StandardsReport,
)

# ---------------------------------------------------------------------------
# Фикстурный in-memory датасет (стенд-ин для PostgreSQL + pgvector)
# ---------------------------------------------------------------------------
SAMPLE_REQUIREMENTS: list[dict[str, Any]] = [
    {
        "requirement_id": "REQ-001",
        "title": "Авторизация по email",
        "requirement_text": "Пользователь логинится по email и паролю",
        "category": "functional",
        "priority": "Critical",
        "qa_requirements_review": "Passed",
        "rejection_reason": "",
    },
    {
        "requirement_id": "REQ-002",
        "title": "Поиск товаров",
        "requirement_text": "Поиск по названию и категории",
        "category": "functional",
        "priority": "High",
        "qa_requirements_review": "Passed",
        "rejection_reason": "",
    },
    {
        "requirement_id": "REQ-003",
        "title": "Экспорт отчётов",
        "requirement_text": "Выгрузка отчёта в CSV",
        "category": "non-functional",
        "priority": "Low",
        "qa_requirements_review": "Passed",
        "rejection_reason": "",
    },
]

SAMPLE_TEST_CASES: list[dict[str, Any]] = [
    {
        "test_case_id": "TC-STORE-0001",
        "req": "REQ-001",
        "title": "Успешный логин",
        "description": "Логин валидным email",
        "preconditions": "Пользователь зарегистрирован",
        "test_data": "email=a@b.com; password=secret123",
        "steps": "1. Открыть форму 2. Ввести данные 3. Отправить",
        "expected_result": "Вход выполнен, открыт профиль",
        "priority": "High",
        "test_type": "Functional",
        "design_quality": "Good",
        "qa_review": "Passed",
        "review_comment": "",
    },
    {
        "test_case_id": "TC-STORE-0002",
        "req": "REQ-001",
        "title": "Неверный пароль",
        "description": "Логин неверным паролем",
        "preconditions": "Пользователь зарегистрирован",
        "test_data": "email=a@b.com; password=wrong",
        "steps": "1. Открыть форму 2. Ввести неверный пароль 3. Отправить",
        "expected_result": "Ошибка авторизации",
        "priority": "High",
        "test_type": "Negative",
        "design_quality": "Good",
        "qa_review": "Passed",
        "review_comment": "",
    },
    {
        "test_case_id": "TC-STORE-0003",
        "req": "REQ-002",
        "title": "Поиск по названию",
        "description": "Поиск существующего товара",
        "preconditions": "Каталог заполнен",
        "test_data": "query=phone",
        "steps": "1. Ввести запрос 2. Нажать Поиск",
        "expected_result": "Список товаров phone",
        "priority": "Medium",
        "test_type": "Functional",
        "design_quality": "Good",
        "qa_review": "Passed",
        "review_comment": "",
    },
    {
        "test_case_id": "TC-STORE-0004",
        "req": "REQ-003",
        "title": "Экспорт CSV",
        "description": "Выгрузка отчёта",
        "preconditions": "Есть данные",
        "test_data": "format=csv",
        "steps": "1. Нажать Экспорт",
        "expected_result": "Файл скачан",
        "priority": "Low",
        "test_type": "Functional",
        "design_quality": "Good",
        "qa_review": "Passed",
        "review_comment": "",
    },
    {
        "test_case_id": "TC-STORE-0005",
        "req": "",
        "title": "Без привязки",
        "description": "Тест без REQ",
        "preconditions": "",
        "test_data": "password=supersecret",
        "steps": "1. Действие",
        "expected_result": "Результат",
        "priority": "Medium",
        "test_type": "Functional",
        "design_quality": "Poor",
        "qa_review": "Failed",
        "review_comment": "нет связи с требованием",
    },
]

# ---------------------------------------------------------------------------
# Scripted LLM-ответы (детерминированные)
# ---------------------------------------------------------------------------
DEFAULT_RAG: list[dict[str, Any]] = [
    {"id": "TC-STORE-0003", "title": "Поиск по названию", "similarity": 0.91},
    {"id": "TC-STORE-0001", "title": "Успешный логин", "similarity": 0.74},
]

DEFAULT_COVERAGE: str = json.dumps(
    {
        "total_coverage": 0.0,
        "critical_coverage": 0.0,
        "matrix": [
            {
                "requirement_id": "REQ-001",
                "title": "Авторизация по email",
                "category": "functional",
                "priority": "Critical",
                "weight": 3,
                "covered": True,
                "test_count": 2,
                "test_types": ["Functional", "Negative"],
            },
            {
                "requirement_id": "REQ-002",
                "title": "Поиск товаров",
                "category": "functional",
                "priority": "High",
                "weight": 2,
                "covered": True,
                "test_count": 1,
                "test_types": ["Functional"],
            },
            {
                "requirement_id": "REQ-003",
                "title": "Экспорт отчётов",
                "category": "non-functional",
                "priority": "Low",
                "weight": 0.5,
                "covered": False,
                "test_count": 0,
                "test_types": [],
            },
        ],
        "uncovered_requirements": ["REQ-003"],
        "tests_without_requirements": ["TC-STORE-0005"],
        "indirect_coverage": [],
        "gaps": ["REQ-003: нет тестов на экспорт"],
        "recommendations": ["Добавить тесты экспорта REQ-003"],
        "residual_risk": "low",
    },
    ensure_ascii=False,
)

DEFAULT_DESIGN: str = json.dumps(
    {
        "overall_score": 75.0,
        "techniques_applied": [{"technique": "Equivalence Partitioning", "coverage": "partial"}],
        "missing_techniques": ["Boundary Value Analysis"],
        "weak_tests": [{"test_case_id": "TC-STORE-0005", "reason": "нет привязки к требованию"}],
        "duplicate_tests": [],
        "recommendations": ["Добавить BVA"],
        "test_scores": [{"test_case_id": "TC-STORE-0001", "score": 80.0}],
    },
    ensure_ascii=False,
)

DEFAULT_STANDARDS: str = json.dumps(
    {
        "violations": [
            {
                "rule_id": "QA-TEST-010",
                "severity": "critical",
                "test_case_id": "TC-STORE-0005",
                "description": "секрет в test_data",
                "auto_fixable": False,
            },
            {
                "rule_id": "QA-TEST-002",
                "severity": "major",
                "test_case_id": "TC-STORE-0001",
                "description": "неинформативное название",
                "auto_fixable": True,
            },
        ],
        "blocking_violations": [],
        "auto_fix_available": [],
        "human_review_required": [],
    },
    ensure_ascii=False,
)


class ScriptedLLM:
    """Детерминированная замена ``RouterAIProvider`` для тестов.

    Маршрутизирует ответ по тексту system-промпта, поэтому один и тот же
    экземпляр корректно отвечает и роутеру, и всем агентам. Ответы задаются
    заранее (scripted) -> тесты воспроизводимы и не зависят от сети.
    """

    def __init__(
        self,
        *,
        router_response: str | None = None,
        coverage_response: str | None = None,
        design_response: str | None = None,
        standards_response: str | None = None,
        rag_results: list[dict[str, Any]] | None = None,
        default_response: str | None = None,
        fail_router: bool = False,
    ) -> None:
        # ``None`` => динамический «идеальный» роутер (через _regex_route).
        self.router_response = router_response
        self.coverage_response = coverage_response or DEFAULT_COVERAGE
        self.design_response = design_response or DEFAULT_DESIGN
        self.standards_response = standards_response or DEFAULT_STANDARDS
        self.rag_results = DEFAULT_RAG if rag_results is None else rag_results
        self.default_response = default_response or '{"ok": true}'
        self.fail_router = fail_router
        self.calls: list[tuple[str, Any, Any]] = []

    # --- интерфейс, совместимый с RouterAIProvider -------------------------
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        json_mode: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append(("chat", model, messages))
        system = (messages[0].get("content") or "") if messages else ""
        sl = system.lower()
        if "классификатор" in sl:
            if self.fail_router:
                raise RuntimeError("LLM unavailable (simulated outage)")
            if self.router_response is None:
                return self._router_response_for(messages)
            return self.router_response
        if "анализе покрытия требований" in sl:
            return self.coverage_response
        if "тест-дизайна" in sl:
            return self.design_response
        if "стандарт" in sl or "qa lead" in sl:
            return self.standards_response
        return self.default_response

    @staticmethod
    def _router_response_for(messages: list[dict[str, str]]) -> str:
        """Имитирует корректный LLM-роутер, переиспользуя логику _regex_route."""
        user_msg = messages[1].get("content", "") if len(messages) > 1 else ""
        marker = "Классифицируй запрос пользователя:\n"
        q = user_msg.split(marker, 1)[-1] if marker in user_msg else user_msg
        from src.graph import _regex_route

        scenario, ids = _regex_route(q)
        return json.dumps({"scenario": scenario, "requirement_ids": ids}, ensure_ascii=False)

    def invoke_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_iterations: int = 5,
        return_tool_results: bool = False,
        tool_choice: Any = "auto",
    ) -> str:
        self.calls.append(("tools", model, user_message))
        if return_tool_results:
            return json.dumps(self.rag_results, ensure_ascii=False)
        return "tool execution completed"


def apply_llm_patch(monkeypatch, stub: ScriptedLLM) -> None:
    """Подменяет ``RouterAIProvider`` во всех модулях, создающих LLM."""
    import src.agents.coverage_agent as ca
    import src.agents.design_agent as da
    import src.agents.standards_agent as sa
    import src.graph as g
    import src.llm_provider as lp

    for mod in (g, ca, da, sa, lp):
        monkeypatch.setattr(mod, "RouterAIProvider", lambda settings=None: stub)


def make_fake_execute_sql(
    requirements: list[dict[str, Any]], test_cases: list[dict[str, Any]]
) -> Callable[..., dict[str, Any]]:
    """Возвращает fake-реализацию ``src.tools.sql_tool.execute_sql``."""

    def fake(query: str, params=None, settings=None) -> dict[str, Any]:
        q = query.upper()
        if "FROM REQUIREMENTS" in q:
            if params:
                ids = set(params)
                rows = [r for r in requirements if r["requirement_id"] in ids]
            else:
                rows = list(requirements)
            return {"results": rows, "row_count": len(rows), "truncated": False, "error": None}
        if "FROM TEST_CASES" in q:
            if "IS NULL" in q:
                rows = [tc for tc in test_cases if not tc.get("req")]
            elif "REQ IN" in q:
                ids = set(params or [])
                rows = [tc for tc in test_cases if tc.get("req") in ids]
            elif "REQ =" in q:
                rid = (params or [None])[0]
                rows = [tc for tc in test_cases if tc.get("req") == rid]
            else:
                rows = list(test_cases)
            return {"results": rows, "row_count": len(rows), "truncated": False, "error": None}
        return {"results": [], "row_count": 0, "truncated": False, "error": None}

    return fake


# ---------------------------------------------------------------------------
# Валидация структуры выходных данных (schema validation)
# ---------------------------------------------------------------------------
def validate_model(model_cls: type, data: dict[str, Any]):
    """Валидирует dict против pydantic-модели (== JSON Schema контракта вывода)."""
    try:
        return model_cls(**data)
    except Exception as e:  # pragma: no cover - test helper
        raise AssertionError(
            f"{model_cls.__name__} validation failed: {e}\nData: {data}"
        ) from e


def to_review_state(raw: Any) -> ReviewState:
    """LangGraph возвращает состояние как dict; приводим к pydantic-модели.

    ``model_validate`` корректно восстанавливает вложенные отчёты
    (CoverageReport/DesignReport/StandardsReport) из dict.
    """
    if isinstance(raw, ReviewState):
        return raw
    return ReviewState.model_validate(raw)


def assert_review_state_schema(state: ReviewState) -> None:
    """Структурные ассерции итогового состояния (без сравнения текста)."""
    assert isinstance(state, ReviewState)
    if state.coverage_report is not None:
        assert isinstance(state.coverage_report, CoverageReport)
        assert 0.0 <= state.coverage_report.total_coverage <= 100.0
        assert 0.0 <= state.coverage_report.critical_coverage <= 100.0
        assert state.coverage_report.residual_risk in ("low", "medium", "high")
    if state.design_report is not None:
        assert isinstance(state.design_report, DesignReport)
        assert 0.0 <= state.design_report.overall_score <= 100.0
    if state.standards_report is not None:
        assert isinstance(state.standards_report, StandardsReport)
        assert 0.0 <= state.standards_report.compliance_percentage <= 100.0
        assert isinstance(state.standards_report.violations, list)
