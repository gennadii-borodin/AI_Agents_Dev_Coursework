import json
import logging
import re
from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send

from src.agents.coverage_agent import run_coverage_agent
from src.agents.design_agent import run_design_agent
from src.agents.standards_agent import run_standards_agent
from src.config import Settings, get_settings
from src.llm_provider import RouterAIProvider
from src.models import ReviewState
from src.skills import ToolRegistry

logger = logging.getLogger(__name__)

KNOWN_SCENARIOS = {
    "full_review",
    "coverage_review",
    "design_review",
    "standards_review",
    "requirement_coverage",
    "find_unlinked_tests",
}

# Сообщение, выдаваемое пользователю, когда роутер не смог определить
# сценарий (сбой LLM или недопустимый/неизвестный scenario от модели).
ROUTING_FAILED_MESSAGE = (
    "Не удалось определить сценарий по запросу. "
    "Переформулируйте задачу, например: "
    "'провести полное ревью', 'проверить покрытие REQ-001', "
    "'оценить дизайн тестов', 'проверить стандарты', "
    "'найти тесты без требований'."
)


def _regex_route(query: str) -> tuple[str, list[str]]:
    """Детерминированный запасной роутинг (используется при сбое LLM)."""
    q = query.lower()
    requirement_ids = re.findall(r"REQ-\d+", query.upper())

    scenario = "full_review"
    if any(w in q for w in ["покрытие", "coverage", "req-"]):
        scenario = "requirement_coverage" if requirement_ids else "coverage_review"
    elif any(w in q for w in ["дизайн", "design", "качество"]):
        scenario = "design_review"
    elif any(w in q for w in ["стандарт", "standard", "соответстви"]):
        scenario = "standards_review"
    elif any(w in q for w in ["без требован", "unlinked", "без req"]):
        scenario = "find_unlinked_tests"
    elif any(w in q for w in ["полное", "full", "всё", "комплексн"]):
        scenario = "full_review"
    return scenario, requirement_ids


def _llm_route(query: str, llm: RouterAIProvider, settings: Settings) -> tuple[str, list[str]]:
    """Классификация запроса LLM-роутером (prompts/router.yaml)."""
    from src.prompts import build_agent_system_prompt

    system = build_agent_system_prompt("router")
    user = f"Классифицируй запрос пользователя:\n{query}"
    response = llm.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=settings.model_junior,
        temperature=settings.llm_temperature,
        json_mode=True,
    )
    data = json.loads(response)
    scenario = (data.get("scenario") or "full_review").strip()
    requirement_ids = [str(r) for r in (data.get("requirement_ids") or [])]
    return scenario, requirement_ids


def route_request(state: ReviewState, settings: Optional[Settings] = None) -> ReviewState:
    from src.tracing import set_span_output, trace_agent

    if settings is None:
        settings = get_settings()

    with trace_agent("Router", **{"agent.type": "router"}) as span:

        def _fail(detail: str) -> ReviewState:
            logger.warning(f"Routing failed: {detail}")
            state.scenario = ""
            state.agents_to_run = []
            state.requirement_ids = None
            state.errors.append(f"routing_failed: {detail}")
            state.unresolved_questions.append(ROUTING_FAILED_MESSAGE)
            state.final_answer = ROUTING_FAILED_MESSAGE
            if span is not None:
                span.set_attribute("router.method", "llm-failed")
                span.set_attribute("router.scenario", "unknown")
                set_span_output(
                    span,
                    {"scenario": "unknown", "error": detail},
                    mime_type="application/json",
                )
            logger.info("Routing failed, asking user to rephrase")
            return state

        if settings.router_llm_enabled:
            # Маршрутизация целиком ложится на младшую LLM-модель (источник истины).
            # requirement_ids берутся ТОЛЬКО из ответа LLM.
            try:
                scenario, requirement_ids = _llm_route(state.user_query, RouterAIProvider(settings), settings)
            except Exception as e:
                return _fail(f"LLM-роутер недоступен: {e}")
            if scenario not in KNOWN_SCENARIOS:
                return _fail(f"недопустимый сценарий от LLM: '{scenario}'")
            method = "llm"
            ids = requirement_ids
        else:
            # Legacy-путь: LLM-роутер отключён, используем детерминированный regex.
            scenario, ids = _regex_route(state.user_query)
            method = "regex-disabled"

        agents_map = {
            "full_review": ["coverage", "design", "standards"],
            "coverage_review": ["coverage"],
            "design_review": ["design"],
            "standards_review": ["standards"],
            "requirement_coverage": ["coverage", "design", "standards"],
            "find_unlinked_tests": [],
        }

        state.scenario = scenario
        state.requirement_ids = ids if ids else None
        state.agents_to_run = agents_map.get(scenario, [])

        if span is not None:
            span.set_attribute("router.method", method)
            span.set_attribute("router.scenario", scenario)
            span.set_attribute("router.requirement_ids", ",".join(state.requirement_ids or []))
            span.set_attribute("router.agents", ",".join(state.agents_to_run))
            set_span_output(span, {"scenario": scenario}, mime_type="application/json")

        logger.info(f"Routed query to scenario={scenario}, agents={state.agents_to_run}")
        return state


def _fan_out(state: ReviewState) -> list:
    """Параллельный fan-out независимых агентов через Send API (revью §1/T2).

    Независимые агенты (coverage/design/standards) запускаются одновременно
    вместо последовательной цепочки. Каждый узел пишет ТОЛЬКО свой report-ключ,
    поэтому параллельные записи в state не пересекаются — гонки исключены.
    Сценарий ``find_unlinked_tests`` не имеет агентов в плане, поэтому
    маршрутизируется на отдельный узел.
    """
    if state.scenario == "find_unlinked_tests":
        return [Send("find_unlinked", state)]
    # Неизвестный/пустой scenario (в т.ч. провал роутинга) — агентов не запускаем.
    if state.scenario not in KNOWN_SCENARIOS:
        return []
    branch_order = ("coverage", "design", "standards")
    return [Send(name, state) for name in branch_order if name in (state.agents_to_run or [])]


def run_coverage_node(state: ReviewState) -> dict:
    logger.info("Running Coverage Agent")
    try:
        report = run_coverage_agent(
            requirement_ids=state.requirement_ids,
            requirements=state.requirements or None,
            test_cases=state.test_cases or None,
        )
        return {"coverage_report": report}
    except Exception as e:
        logger.exception("Coverage agent failed")
        return {"errors": [f"coverage: {type(e).__name__}: {e}"]}


def run_design_node(state: ReviewState) -> dict:
    logger.info("Running Design Agent")
    try:
        report = run_design_agent(
            requirement_ids=state.requirement_ids,
            requirements=state.requirements or None,
            test_cases=state.test_cases or None,
        )
        return {"design_report": report}
    except Exception as e:
        logger.exception("Design agent failed")
        return {"errors": [f"design: {type(e).__name__}: {e}"]}


def run_standards_node(state: ReviewState) -> dict:
    logger.info("Running Standards Agent")
    try:
        report = run_standards_agent(
            requirement_ids=state.requirement_ids,
            test_cases=state.test_cases or None,
        )
        return {"standards_report": report}
    except Exception as e:
        logger.exception("Standards agent failed")
        return {"errors": [f"standards: {type(e).__name__}: {e}"]}


def run_find_unlinked_node(state: ReviewState) -> ReviewState:
    logger.info("Finding unlinked tests")
    state.current_step = "find_unlinked_tests"
    registry = ToolRegistry()
    result = registry.execute(
        "sql_query",
        {
            "query": (
                "SELECT test_case_id, req, title, description, preconditions, test_data, steps, "
                "expected_result, priority, test_type, design_quality, qa_review, review_comment "
                "FROM test_cases WHERE req IS NULL OR req = '' ORDER BY test_case_id"
            )
        },
    )
    unlinked = result.get("results", []) if isinstance(result, dict) else []
    state.sql_results["unlinked_tests"] = unlinked
    return state


def load_data_once(state: ReviewState, settings: Optional[Settings] = None) -> ReviewState:
    """Селективная однократная выгрузка данных в state (T3, §3 ревью).

    Гибрид:
    - ``find_unlinked_tests``: целевой SQL выполняет ``run_find_unlinked_node``,
      общая выгрузка не нужна — возвращаем state без обращения к БД;
    - ``requirement_coverage``: грузим только запрошенные требования и их ТК
      (базовое ядро под сценарий), остальное агенты дочитывают сами при
      необходимости (ветка ``if requirements is None`` в агентах);
    - остальные сценарии: полная выгрузка (текущее поведение).
    DB-вызовы обёрнуты в спан ``load_data_once`` для наблюдаемости.
    """
    if settings is None:
        settings = get_settings()
    from src.models import Requirement, TestCase
    from src.tracing import trace_tool
    from src.tools.sql_tool import (
        get_all_requirements,
        get_all_test_cases,
        get_requirements_by_ids,
        get_test_cases_by_reqs,
    )

    req_ids = state.requirement_ids or []

    # find_unlinked_tests: общая выгрузка не требуется (свой целевой SQL в узле).
    if state.scenario == "find_unlinked_tests":
        with trace_tool(
            "load_data_once",
            {"scenario": state.scenario, "skipped": True},
            tool_type="CHAIN",
        ):
            pass
        return state

    with trace_tool(
        "load_data_once",
        {"scenario": state.scenario, "requirement_ids": ",".join(req_ids)},
        tool_type="CHAIN",
    ):
        # requirement_coverage: базовое ядро — только запрошенные REQ + их ТК.
        if state.scenario == "requirement_coverage" and req_ids:
            if not state.requirements:
                state.requirements = [Requirement(**r) for r in get_requirements_by_ids(req_ids)]
            if not state.test_cases:
                state.test_cases = [TestCase(**t) for t in get_test_cases_by_reqs(req_ids)]
            return state

        # Полная выгрузка для остальных сценариев (и requirement_coverage без REQ-id).
        if not state.requirements:
            state.requirements = [Requirement(**r) for r in get_all_requirements()]
        if not state.test_cases:
            state.test_cases = [TestCase(**t) for t in get_all_test_cases()]
    return state


def finalize(state: ReviewState) -> ReviewState:
    """Точка сбора после параллельного fan-out агентов (revью §1/T2).

    Конвергенция веток перед quality_gate.
    """
    logger.info("Finalizing review")
    return state


_REPORT_FIELDS = {
    "coverage": "coverage_report",
    "design": "design_report",
    "standards": "standards_report",
}


def _build_final_answer(state: ReviewState) -> str:
    # При провале роутинга агенты не запускались — возвращаем просьбу
    # переформулировать задачу, а не неинформативное «no reports produced».
    if not state.agents_to_run and state.unresolved_questions:
        return state.unresolved_questions[-1]
    parts: list[str] = []
    if state.coverage_report:
        parts.append(f"Coverage: {state.coverage_report.total_coverage:.0f}% total")
    if state.design_report:
        parts.append(f"Design score: {state.design_report.overall_score:.0f}")
    if state.standards_report:
        parts.append(f"Standards compliance: {state.standards_report.compliance_percentage:.0f}%")
    if state.sql_results.get("unlinked_tests"):
        parts.append(f"Unlinked tests: {len(state.sql_results['unlinked_tests'])}")
    return "; ".join(parts) if parts else "no reports produced"


def quality_gate(state: ReviewState, settings: Optional[Settings] = None) -> ReviewState | Command:
    """QA-валидация и targeted retry (revью §4, Этап 4).

    После параллельного fan-out проверяем, что ожидаемые агенты выдали отчёты.
    Если какой-то упал (partial, report=None) и включён targeted_retry — повторно
    запускаем ТОЛЬКО упавших агентов через Send, без перезапуска всего прогона.
    Число попыток ограничено ``max_retry_attempts`` (защита от петли).
    В финале агрегируем ``final_answer`` и фиксируем частичные сбои.
    """
    if settings is None:
        settings = get_settings()

    state.iteration += 1
    expected = [a for a in ("coverage", "design", "standards") if a in (state.agents_to_run or [])]
    missing = [a for a in expected if getattr(state, _REPORT_FIELDS[a]) is None]

    if missing and settings.targeted_retry_enabled and state.iteration <= settings.max_retry_attempts:
        logger.warning(
            f"quality_gate: повторный запуск агентов {missing} (попытка {state.iteration})"
        )
        return Command(
            goto=[Send(a, state) for a in missing],
            update={"iteration": state.iteration},
        )

    # Финальная агрегация.
    state.final_answer = _build_final_answer(state)
    if missing:
        state.unresolved_questions.append(
            f"partial review: missing reports for {missing}"
        )
    return state


def build_graph(checkpointer=None) -> StateGraph:
    workflow = StateGraph(ReviewState)

    workflow.add_node("router", route_request)
    workflow.add_node("load_data_once", load_data_once)
    workflow.add_node("coverage", run_coverage_node)
    workflow.add_node("design", run_design_node)
    workflow.add_node("standards", run_standards_node)
    workflow.add_node("find_unlinked", run_find_unlinked_node)
    workflow.add_node("finalize", finalize)
    workflow.add_node("quality_gate", quality_gate)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "load_data_once")

    # Параллельный fan-out независимых агентов через Send API (T2, §1 ревью).
    # Каждый агент пишет только свой report-ключ — параллельные записи
    # не пересекаются, гонки исключены. Ветки сходятся в finalize -> quality_gate.
    workflow.add_conditional_edges("load_data_once", _fan_out)

    workflow.add_edge("coverage", "finalize")
    workflow.add_edge("design", "finalize")
    workflow.add_edge("standards", "finalize")
    workflow.add_edge("find_unlinked", "finalize")
    workflow.add_edge("finalize", "quality_gate")
    workflow.add_edge("quality_gate", END)

    return workflow.compile(checkpointer=checkpointer)
