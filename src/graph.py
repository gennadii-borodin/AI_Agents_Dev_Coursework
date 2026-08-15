import json
import logging
import re
from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.types import Send

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
        scenario, requirement_ids = _regex_route(state.user_query)
        # router-LLM избыточен: его scenario отбрасывается, REQ-IDs и так даёт
        # regex. Вызов опционален через флаг (revью §2, Этап 3) — по умолчанию
        # сохраняем поведение, но при router_llm_enabled=False идём без LLM.
        if settings.router_llm_enabled:
            llm = RouterAIProvider(settings)
            try:
                llm_scenario, llm_req_ids = _llm_route(state.user_query, llm, settings)
                # Сценарий остаётся детерминированным (regex) — источник истины.
                # LLM используется только для извлечения requirement_ids; его
                # scenario игнорируется (защита от prompt-injection/переопределения — M2).
                # Невалидный scenario от LLM не должен попасть в state (защита C1).
                if llm_scenario and llm_scenario not in KNOWN_SCENARIOS:
                    logger.warning(
                        f"LLM returned unknown scenario '{llm_scenario}', ignored; "
                        f"keeping regex scenario '{scenario}'"
                    )
                # LLM может не вернуть REQ-идентификаторы — добавляем из регулярки.
                regex_ids = re.findall(r"REQ-\d+", state.user_query.upper())
                merged = list(dict.fromkeys(list(requirement_ids) + llm_req_ids + regex_ids))
                requirement_ids = merged
                if span is not None:
                    span.set_attribute("router.method", "llm")
            except Exception as e:
                logger.warning(f"LLM routing failed, using regex fallback: {e}")
                if span is not None:
                    span.set_attribute("router.method", "regex")
        else:
            if span is not None:
                span.set_attribute("router.method", "regex-disabled")

        agents_map = {
            "full_review": ["coverage", "design", "standards"],
            "coverage_review": ["coverage"],
            "design_review": ["design"],
            "standards_review": ["standards"],
            "requirement_coverage": ["coverage", "design", "standards"],
            "find_unlinked_tests": [],
        }

        state.scenario = scenario
        state.requirement_ids = requirement_ids if requirement_ids else None
        state.agents_to_run = agents_map.get(scenario, ["coverage"])

        if span is not None:
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
    """Однократная выгрузка требований и ТК в state (T3, §3 ревью).

    Устраняет тройную полную выгрузку БД агентами: данные грузятся 1× здесь,
    далее агенты читают из ``state.requirements`` / ``state.test_cases``.
    """
    if settings is None:
        settings = get_settings()
    from src.models import Requirement, TestCase
    from src.tools.sql_tool import get_all_requirements, get_all_test_cases

    if not state.requirements:
        state.requirements = [Requirement(**r) for r in get_all_requirements()]
    if not state.test_cases:
        state.test_cases = [TestCase(**t) for t in get_all_test_cases()]
    return state


def finalize(state: ReviewState) -> ReviewState:
    """Точка сбора после параллельного fan-out агентов (revью §1/T2).

    Место для будущей QA-валидации/итоговой свёртки; пока — точка
    конвергенции веток перед END.
    """
    logger.info("Finalizing review")
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

    workflow.set_entry_point("router")
    workflow.add_edge("router", "load_data_once")

    # Параллельный fan-out независимых агентов через Send API (T2, §1 ревью).
    # Каждый агент пишет только свой report-ключ — параллельные записи
    # не пересекаются, гонки исключены. Ветки сходятся в finalize -> END.
    workflow.add_conditional_edges("load_data_once", _fan_out)

    workflow.add_edge("coverage", "finalize")
    workflow.add_edge("design", "finalize")
    workflow.add_edge("standards", "finalize")
    workflow.add_edge("find_unlinked", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=checkpointer)
