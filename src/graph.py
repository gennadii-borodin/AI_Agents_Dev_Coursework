import json
import logging
import re
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agents.coverage_agent import run_coverage_agent
from src.agents.design_agent import run_design_agent
from src.agents.standards_agent import run_standards_agent
from src.config import Settings, get_settings
from src.llm_provider import RouterAIProvider
from src.models import ReviewState
from src.skills import ToolRegistry

logger = logging.getLogger(__name__)


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
        temperature=0.0,
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
        llm = RouterAIProvider(settings)

        scenario, requirement_ids = _regex_route(state.user_query)
        try:
            llm_scenario, llm_req_ids = _llm_route(state.user_query, llm, settings)
            if llm_scenario:
                scenario = llm_scenario
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


def _route_next(state: ReviewState) -> str:
    """Передаёт управление следующему агенту из agents_to_run (в каноническом порядке)."""
    plan = [a for a in ("coverage", "design", "standards") if a in (state.agents_to_run or [])]
    current = state.current_step
    if current not in plan:
        return END
    idx = plan.index(current)
    return plan[idx + 1] if idx + 1 < len(plan) else END


def run_coverage_node(state: ReviewState) -> ReviewState:
    logger.info("Running Coverage Agent")
    state.current_step = "coverage"
    report = run_coverage_agent(
        requirement_ids=state.requirement_ids,
    )
    state.coverage_report = report
    return state


def run_design_node(state: ReviewState) -> ReviewState:
    logger.info("Running Design Agent")
    state.current_step = "design"
    report = run_design_agent(
        requirement_ids=state.requirement_ids,
    )
    state.design_report = report
    return state


def run_standards_node(state: ReviewState) -> ReviewState:
    logger.info("Running Standards Agent")
    state.current_step = "standards"
    report = run_standards_agent(
        requirement_ids=state.requirement_ids,
    )
    state.standards_report = report
    return state


def run_find_unlinked_node(state: ReviewState) -> ReviewState:
    logger.info("Finding unlinked tests")
    state.current_step = "find_unlinked_tests"
    registry = ToolRegistry()
    result = registry.execute(
        "sql_query",
        {"query": "SELECT * FROM test_cases WHERE req IS NULL OR req = '' ORDER BY test_case_id"},
    )
    unlinked = result.get("results", []) if isinstance(result, dict) else []
    state.sql_results["unlinked_tests"] = unlinked
    return state


def build_graph(checkpointer=None) -> StateGraph:
    workflow = StateGraph(ReviewState)

    workflow.add_node("router", route_request)
    workflow.add_node("coverage", run_coverage_node)
    workflow.add_node("design", run_design_node)
    workflow.add_node("standards", run_standards_node)
    workflow.add_node("find_unlinked", run_find_unlinked_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state.scenario,
        {
            "full_review": "coverage",
            "coverage_review": "coverage",
            "design_review": "design",
            "standards_review": "standards",
            "requirement_coverage": "coverage",
            "find_unlinked_tests": "find_unlinked",
        },
    )

    # Агенты в цепочке определяются state.agents_to_run (а не жёстко заданы),
    # поэтому трасса соответствует metadata qa.agents и не запускает лишних агентов.
    agent_targets = {
        "coverage": "coverage",
        "design": "design",
        "standards": "standards",
        END: END,
    }
    workflow.add_conditional_edges("coverage", _route_next, agent_targets)
    workflow.add_conditional_edges("design", _route_next, agent_targets)
    workflow.add_conditional_edges("standards", _route_next, agent_targets)
    workflow.add_edge("find_unlinked", END)

    return workflow.compile(checkpointer=checkpointer)
