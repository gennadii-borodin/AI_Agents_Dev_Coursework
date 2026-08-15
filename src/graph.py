import json
import logging
import re
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from src.agents.coverage_agent import run_coverage_agent
from src.agents.design_agent import run_design_agent
from src.agents.standards_agent import run_standards_agent
from src.config import Settings, get_settings
from src.llm_provider import RouterAIProvider
from src.models import ReviewState
from src.tools.sql_tool import (
    get_all_requirements,
    get_all_test_cases,
    get_tests_without_requirements,
)

logger = logging.getLogger(__name__)


def route_request(state: ReviewState, settings: Optional[Settings] = None) -> ReviewState:
    if settings is None:
        settings = get_settings()
    llm = RouterAIProvider(settings)

    query = state.user_query.lower()

    requirement_ids = re.findall(r"REQ-\d+", query.upper())

    scenario = "full_review"
    if any(w in query for w in ["покрытие", "coverage", "req-"]):
        if requirement_ids:
            scenario = "requirement_coverage"
        else:
            scenario = "coverage_review"
    elif any(w in query for w in ["дизайн", "design", "качество"]):
        scenario = "design_review"
    elif any(w in query for w in ["стандарт", "standard", "соответстви"]):
        scenario = "standards_review"
    elif any(w in query for w in ["без требован", "unlinked", "без req"]):
        scenario = "find_unlinked_tests"
    elif any(w in query for w in ["полное", "full", "всё", "комплексн"]):
        scenario = "full_review"

    agents_map = {
        "full_review": ["coverage", "design", "standards"],
        "coverage_review": ["coverage"],
        "design_review": ["design"],
        "standards_review": ["standards"],
        "requirement_coverage": ["coverage", "design"],
        "find_unlinked_tests": [],
    }

    state.scenario = scenario
    state.requirement_ids = requirement_ids if requirement_ids else None
    state.agents_to_run = agents_map.get(scenario, ["coverage"])

    logger.info(f"Routed query to scenario={scenario}, agents={state.agents_to_run}")
    return state


def run_coverage_node(state: ReviewState) -> ReviewState:
    logger.info("Running Coverage Agent")
    state.current_step = "coverage_analysis"
    report = run_coverage_agent(
        requirement_ids=state.requirement_ids,
    )
    state.coverage_report = report
    return state


def run_design_node(state: ReviewState) -> ReviewState:
    logger.info("Running Design Agent")
    state.current_step = "design_analysis"
    report = run_design_agent(
        requirement_ids=state.requirement_ids,
    )
    state.design_report = report
    return state


def run_standards_node(state: ReviewState) -> ReviewState:
    logger.info("Running Standards Agent")
    state.current_step = "standards_analysis"
    report = run_standards_agent(
        requirement_ids=state.requirement_ids,
    )
    state.standards_report = report
    return state


def run_find_unlinked_node(state: ReviewState) -> ReviewState:
    logger.info("Finding unlinked tests")
    state.current_step = "find_unlinked_tests"
    unlinked = get_tests_without_requirements()
    state.sql_results["unlinked_tests"] = unlinked
    return state


def build_graph() -> StateGraph:
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

    workflow.add_edge("coverage", "design")
    workflow.add_edge("design", "standards")
    workflow.add_edge("standards", END)
    workflow.add_edge("find_unlinked", END)

    return workflow.compile()
