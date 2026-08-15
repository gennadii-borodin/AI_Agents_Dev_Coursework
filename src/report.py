import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import get_settings
from src.graph import build_graph
from src.models import ReviewState
from src.tracing import (
    get_current_session_id,
    get_run_stats,
    set_span_output,
    trace_run,
    trace_tool,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def generate_coverage_markdown(report: Any) -> str:
    lines = ["# Отчёт по покрытию требований\n"]
    lines.append(f"**Общее покрытие:** {report.total_coverage}%\n")
    lines.append(f"**Покрытие критичных требований:** {report.critical_coverage}%\n")

    lines.append("## Матрица покрытия\n")
    lines.append("| Requirement | Category | Priority | Weight | Covered | Test Count | Test Types |")
    lines.append("|-------------|----------|----------|--------|---------|------------|------------|")
    for m in report.matrix:
        covered_mark = "OK" if m.get("covered") else "FAIL"
        test_count = m.get("test_count", m.get("testCount", 0))
        test_types = m.get("test_types", m.get("testTypes", []))
        if isinstance(test_types, list):
            test_types_str = ", ".join(test_types)
        else:
            test_types_str = str(test_types)
        lines.append(
            f"| {m.get('requirement_id', m.get('requirementId', '?'))} | {m.get('category', '?')} | {m.get('priority', '?')} | "
            f"{m.get('weight', '?')} | {covered_mark} | {test_count} | {test_types_str} |"
        )
    lines.append("")

    if report.uncovered_requirements:
        lines.append("## Требования без тестов\n")
        for req_id in report.uncovered_requirements:
            lines.append(f"- `{req_id}`")
        lines.append("")

    if report.tests_without_requirements:
        lines.append("## Тесты без требований\n")
        for tc_id in report.tests_without_requirements:
            lines.append(f"- `{tc_id}`")
        lines.append("")

    if report.indirect_coverage:
        lines.append("## Косвенное покрытие\n")
        for item in report.indirect_coverage:
            lines.append(f"- `{item.get('requirement_id', '?')}`: {item.get('reason', '')}")
        lines.append("")

    if report.gaps:
        lines.append("## Пробелы и рекомендации\n")
        for gap in report.gaps:
            lines.append(f"- {gap}")
        lines.append("")

    lines.append(f"**Остаточный риск:** {report.residual_risk}\n")
    return "\n".join(lines)


def generate_design_markdown(report: Any) -> str:
    lines = ["# Отчёт по качеству тест-дизайна\n"]
    lines.append(f"**Общая оценка:** {report.overall_score}/100\n")

    if report.techniques_applied:
        lines.append("## Применённые техники\n")
        lines.append("| Техника | Покрытие |")
        lines.append("|---------|----------|")
        for t in report.techniques_applied:
            lines.append(f"| {t.get('technique', '?')} | {t.get('coverage', 'n/a')} |")
        lines.append("")

    if report.missing_techniques:
        lines.append("## Отсутствующие техники\n")
        for m in report.missing_techniques:
            lines.append(f"- {m}")
        lines.append("")

    if report.weak_tests:
        lines.append("## Слабые тесты\n")
        for wt in report.weak_tests:
            lines.append(f"- `{wt.get('test_case_id', '?')}`: {wt.get('reason', '')}")
        lines.append("")

    if report.duplicate_tests:
        lines.append("## Дублирующие тесты\n")
        for dup in report.duplicate_tests:
            dups = dup.get("duplicates", []) if isinstance(dup, dict) else []
            lines.append(f"- `{dup.get('original', '?')}` дублирует: {', '.join(dups)}")
        lines.append("")

    if report.recommendations:
        lines.append("## Рекомендации\n")
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


def generate_standards_markdown(report: Any) -> str:
    lines = ["# Отчёт по соответствию стандартам QA\n"]
    lines.append(f"**Соответствие:** {report.compliance_percentage}%\n")

    if report.violations:
        lines.append("## Нарушения\n")
        lines.append("| Rule ID | Severity | Test Case | Description |")
        lines.append("|---------|----------|-----------|-------------|")
        for v in report.violations:
            lines.append(
                f"| {v.get('rule_id', '?')} | {v.get('severity', '?')} | "
                f"{v.get('test_case_id', '?')} | {v.get('description', '')} |"
            )
        lines.append("")

    if report.blocking_violations:
        lines.append("## 🚫 Блокирующие нарушения\n")
        for bv in report.blocking_violations:
            lines.append(f"- {bv}")
        lines.append("")

    if report.auto_fix_available:
        lines.append("## 🔧 Auto-fix доступен\n")
        for af in report.auto_fix_available:
            lines.append(f"- {af}")
        lines.append("")

    if report.human_review_required:
        lines.append("## 👤 Требуется ручное ревью\n")
        for hr in report.human_review_required:
            lines.append(f"- {hr}")
        lines.append("")

    return "\n".join(lines)


def generate_summary_markdown(state: ReviewState) -> str:
    lines = [
        "# Сводный отчёт QA Review Agent\n",
        f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Запрос:** {state.user_query}\n",
        f"**Сценарий:** {state.scenario}\n",
    ]

    if state.requirement_ids:
        lines.append(f"**Требования:** {', '.join(state.requirement_ids)}\n")

    lines.append("---\n")

    if state.coverage_report:
        cr = state.coverage_report
        lines.append("## Покрытие требований\n")
        lines.append(f"- **Общее:** {cr.total_coverage}%")
        lines.append(f"- **Критичные:** {cr.critical_coverage}%")
        lines.append(f"- **Без тестов:** {len(cr.uncovered_requirements)}")
        lines.append(f"- **Тестов без требований:** {len(cr.tests_without_requirements)}")
        lines.append(f"- **Остаточный риск:** {cr.residual_risk}")
        lines.append("")

    if state.design_report:
        dr = state.design_report
        lines.append("## Качество тест-дизайна\n")
        lines.append(f"- **Оценка:** {dr.overall_score}/100")
        lines.append(f"- **Слабых тестов:** {len(dr.weak_tests)}")
        lines.append(f"- **Дубликатов:** {len(dr.duplicate_tests)}")
        lines.append("")

    if state.standards_report:
        sr = state.standards_report
        lines.append("## Соответствие стандартам\n")
        lines.append(f"- **Compliance:** {sr.compliance_percentage}%")
        lines.append(f"- **Нарушений:** {len(sr.violations)}")
        lines.append(f"- **Блокирующих:** {len(sr.blocking_violations)}")
        lines.append("")

    return "\n".join(lines)


def save_reports(state: ReviewState) -> dict[str, Path]:
    if isinstance(state, dict):
        state = ReviewState(**state)
    with trace_tool(
        "save_reports",
        {
            "has_coverage": state.coverage_report is not None,
            "has_design": state.design_report is not None,
            "has_standards": state.standards_report is not None,
        },
        tool_type="CHAIN",
    ) as span:
        saved = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if state.coverage_report:
            path = REPORTS_DIR / f"report_coverage_{timestamp}.md"
            path.write_text(generate_coverage_markdown(state.coverage_report), encoding="utf-8")
            saved["coverage"] = path

        if state.design_report:
            path = REPORTS_DIR / f"report_design_{timestamp}.md"
            path.write_text(generate_design_markdown(state.design_report), encoding="utf-8")
            saved["design"] = path

        if state.standards_report:
            path = REPORTS_DIR / f"report_standards_{timestamp}.md"
            path.write_text(generate_standards_markdown(state.standards_report), encoding="utf-8")
            saved["standards"] = path

        summary_path = REPORTS_DIR / f"report_summary_{timestamp}.md"
        summary_path.write_text(generate_summary_markdown(state), encoding="utf-8")
        saved["summary"] = summary_path

        if span is not None:
            span.set_attribute("reports.saved", ",".join(saved.keys()))
            set_span_output(span, {"saved": list(saved.keys())}, mime_type="application/json")

    return saved


def run_review(
    user_query: str,
    thread_id: Optional[str] = None,
    checkpointer: Optional[Any] = None,
) -> ReviewState:
    """Запускает ревью запроса.

    Args:
        user_query: текст запроса пользователя.
        thread_id: идентификатор прогона для checkpoint/resume. Если не задан —
            генерируется UUID. Тот же ``thread_id`` + ``checkpointer`` позволяют
            возобновить прогон после сбоя с узла падения (C2).
        checkpointer: LangGraph checkpointer. По умолчанию ``MemorySaver``
            (в памяти процесса; для возобновления между перезапусками используйте
            ``PostgresSaver``).

    При сбое агента в середине прогона частично готовые отчёты всё равно
    сохраняются (блок ``finally``), а состояние — в checkpointer для resume.
    """
    from langgraph.checkpoint.memory import MemorySaver

    settings = get_settings()
    if checkpointer is None:
        checkpointer = MemorySaver()
    graph = build_graph(checkpointer)

    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    scenario = "pending"
    agents = []
    saved: dict[str, Path] = {}
    state: ReviewState = ReviewState(user_query=user_query)
    with trace_run(user_query, "pending", []) as run_span:
        try:
            state = graph.invoke(state, config=config)
        except Exception:
            logger.exception("Graph run failed mid-execution (thread_id=%s)", thread_id)
            # C2: подхватываем последнее сохранённое состояние (частичные отчёты)
            snapshot = graph.get_state(config)
            if snapshot is not None and snapshot.values is not None:
                try:
                    state = ReviewState.model_validate(snapshot.values)
                except Exception:
                    logger.exception("Failed to restore partial state from checkpoint")
            raise
        finally:
            # C2: сохраняем частичные отчёты даже при сбое любого агента
            try:
                saved = save_reports(state)
            except Exception:
                logger.exception("Failed to save partial reports after graph run")

        if run_span is not None:
            def _get(obj, attr, default=None):
                if isinstance(obj, dict):
                    return obj.get(attr, default)
                return getattr(obj, attr, default)

            scenario = _get(state, "scenario") or "unknown"
            agents = _get(state, "agents_to_run") or []
            run_span.set_attribute("qa.scenario", scenario)
            run_span.set_attribute("qa.agents", ",".join(agents))
            run_span.set_attribute("qa.requirement_ids", ",".join(_get(state, "requirement_ids") or []))
            cov = _get(state, "coverage_report")
            des = _get(state, "design_report")
            std = _get(state, "standards_report")
            if cov:
                run_span.set_attribute("qa.coverage_pct", float(cov.total_coverage))
            if des:
                run_span.set_attribute("qa.design_score", float(des.overall_score))
            if std:
                run_span.set_attribute("qa.standards_compliance_pct", float(std.compliance_percentage))
                run_span.set_attribute("qa.violations_count", len(std.violations))

            stats = get_run_stats()
            run_span.set_attribute("qa.llm_calls", stats["llm_calls"])
            run_span.set_attribute("qa.prompt_tokens", stats["prompt_tokens"])
            run_span.set_attribute("qa.completion_tokens", stats["completion_tokens"])
            run_span.set_attribute("qa.total_tokens", stats["prompt_tokens"] + stats["completion_tokens"])
            pricing = settings.model_pricing
            est_cost = 0.0
            for model, toks in stats.get("by_model", {}).items():
                price = pricing.get(model, {})
                est_cost += (
                    toks["prompt"] / 1_000_000 * price.get("input", 0)
                    + toks["completion"] / 1_000_000 * price.get("output", 0)
                )
            run_span.set_attribute("qa.estimated_cost_usd", round(est_cost, 4))

            set_span_output(run_span, {
                "scenario": scenario,
                "session_id": get_current_session_id(),
                "agents": agents,
            }, mime_type="application/json")

            run_span.set_attribute("qa.reports_saved", ",".join(saved.keys()))

    print(f"\nOK: Отчёты сохранены в {REPORTS_DIR}")
    for name, path in saved.items():
        print(f"  - {name}: {path}")
    print(f"  - session: {get_current_session_id()}")

    return state
