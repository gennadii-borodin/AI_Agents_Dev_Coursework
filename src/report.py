import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.coverage_agent import run_coverage_agent
from src.agents.design_agent import run_design_agent
from src.agents.standards_agent import run_standards_agent
from src.config import get_settings
from src.graph import build_graph
from src.models import ReviewState

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
            lines.append(f"- `{item['requirement_id']}`: {item['reason']}")
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
            lines.append(f"| {t['technique']} | {t['coverage']} |")
        lines.append("")

    if report.missing_techniques:
        lines.append("## Отсутствующие техники\n")
        for m in report.missing_techniques:
            lines.append(f"- {m}")
        lines.append("")

    if report.weak_tests:
        lines.append("## Слабые тесты\n")
        for wt in report.weak_tests:
            lines.append(f"- `{wt['test_case_id']}`: {wt['reason']}")
        lines.append("")

    if report.duplicate_tests:
        lines.append("## Дублирующие тесты\n")
        for dup in report.duplicate_tests:
            lines.append(f"- `{dup['original']}` дублирует: {', '.join(dup['duplicates'])}")
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
                f"| {v['rule_id']} | {v['severity']} | {v['test_case_id']} | {v['description']} |"
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

    return saved


def run_review(user_query: str) -> ReviewState:
    settings = get_settings()
    graph = build_graph()

    state = ReviewState(user_query=user_query)
    state = graph.invoke(state)

    saved = save_reports(state)

    print(f"\nOK: Отчёты сохранены в {REPORTS_DIR}")
    for name, path in saved.items():
        print(f"  - {name}: {path}")

    return state
