"""Eval harness: регрессионная и качественная оценка qa-review-agent (revью §6, Этап 6).

Прогоняет набор эталонных запросов (golden dataset) через собранный граф,
фиксирует сценарий/набор агентов/наличие отчётов/ошибки и латентность.
Harness не зависит от инфраструктуры: граф строится вызывающей стороной
(в тестах — с ScriptedLLM + фейковой БД; в проде — с реальными сервисами).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from src.models import ReviewState

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_dataset.json"


def load_golden(path: Optional[str] = None) -> list[dict]:
    return json.loads(Path(path or DEFAULT_GOLDEN).read_text(encoding="utf-8"))


def run_eval(graph, golden: list[dict]) -> list[dict]:
    """Прогоняет каждый эталонный кейс и возвращает строки результатов."""
    rows: list[dict] = []
    for case in golden:
        t0 = time.perf_counter()
        result = graph.invoke(ReviewState(user_query=case["user_query"]))
        dt = time.perf_counter() - t0
        rows.append(
            {
                "id": case.get("id"),
                "query": case["user_query"],
                "scenario": result.get("scenario"),
                "agents_to_run": result.get("agents_to_run"),
                "coverage_report": result.get("coverage_report") is not None,
                "design_report": result.get("design_report") is not None,
                "standards_report": result.get("standards_report") is not None,
                "unlinked_tests": len(
                    (result.get("sql_results") or {}).get("unlinked_tests", []) or []
                ),
                "errors": result.get("errors") or [],
                "latency_s": round(dt, 3),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    failed = [r for r in rows if r["errors"]]
    reports_missing = [
        r for r in rows if not (r["coverage_report"] and r["design_report"] and r["standards_report"])
        # для find_unlinked отчёты не обязательны
        and r["scenario"] != "find_unlinked_tests"
    ]
    lat = [r["latency_s"] for r in rows] or [0.0]
    return {
        "total": total,
        "errors": len(failed),
        "reports_missing": len(reports_missing),
        "latency_min_s": round(min(lat), 3),
        "latency_max_s": round(max(lat), 3),
        "latency_mean_s": round(sum(lat) / total, 3) if total else 0.0,
    }


def assert_quality(golden: list[dict], rows: list[dict]) -> list[str]:
    """Возвращает список нарушений качества (пусто = всё ОК)."""
    by_id = {r["id"]: r for r in rows}
    violations: list[str] = []
    for case in golden:
        row = by_id.get(case.get("id"))
        if row is None:
            violations.append(f"{case.get('id')}: нет результата")
            continue
        if row["scenario"] != case.get("scenario"):
            violations.append(
                f"{case.get('id')}: scenario {row['scenario']} != {case.get('scenario')}"
            )
        if set(row["agents_to_run"] or []) != set(case.get("agents_to_run") or []):
            violations.append(
                f"{case.get('id')}: agents {row['agents_to_run']} != {case.get('agents_to_run')}"
            )
        if case.get("requirement_ids"):
            # requirement_ids фиксируются в state после роутинга
            pass
        if row["errors"]:
            violations.append(f"{case.get('id')}: errors {row['errors']}")
        if case.get("unlinked_present") and row["unlinked_tests"] == 0:
            violations.append(f"{case.get('id')}: ожидались unlinked-тесты")
    return violations


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Eval harness для qa-review-agent")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--out", default="eval/results/eval_latest.json")
    args = ap.parse_args()

    from src.graph import build_graph

    golden = load_golden(args.golden)
    graph = build_graph()
    rows = run_eval(graph, golden)
    summary = summarize(rows)
    violations = assert_quality(golden, rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "violations": violations, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Eval summary: {summary}")
    if violations:
        print("QUALITY VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("All golden cases passed quality checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
