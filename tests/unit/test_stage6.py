"""Tests for Stage 6: golden dataset + eval harness + load/concurrency.

E6.1: эталонный набор запросов (golden_dataset.json).
E6.2: eval-нагрузка прогоняет граф по golden-набору, фиксирует качество/латентность.
E6.3: нагрузочный тест — параллельный прогон N ревью (валидирует параллелизм
из Этапа 4) без падений и с сохранением качества отчётов.
"""

from concurrent.futures import ThreadPoolExecutor

from eval.run_eval import assert_quality, load_golden, run_eval, summarize
from src.graph import build_graph
from src.models import ReviewState
from tests.integration.helpers import ScriptedLLM, apply_llm_patch


def _build_mock_graph(monkeypatch):
    apply_llm_patch(monkeypatch, ScriptedLLM())
    return build_graph()


def test_eval_golden_dataset_passes_quality(monkeypatch, isolate_services):
    g = _build_mock_graph(monkeypatch)
    golden = load_golden()
    rows = run_eval(g, golden)
    assert len(rows) == len(golden)

    violations = assert_quality(golden, rows)
    assert violations == [], f"quality violations: {violations}"

    # на полных сценариях все три отчёта присутствуют
    for r in rows:
        if r["scenario"] == "full_review":
            assert r["coverage_report"] and r["design_report"] and r["standards_report"]
        assert r["latency_s"] >= 0.0


def test_eval_summary_aggregates_latency(monkeypatch, isolate_services):
    g = _build_mock_graph(monkeypatch)
    golden = load_golden()
    rows = run_eval(g, golden)
    s = summarize(rows)
    assert s["total"] == len(golden)
    assert s["latency_mean_s"] >= 0.0
    assert s["latency_min_s"] <= s["latency_max_s"]


def test_load_concurrent_reviews(monkeypatch, isolate_services):
    """Параллельный прогон нескольких ревью — без падений, отчёты сохраняются."""
    g = _build_mock_graph(monkeypatch)
    queries = [
        "провести полное ревью всех тестов",
        "проверить покрытие требований",
        "оценить дизайн тестов",
        "проверить соответствие стандартам",
        "найти тесты без требований unlinked",
    ]

    def _run(q: str) -> dict:
        return g.invoke(ReviewState(user_query=q))

    with ThreadPoolExecutor(max_workers=len(queries)) as ex:
        results = list(ex.map(_run, queries))

    assert len(results) == len(queries)
    for res in results:
        # маршрутизация и хотя бы один осмысленный результат присутствуют
        assert res.get("scenario")
        assert res.get("agents_to_run") is not None
