"""Тесты рендеринга сценария find_unlinked_tests (H1)."""

from src.models import ReviewState
from src.report import generate_summary_markdown, generate_unlinked_tests_markdown
from src.graph import run_find_unlinked_node


SAMPLE_UNLINKED = [
    {"test_case_id": "TC-STORE-0005", "title": "Без привязки"},
    {"test_case_id": "TC-STORE-0009", "title": "Ещё без REQ"},
]


def test_generate_unlinked_tests_markdown_lists_all():
    md = generate_unlinked_tests_markdown(SAMPLE_UNLINKED)
    assert "Тесты без привязки к требованиям" in md
    assert "TC-STORE-0005" in md
    assert "TC-STORE-0009" in md
    assert "**Всего тестов без требования:** 2" in md


def test_generate_unlinked_tests_markdown_empty():
    md = generate_unlinked_tests_markdown([])
    assert "Нет тестов без привязки" in md


def test_summary_includes_unlinked_section():
    state = ReviewState(
        user_query="найти тесты без требований",
        scenario="find_unlinked_tests",
        sql_results={"unlinked_tests": SAMPLE_UNLINKED},
    )
    md = generate_summary_markdown(state)
    assert "Тесты без привязки к требованиям" in md
    assert "TC-STORE-0005" in md


def test_summary_omits_unlinked_when_absent():
    state = ReviewState(user_query="проверь покрытие", scenario="coverage_review")
    md = generate_summary_markdown(state)
    assert "Тесты без привязки к требованиям" not in md


def test_find_unlinked_node_populates_state(isolate_services):
    state = ReviewState(user_query="найти тесты без требований", scenario="find_unlinked_tests")
    result = run_find_unlinked_node(state)
    unlinked = (result.sql_results or {}).get("unlinked_tests")
    assert isinstance(unlinked, list)
    # в SAMPLE_TEST_CASES один тест без req (TC-STORE-0005)
    ids = [tc.get("test_case_id") for tc in unlinked if isinstance(tc, dict)]
    assert "TC-STORE-0005" in ids
