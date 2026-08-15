
from src.graph import _regex_route, route_request
from src.models import ReviewState


def test_regex_route_full_review_by_default():
    assert _regex_route("проанализируй всё") == ("full_review", [])


def test_regex_route_coverage_review():
    assert _regex_route("проверь покрытие") == ("coverage_review", [])


def test_regex_route_requirement_coverage_with_id():
    scenario, ids = _regex_route("покрытие требования REQ-001")
    assert scenario == "requirement_coverage"
    assert ids == ["REQ-001"]


def test_regex_route_design_and_standards():
    assert _regex_route("оцени дизайн тестов")[0] == "design_review"
    assert _regex_route("проверь соответствие стандартам")[0] == "standards_review"


def test_regex_route_unlinked():
    assert _regex_route("найди тесты без требований unlinked")[0] == "find_unlinked_tests"


def test_route_request_falls_back_to_regex_on_llm_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(
        "src.llm_provider.RouterAIProvider.chat_completion", boom
    )

    state = route_request(ReviewState(user_query="проверь покрытие требований"))
    assert state.scenario == "coverage_review"
    assert state.agents_to_run == ["coverage"]
