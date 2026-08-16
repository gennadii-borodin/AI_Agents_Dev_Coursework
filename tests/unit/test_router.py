
from src.graph import _regex_route, route_request
from src.models import ReviewState
from tests.integration.helpers import ScriptedLLM, apply_llm_patch


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


def test_route_request_uses_llm_scenario_authoritatively(monkeypatch):
    # regex отправил бы "сделай всё" в full_review, но LLM-роутер — единственный
    # источник истины — возвращает find_unlinked_tests. Проверяет, что scenario
    # берётся из LLM, а не из regex (фикс семантического разрыва роутинга).
    stub = ScriptedLLM(router_response='{"scenario": "find_unlinked_tests", "requirement_ids": []}')
    apply_llm_patch(monkeypatch, stub)

    state = route_request(ReviewState(user_query="сделай всё и сразу"))
    assert state.scenario == "find_unlinked_tests"
    assert state.agents_to_run == []


def test_route_request_requirement_ids_only_from_llm(monkeypatch):
    # requirement_ids берутся только из LLM-ответа, а не из regex-извлечения
    # из текста запроса.
    stub = ScriptedLLM(router_response='{"scenario": "requirement_coverage", "requirement_ids": ["REQ-007"]}')
    apply_llm_patch(monkeypatch, stub)

    state = route_request(ReviewState(user_query="проверь покрытие"))
    assert state.scenario == "requirement_coverage"
    assert state.requirement_ids == ["REQ-007"]


def test_route_request_asks_rephrase_on_llm_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(
        "src.llm_provider.RouterAIProvider.chat_completion", boom
    )

    state = route_request(ReviewState(user_query="проверь покрытие требований"))
    # LLM недоступен → маршрутизация не может определить сценарий,
    # просим пользователя переформулировать (агенты не запускаются).
    assert state.scenario == ""
    assert state.agents_to_run == []
    assert any("routing_failed" in e for e in state.errors)
    assert state.unresolved_questions


def test_unknown_llm_scenario_asks_rephrase(monkeypatch):
    def bad_scenario(self, *args, **kwargs):
        # LLM возвращает сценарий вне допустимого набора (adversarial #1, C1)
        return '{"scenario": "coverage", "requirement_ids": []}'

    monkeypatch.setattr(
        "src.llm_provider.RouterAIProvider.chat_completion", bad_scenario
    )

    state = route_request(ReviewState(user_query="проверь покрытие требований"))
    # Недопустимый scenario от LLM → просим переформулировать, а не молча
    # подменяем regex-значением.
    assert state.scenario == ""
    assert state.agents_to_run == []
    assert any("routing_failed" in e for e in state.errors)


def test_graph_invoke_does_not_crash_on_unknown_scenario(monkeypatch, isolate_services):
    from src.graph import build_graph
    from tests.integration.helpers import ScriptedLLM, apply_llm_patch

    # LLM-роутер возвращает сценарий вне допустимого набора (adversarial #1, C1)
    stub = ScriptedLLM(router_response='{"scenario": "coverage", "requirement_ids": []}')
    apply_llm_patch(monkeypatch, stub)

    # graph.invoke не падает: провал роутинга завершает прогон корректно
    # и просит пользователя переформулировать задачу.
    g = build_graph()
    result = g.invoke(ReviewState(user_query="проверь покрытие требований"))
    assert result["scenario"] == ""
    assert any("routing_failed" in e for e in result["errors"])
    assert result["final_answer"]
