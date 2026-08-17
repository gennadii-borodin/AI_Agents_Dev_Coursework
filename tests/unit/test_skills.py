import json

from src.skills import (
    ToolRegistry,
    build_tool_definition,
    list_skill_names,
    load_skill,
)


def test_skills_loaded_from_yaml():
    names = list_skill_names()
    assert "sql_query" in names
    assert "rag_search" in names
    skill = load_skill("rag_search")
    assert skill["name"] == "rag_search"


def test_rag_search_schema_has_no_filter_or_standards():
    skill = load_skill("rag_search")
    assert "filter" not in skill["input"]["properties"]
    assert skill["input"]["properties"]["collection"]["type"] == "string"


def test_build_tool_definition_maps_types():
    skill = load_skill("rag_search")
    tool = build_tool_definition(skill)
    assert tool["type"] == "function"
    fn = tool["function"]
    assert fn["name"] == "rag_search"
    props = fn["parameters"]["properties"]
    assert props["collection"]["type"] == "string"
    assert props["top_k"]["type"] == "integer"
    assert props["top_k"]["nullable"] is True
    assert "top_k" not in fn["parameters"]["required"]
    assert "collection" in fn["parameters"]["required"]


def test_registry_exposes_tools():
    registry = ToolRegistry()
    names = {t["function"]["name"] for t in registry.tools}
    assert names == {"sql_query", "rag_search", "code_validator"}


def test_registry_unknown_tool_returns_message():
    registry = ToolRegistry()
    assert not registry.has("does_not_exist")
    result = registry._execute("does_not_exist", {})
    assert "Unknown tool" in result


def test_registry_validate_missing_required():
    registry = ToolRegistry()
    try:
        registry._execute("rag_search", {"query": "x"})  # нет collection
        assert False, "should raise"
    except ValueError as e:
        assert "collection" in str(e)


def test_registry_execute_to_json_unknown_is_serialized():
    registry = ToolRegistry()
    payload = registry.execute_to_json("does_not_exist", {})
    assert "Unknown tool" in payload
    # execute_to_json всегда возвращает валидную JSON-строку
    json.loads(payload)


# --- Инъекция constraints из YAML в JSON-схему (критическое замечание #1) ---


def test_build_tool_definition_injects_enum_constraint():
    tool = build_tool_definition(load_skill("rag_search"))
    props = tool["function"]["parameters"]["properties"]
    assert props["collection"]["enum"] == ["requirements", "test_cases"]


def test_build_tool_definition_injects_min_max_constraint():
    tool = build_tool_definition(load_skill("rag_search"))
    props = tool["function"]["parameters"]["properties"]
    assert props["top_k"]["minimum"] == 1
    assert props["top_k"]["maximum"] == 50


def test_build_tool_definition_injects_allowlist_prefix_pattern():
    tool = build_tool_definition(load_skill("sql_query"))
    props = tool["function"]["parameters"]["properties"]
    assert "pattern" in props["query"]
    assert props["query"]["pattern"].startswith("^(")


# --- when_to_use / objective попадают в description (замечание #2) ---


def test_build_tool_definition_includes_when_to_use_and_objective():
    tool = build_tool_definition(load_skill("rag_search"))
    desc = tool["function"]["description"]
    assert "When to use:" in desc
    assert "Objective:" in desc
    skill = load_skill("rag_search")
    assert skill["when_to_use"].strip() in desc
    assert skill["objective"].strip() in desc


# --- Декларативный dispatch: каждый скилл имеет handler (#3 regression) ---


def test_every_skill_has_a_handler():
    registry = ToolRegistry()
    for name in list_skill_names():
        # _get_handlers строит маппинг для всех зарегистрированных скиллов,
        # иначе вызов _dispatch вернул бы "Unhandled tool".
        assert name in registry._get_handlers()


# --- Enforcement прав доступа в execute_for_agent (#4) ---


def test_execute_for_agent_denies_unpermitted_skill():
    registry = ToolRegistry()
    result = registry.execute_for_agent("design_agent", "sql_query", {"query": "SELECT 1"})
    assert isinstance(result, dict)
    assert result["error"]
    assert "not permitted" in result["error"]


def test_execute_for_agent_allows_permitted_skill():
    registry = ToolRegistry()
    result = registry.execute_for_agent(
        "coverage_agent", "rag_search", {"collection": "test_cases", "query": "x"}
    )
    # rag_search при отсутствии БД вернёт error-конверт, но НЕ «not permitted».
    assert (
        result.get("error")
        != "agent 'coverage_agent' is not permitted to use skill 'rag_search'"
    )
