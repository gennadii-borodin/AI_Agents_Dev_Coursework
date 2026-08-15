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
    assert "filter" not in skill["input_schema"]
    assert skill["input_schema"]["collection"] == "string"


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
    result = registry.execute("does_not_exist", {})
    assert "Unknown tool" in result


def test_registry_validate_missing_required():
    registry = ToolRegistry()
    try:
        registry.execute("rag_search", {"query": "x"})  # нет collection
        assert False, "should raise"
    except ValueError as e:
        assert "collection" in str(e)


def test_registry_execute_to_json_unknown_is_serialized():
    registry = ToolRegistry()
    payload = registry.execute_to_json("does_not_exist", {})
    assert "Unknown tool" in payload
    # execute_to_json всегда возвращает валидную JSON-строку
    json.loads(payload)
