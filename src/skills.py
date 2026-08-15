import functools
import json
from pathlib import Path
from typing import Any

import yaml

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Маппинг объявленных в YAML типов на JSON Schema (для function-calling).
_TYPE_MAP: dict[str, dict] = {
    "string": {"type": "string"},
    "int": {"type": "integer"},
    "integer": {"type": "integer"},
    "float": {"type": "number"},
    "number": {"type": "number"},
    "bool": {"type": "boolean"},
    "boolean": {"type": "boolean"},
    "list": {"type": "array", "items": {}},
}


def _json_type_spec(type_hint: str) -> dict:
    t = (type_hint or "string").strip().lower()
    if t.startswith("optional["):
        inner = t[len("optional[") : -1]
        spec = dict(_TYPE_MAP.get(inner, {"type": "string"}))
        spec["nullable"] = True
        return spec
    return dict(_TYPE_MAP.get(t, {"type": "string"}))


@functools.lru_cache(maxsize=None)
def load_skill(name: str) -> dict[str, Any]:
    """Загружает описание скилла из skills/<name>.yaml (с кэшированием)."""
    path = _SKILLS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=None)
def list_skill_names() -> list[str]:
    return sorted(p.stem for p in _SKILLS_DIR.glob("*.yaml"))


def build_tool_definition(skill: dict[str, Any]) -> dict[str, Any]:
    """Строит OpenAI-совместимое описание tool из YAML скилла."""
    props: dict[str, Any] = {}
    required: list[str] = []
    for param, spec in (skill.get("input_schema") or {}).items():
        spec_str = str(spec)
        props[param] = _json_type_spec(spec_str)
        if not spec_str.strip().lower().startswith("optional"):
            required.append(param)
    return {
        "type": "function",
        "function": {
            "name": skill["name"],
            "description": (skill.get("description") or "").strip(),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


class ToolRegistry:
    """Реестр скиллов, загружаемый из skills/*.yaml.

    Скиллы теперь — единственный источник описания инструментов: на их основе
    строятся определения для function-calling, а их схемы валидируют аргументы.
    """

    def __init__(self) -> None:
        self._skills = {n: load_skill(n) for n in list_skill_names()}
        self._tools = [build_tool_definition(s) for s in self._skills.values()]

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    def has(self, name: str) -> bool:
        return name in self._skills

    def _validate(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self._skills[name].get("input_schema") or {}
        clean: dict[str, Any] = {}
        for key, spec in schema.items():
            if key not in args:
                if str(spec).strip().lower().startswith("optional"):
                    continue
                raise ValueError(f"Missing required argument '{key}' for tool '{name}'")
            value = args[key]
            if "int" in str(spec).lower() and value is not None:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            clean[key] = value
        return clean

    def execute(self, name: str, args: dict[str, Any]) -> Any:
        if not self.has(name):
            return f"Unknown tool: {name}"
        args = self._validate(name, args or {})

        if name == "sql_query":
            from src.tools.sql_tool import execute_sql

            return execute_sql(args["query"], args.get("params"))
        if name == "rag_search":
            from src.tools.rag_tool import rag_search

            return rag_search(
                args["collection"],
                args["query"],
                args.get("top_k", 10),
            )
        return f"Unhandled tool: {name}"

    def execute_to_json(self, name: str, args: dict[str, Any]) -> str:
        """Выполняет скилл и сериализует результат в JSON-строку для tool-сообщения."""
        result = self.execute(name, args)
        return json.dumps(result, ensure_ascii=False, default=str)
