import functools
import json
from pathlib import Path
from typing import Any

import yaml

from src.json_utils import json_type_spec

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


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
    """Строит OpenAI-совместимое описание tool из YAML скилла.

    В описание (``description``), помимо базового summary, включаются SOP,
    guardrails и контракт вывода (``returns``) — чтобы модель получала рабочую
    инструкцию использования инструмента, а не только тонкое описание.
    """
    props: dict[str, Any] = {}
    required: list[str] = []
    for param, spec in (skill.get("input_schema") or {}).items():
        spec_str = str(spec)
        props[param] = json_type_spec(spec_str)
        if not spec_str.strip().lower().startswith("optional"):
            required.append(param)

    desc = (skill.get("description") or "").strip()
    extras: list[str] = []
    if skill.get("returns"):
        extras.append("Returns:\n" + str(skill["returns"]).strip())
    if skill.get("sop"):
        extras.append("SOP:\n" + str(skill["sop"]).strip())
    if skill.get("guardrails"):
        gb = skill["guardrails"]
        lines: list[str] = []
        if gb.get("forbidden_actions"):
            lines.append("Forbidden actions: " + "; ".join(gb["forbidden_actions"]))
        if "requires_human_approval" in gb:
            lines.append(f"Requires human approval: {gb['requires_human_approval']}")
        if gb.get("notes"):
            lines.append(str(gb["notes"]))
        if lines:
            extras.append("Guardrails:\n" + "\n".join(lines))
    if extras:
        desc = desc + "\n\n" + "\n\n".join(extras)

    return {
        "type": "function",
        "function": {
            "name": skill["name"],
            "description": desc,
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

        # При нарушении выбрасывается ValueError — ошибка ловится и деградирует
        # на уровне агента/узла/графа (как и при missing required argument).
        for key, cons in (self._skills[name].get("constraints") or {}).items():
            if key not in clean or clean[key] is None:
                continue
            value = clean[key]
            if "enum" in cons and value not in cons["enum"]:
                raise ValueError(
                    f"{name}.{key} must be one of {cons['enum']}, got {value!r}"
                )
            if "allowlist_prefix" in cons and isinstance(value, str):
                up = value.strip().upper()
                if not any(up.startswith(p) for p in cons["allowlist_prefix"]):
                    raise ValueError(
                        f"{name}.{key} must start with one of {cons['allowlist_prefix']}"
                    )
            if (
                cons.get("single_statement")
                and isinstance(value, str)
                and ";" in value.rstrip(";").strip()
            ):
                raise ValueError(f"{name}.{key} must be a single SQL statement (no ';')")
            if "min" in cons or "max" in cons:
                try:
                    num = int(value)
                    if "min" in cons and num < cons["min"]:
                        num = cons["min"]
                    if "max" in cons and num > cons["max"]:
                        num = cons["max"]
                    clean[key] = num
                except (TypeError, ValueError):
                    pass
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
        if name == "code_validator":
            from src.tools.code_validator import validate_test_cases

            return validate_test_cases(
                args["test_cases"],
                set(args["known_requirement_ids"]) if args.get("known_requirement_ids") else None,
            )
        return f"Unhandled tool: {name}"

    def execute_to_json(self, name: str, args: dict[str, Any]) -> str:
        """Выполняет скилл и сериализует результат в JSON-строку для tool-сообщения."""
        result = self.execute(name, args)
        return json.dumps(result, ensure_ascii=False, default=str)
