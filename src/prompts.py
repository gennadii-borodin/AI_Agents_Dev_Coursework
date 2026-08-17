import functools
import json
from pathlib import Path
from typing import Any

import yaml

from src.json_utils import build_schema_from_spec

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@functools.lru_cache(maxsize=None)
def load_prompt(name: str) -> dict[str, Any]:
    """Загружает промпт из prompts/<name>.yaml (с кэшированием)."""
    path = _PROMPTS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_example(example: Any) -> str:
    return json.dumps(example, ensure_ascii=False, indent=2)


def build_agent_system_prompt(name: str) -> str:
    """Собирает system_prompt агента из prompts/<name>.yaml.

    Единственный источник истины для промптов теперь — prompts/*.yaml.
    К промпту добавляется JSON-контракт вывода (поле ``response_json`` YAML)
    и, при наличии, few-shot примеры (поле ``few_shot`` YAML).
    """
    data = load_prompt(name)
    system = data.get("system_prompt", "")

    response_json = data.get("response_json")
    if response_json:
        system = system + (
            "\n\nВерни СТРОГО валидный JSON точно в следующей структуре:\n"
            "```json\n" + response_json.strip() + "\n```"
        )

    few_shot = data.get("few_shot")
    if few_shot:
        examples = "\n\n## Примеры\n"
        for ex in few_shot:
            if "input" in ex:
                examples += f"Ввод: {ex['input']}\n"
            if "output" in ex:
                examples += f"Вывод:\n{_format_example(ex['output'])}\n"
        system = system + examples

    return system


def build_json_schema(name: str) -> dict[str, Any]:
    """Строит JSON Schema (OpenAI-style) из output_schema YAML для structured outputs."""
    data = load_prompt(name)
    return build_schema_from_spec(data.get("output_schema") or {})
