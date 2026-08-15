import functools
from pathlib import Path
from typing import Any

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@functools.lru_cache(maxsize=None)
def load_prompt(name: str) -> dict[str, Any]:
    """Загружает промпт из prompts/<name>.yaml (с кэшированием)."""
    path = _PROMPTS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent_system_prompt(name: str) -> str:
    """Собирает system_prompt агента из prompts/<name>.yaml.

    Единственный источник истины для промптов теперь — prompts/*.yaml.
    К промпту добавляется JSON-контракт вывода (поле ``response_json`` YAML),
    чтобы модель возвращала структуру, ожидаемую pydantic-моделью отчёта.
    """
    data = load_prompt(name)
    system = data.get("system_prompt", "")
    response_json = data.get("response_json")
    if response_json:
        system = system + (
            "\n\nВерни СТРОГО валидный JSON точно в следующей структуре:\n"
            "```json\n" + response_json.strip() + "\n```"
        )
    return system
