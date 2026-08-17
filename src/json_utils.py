"""Общие утилиты разбора/восстановления JSON-ответов LLM и генерации JSON-схем.

Единый источник логики репейра JSON (раньше дублировался в coverage/design/
standards-агентах) и построения JSON-схем (раньше дублировался в prompts.py и
skills.py).
"""
import json
import re
from typing import Any, Callable, Optional

import json_repair

# Маппинг объявленных в YAML типов на JSON Schema (для function-calling и
# structured outputs). Единый источник для prompts.py и skills.py.
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


def json_type_spec(type_hint: str) -> dict:
    t = (type_hint or "string").strip().lower()
    if t.startswith("optional["):
        inner = t[len("optional[") : -1]
        spec = dict(_TYPE_MAP.get(inner, {"type": "string"}))
        spec["nullable"] = True
        return spec
    return dict(_TYPE_MAP.get(t, {"type": "string"}))


def fix_json(text: str) -> str:
    """Грубое восстановление баланса скобок/запятых в обрезанном JSON.

    Закрывает незакрытые ``{``/``[``, убирает висячие запятые, подставляет
    ``null`` при ``key: ,``. Фоллбэк, когда json_repair не справился.
    """
    text = text.strip()
    if not text.startswith("{"):
        idx = text.find("{")
        if idx >= 0:
            text = text[idx:]
    brace_count = text.count("{") - text.count("}")
    if brace_count > 0:
        text = text + "}" * brace_count
    bracket_count = text.count("[") - text.count("]")
    if bracket_count > 0:
        text = text + "]" * bracket_count
    if text.endswith(","):
        text = text[:-1]
    if not text.endswith("}"):
        text = text + "}"
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r":\s*,", ": null,", text)
    return text


def parse_json_response(response: str, on_repair: Optional[Callable[[], None]] = None) -> dict:
    """Единый разбор JSON-ответа LLM с восстановлением.

    Порядок: ``json.loads`` -> ``json_repair`` -> ``fix_json`` (фоллбэк).
    ``on_repair`` — опциональный callback, вызываемый при переходе к ветке
    восстановления (например, чтобы записать событие ``json_repaired`` в спан).
    Бросает ``ValueError``/``json.JSONDecodeError``, если разобрать не удалось.
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    if on_repair is not None:
        on_repair()

    repaired = json_repair.repair_json(response, return_objects=True)
    if isinstance(repaired, str):
        return json.loads(repaired)
    if isinstance(repaired, dict):
        return repaired
    # Неожиданный тип (напр. список) — последняя попытка через fix_json.
    return json.loads(fix_json(response))


def build_schema_from_spec(schema: dict[str, str]) -> dict[str, Any]:
    """Строит JSON Schema (OpenAI-style) из output_schema YAML для structured outputs.

    ``schema`` — dict {имя_поля: type_hint}.
    """
    properties = {key: json_type_spec(str(spec)) for key, spec in schema.items()}
    return {
        "type": "object",
        "properties": properties,
        "required": list(schema.keys()),
    }
