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
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "integer": {"type": "integer"},
    "float": {"type": "number"},
    "number": {"type": "number"},
    "bool": {"type": "boolean"},
    "boolean": {"type": "boolean"},
    "list": {"type": "array", "items": {}},
    "dict": {"type": "object"},
}


def _resolve_type(type_hint: str) -> dict:
    """Разрешает тип YAML в JSON Schema, включая обобщённые подсказки.

    Поддерживает ``list[X]`` (массив с элементами типа X) и ``dict[...]``
    (объект). Ранее ``list[dict]`` не находился в ``_TYPE_MAP`` и тихо
    деградировал до ``{"type": "string"}``, из-за чего модель не видела форму
    вложенных элементов в схеме tool/structured-output.
    """
    t = (type_hint or "string").strip()
    low = t.lower()
    if low.startswith("list[") and low.endswith("]"):
        inner = t[len("list[") : -1].strip()
        return {"type": "array", "items": _resolve_type(inner)}
    if low.startswith("dict[") and low.endswith("]"):
        return {"type": "object"}
    return dict(_TYPE_MAP.get(low, {"type": "string"}))


def json_type_spec(type_hint: str) -> dict:
    t = (type_hint or "string").strip()
    optional = False
    if t.lower().startswith("optional["):
        optional = True
        t = t[len("optional[") : -1].strip()
    spec = _resolve_type(t)
    if optional:
        spec["nullable"] = True
    return spec


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
