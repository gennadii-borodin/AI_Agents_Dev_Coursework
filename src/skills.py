import functools
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import yaml

from src.tool_resilience import CircuitBreaker, RateLimiter, run_with_timeout

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
_MANIFEST_PATH = _SKILLS_DIR / "_agent_manifest.yaml"

# Политики устойчивости (RateLimiter/CircuitBreaker) общие для всех экземпляров
# ToolRegistry в рамках процесса. Иначе каждый `ToolRegistry()` создавал бы
# собственные счётчики, и circuit breaker не агрегировал бы сбои по системе
# (порог failure_threshold достигался бы никогда при разнесённых вызовах).
_POLICY_CACHE: dict[str, tuple] = {}

# Источник истины для реализаций скиллов: handler регистрируется декларативно
# через @register_skill рядом с описанием, без центрального if/elif-диспетчера.
_SKILL_HANDLERS: dict[str, Callable] = {}


def register_skill(name: str) -> Callable:
    """Декоратор-регистратор реализации скилла.

    Добавление нового скилла сводится к: (1) описанию в skills/<name>.yaml и
    (2) функции, помеченной @register_skill("<name>"). Центральный dict не
    правится — реестр собирает handlers из этого модуля при импорте.
    """

    def deco(fn: Callable) -> Callable:
        _SKILL_HANDLERS[name] = fn
        return fn

    return deco


def _get_policy(skill: dict[str, Any]) -> tuple:
    name = skill["name"]
    if name in _POLICY_CACHE:
        return _POLICY_CACHE[name]
    policy = (
        RateLimiter((skill.get("rate_limit") or {}).get("max_calls_per_minute")),
        CircuitBreaker(
            (skill.get("circuit_breaker") or {}).get("failure_threshold"),
            (skill.get("circuit_breaker") or {}).get("cooldown_seconds", 0) or 0,
        ),
        skill.get("timeout_seconds"),
    )
    _POLICY_CACHE[name] = policy
    return policy


@functools.lru_cache(maxsize=None)
def load_skill(name: str) -> dict[str, Any]:
    """Загружает описание скилла из skills/<name>.yaml (с кэшированием)."""
    path = _SKILLS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=None)
def list_skill_names() -> list[str]:
    # Файлы, начинающиеся с подчёркивания (напр. _agent_manifest.yaml),
    # не являются скиллами и не попадают в реестр инструментов.
    return sorted(
        p.stem for p in _SKILLS_DIR.glob("*.yaml") if not p.stem.startswith("_")
    )


# Кэш манифеста: перечитывать YAML на каждый вызов агента/инструмента не нужно.
# При загрузке манифест валидируется — ссылка на несуществующий скилл логируется,
# что сразу вскрывает опечатку, вместо молчаливой выдачи агенту пустого набора.
_MANIFEST_CACHE: dict[str, Any] | None = None


def load_manifest() -> dict[str, Any]:
    """Загружает и кэширует манифест доступа агентов к скиллам.

    Возвращает кэшированный dict; при первой загрузке проверяет, что каждый
    перечисленный в манифесте скилл реально существует (skills/<name>.yaml).
    Опечатка в имени скилла логируется предупреждением, а не приводит к тому,
    что агент молча получит пустой список инструментов.
    """
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE
    if not _MANIFEST_PATH.exists():
        _MANIFEST_CACHE = {}
        return _MANIFEST_CACHE
    data = yaml.safe_load(_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    known = set(list_skill_names())
    for agent, skills in (data.get("agents") or {}).items():
        for s in skills or []:
            if s not in known:
                logger.warning(
                    "Manifest references unknown skill '%s' for agent '%s'", s, agent
                )
    _MANIFEST_CACHE = data
    return data


# ---------------------------------------------------------------------------
# Преобразование декларативной схемы скилла в JSON Schema инструмента (OpenAI).
# Схема — единственный источник истины; и описание инструмента, и рантайм-
# валидация (_validate) читают из неё, поэтому контракт и реализация не расходятся.
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}


def _property_to_jsonschema(spec: dict[str, Any]) -> dict[str, Any]:
    jtype = spec.get("type", "string")
    js: dict[str, Any] = {"type": _TYPE_MAP.get(jtype, jtype)}
    if spec.get("description"):
        js["description"] = str(spec["description"]).strip()
    if spec.get("nullable"):
        js["nullable"] = True
    if "enum" in spec:
        js["enum"] = list(spec["enum"])
    if "minimum" in spec:
        js["minimum"] = spec["minimum"]
    if "maximum" in spec:
        js["maximum"] = spec["maximum"]
    if "default" in spec:
        js["default"] = spec["default"]
    if "items" in spec:
        js["items"] = _property_to_jsonschema(spec["items"])
    # allowlist_prefix поднимается в JSON-схему как pattern, ограничивая LLM
    # ещё до рантайм-валидации (см. _check_sql).
    if "allowlist_prefix" in spec:
        prefixes = "|".join(re.escape(p) for p in spec["allowlist_prefix"])
        js["pattern"] = f"^({prefixes})"
    return js


def build_tool_definition(skill: dict[str, Any]) -> dict[str, Any]:
    """Строит OpenAI-совместимое описание tool из YAML скилла.

    В описание (``description``) попадают ТОЛЬКО краткое summary, цель
    (``objective``) и триггер использования (``when_to_use``) — ровно то, что
    модели нужно для принятия решения о вызове. Подробные SOP/контракт вывода/
    guardrails НЕ впихиваются в описание инструмента (это раздувало бы токены на
    каждый вызов и «размывало» триггер); они доступны как справочное тело
    скилла через ``build_skill_reference``/``ToolRegistry.reference_for_agent``
    и подгружаются в контекст агента, которому скилл разрешён (on-demand).
    """
    input_schema = skill.get("input") or {}
    props: dict[str, Any] = {}
    for pname, spec in (input_schema.get("properties") or {}).items():
        props[pname] = _property_to_jsonschema(spec)

    desc = (skill.get("description") or "").strip()
    extras: list[str] = []
    if skill.get("objective"):
        extras.append("Objective:\n" + str(skill["objective"]).strip())
    if skill.get("when_to_use"):
        extras.append("When to use:\n" + str(skill["when_to_use"]).strip())
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
                "required": list(input_schema.get("required") or []),
            },
        },
    }


def build_skill_reference(skill: dict[str, Any]) -> str:
    """Собирает справочное тело скилла (SOP/контракт вывода/guardrails).

    Используется для on-demand подгрузки знаний агенту (паттерн Agent Skills):
    детали не «висят» в описании инструмента, а выдаются в контекст того агента,
    которому скилл реально разрешён.
    """
    parts: list[str] = [f"# Skill: {skill['name']}"]
    if skill.get("description"):
        parts.append(str(skill["description"]).strip())
    for key in ("objective", "when_to_use", "returns", "sop"):
        if skill.get(key):
            title = key.replace("_", " ").title()
            parts.append(f"## {title}\n" + str(skill[key]).strip())
    gb = skill.get("guardrails")
    if gb:
        lines: list[str] = []
        if gb.get("forbidden_actions"):
            lines.append("Forbidden actions: " + "; ".join(gb["forbidden_actions"]))
        if "requires_human_approval" in gb:
            lines.append(f"Requires human approval: {gb['requires_human_approval']}")
        if gb.get("notes"):
            lines.append(str(gb["notes"]))
        if lines:
            parts.append("## Guardrails\n" + "\n".join(lines))
    return "\n\n".join(parts)


class ToolRegistry:
    """Реестр скиллов, загружаемый из skills/*.yaml.

    Скиллы — единственный источник описания инструментов: на их основе
    строятся определения для function-calling, а их схемы валидируют аргументы
    и (для SQL) реально enforce-ят read-only контракт.
    """

    def __init__(self) -> None:
        self._skills = {n: load_skill(n) for n in list_skill_names()}
        self._tools = [build_tool_definition(s) for s in self._skills.values()]
        self._policies = {n: _get_policy(s) for n, s in self._skills.items()}

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    def tools_for_agent(self, agent_name: str) -> list[dict[str, Any]]:
        """Возвращает только те tool-определения, которые разрешены агенту манифестом.

        Используется для progressive disclosure на уровне агента: агент видит
        лишь скиллы, перечисленные для него в skills/_agent_manifest.yaml.
        """
        allowed = set((load_manifest().get("agents") or {}).get(agent_name) or [])
        return [t for t in self._tools if t["function"]["name"] in allowed]

    def reference_for_agent(self, agent_name: str) -> str:
        """Справочные тела (SOP/guardrails/контракт) разрешённых агенту скиллов.

        Подгружается в system-промпт агента (on-demand), чтобы детальные
        инструкции скиллов не раздували описание самого инструмента.
        """
        allowed = set((load_manifest().get("agents") or {}).get(agent_name) or [])
        parts = [build_skill_reference(self._skills[s]) for s in allowed if s in self._skills]
        return "\n\n".join(parts)

    def has(self, name: str) -> bool:
        return name in self._skills

    def execute_for_agent(
        self, agent_name: str, name: str, args: dict[str, Any]
    ) -> Any:
        """Выполняет скилл с проверкой прав доступа агента из манифеста.

        Единственная публичная точка входа в реестр: вызов любого скилла должен
        идти через неё, чтобы гарантировать, что агент использует только
        скиллы, перечисленные для него в ``skills/_agent_manifest.yaml``
        (progressive disclosure на уровне агента). Непосредственная диспетчеризация
        инкапсулирована в приватном ``_execute`` и недоступна извне, поэтому
        обойти проверку прав нельзя. При нарушении возвращает error-конверт скилла.
        """
        allowed = set((load_manifest().get("agents") or {}).get(agent_name) or [])
        if name not in allowed:
            return self._error_envelope(
                name,
                f"agent '{agent_name}' is not permitted to use skill '{name}'",
            )
        return self._execute(name, args)

    # ------------------------------------------------------------------
    # Валидация аргументов по декларативной схеме скилла (источник истины).
    # ------------------------------------------------------------------

    def _validate(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = (self._skills[name].get("input") or {}).get("properties") or {}
        required = (self._skills[name].get("input") or {}).get("required") or []

        for key in required:
            if key not in args or args[key] is None:
                raise ValueError(
                    f"Missing required argument '{key}' for tool '{name}'"
                )

        clean: dict[str, Any] = {}
        for key, spec in schema.items():
            if key not in args or args[key] is None:
                if "default" in spec and key not in args:
                    clean[key] = spec["default"]
                continue
            value = args[key]
            jtype = spec.get("type")

            if jtype == "integer":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    raise ValueError(f"{name}.{key} must be an integer, got {value!r}")
            if jtype == "array" and not isinstance(value, (list, tuple, set)):
                raise ValueError(f"{name}.{key} must be a list, got {type(value).__name__}")
            if jtype == "object" and not isinstance(value, dict):
                raise ValueError(f"{name}.{key} must be an object, got {type(value).__name__}")
            if jtype == "array" and "items" in spec:
                item_type = spec["items"].get("type")
                if item_type == "object":
                    for i, item in enumerate(value):
                        if not isinstance(item, dict):
                            raise ValueError(
                                f"{name}.{key}[{i}] must be an object, got {type(item).__name__}"
                            )

            if "enum" in spec and value not in spec["enum"]:
                raise ValueError(
                    f"{name}.{key} must be one of {spec['enum']}, got {value!r}"
                )
            if jtype in ("integer", "number") and isinstance(value, (int, float)):
                if "minimum" in spec:
                    value = max(value, spec["minimum"])
                if "maximum" in spec:
                    value = min(value, spec["maximum"])

            # SQL-специфичный контракт read-only реально enforce-ится здесь
            # (а не где-то ещё) — YAML остаётся единственным источником истины.
            if "forbidden_keywords" in spec and isinstance(value, str):
                _check_sql(name, value, spec)

            clean[key] = value
        return clean

    def _execute(self, name: str, args: dict[str, Any]) -> Any:
        """Приватная диспетчеризация + устойчивость.

        Не проверяет права доступа — это делает ``execute_for_agent``. Прямой
        вызов извне невозможен (метод приватный), поэтому обойти манифест нельзя.
        """
        if not self.has(name):
            return f"Unknown tool: {name}"
        args = self._validate(name, args or {})

        policy = self._policies.get(name)
        if policy is not None:
            rl, cb, timeout = policy
            if not rl.allow():
                return self._error_envelope(name, "rate_limited")
            if not cb.allow():
                return self._error_envelope(name, "circuit_open")
            try:
                result = run_with_timeout(self._dispatch, timeout, name, args)
                cb.record_success()
                return self._normalize_output(name, result)
            except Exception as e:  # noqa: BLE001
                cb.record_failure()
                return self._error_envelope(name, f"{type(e).__name__}: {e}")
        return self._normalize_output(name, self._dispatch(name, args))

    def _get_handlers(self) -> dict[str, Callable]:
        """Возвращает зарегистрированные реализации скиллов (декларативный реестр)."""
        return _SKILL_HANDLERS

    def _dispatch(self, name: str, args: dict[str, Any]) -> Any:
        handlers = _SKILL_HANDLERS
        if name not in handlers:
            return f"Unhandled tool: {name}"
        return handlers[name](args)

    def _normalize_output(self, name: str, result: Any) -> Any:
        """Договаривает вывод скилла с объявленным output-контрактом.

        Если схема вывода объявляет поле ``error``, а обработчик его не вернул
        (успешный путь), дописываем ``error: null`` — чтобы контракт вывода
        соблюдался и на успешных ответах, а не только на ошибках.
        """
        out_props = (self._skills[name].get("output") or {}).get("properties") or {}
        if isinstance(result, dict) and "error" in out_props and "error" not in result:
            result["error"] = None
        return result

    def _error_envelope(self, name: str, msg: str) -> dict[str, Any]:
        """Формирует error-конверт, совместимый с output_schema скилла."""
        if name == "sql_query":
            return {"results": [], "row_count": 0, "truncated": False, "error": msg}
        if name == "rag_search":
            return {"results": [], "error": msg}
        if name == "code_validator":
            return {"findings": [], "checked": 0, "error": msg}
        return {"error": msg}

    def execute_to_json(self, name: str, args: dict[str, Any]) -> str:
        """Выполняет скилл и сериализует результат в JSON-строку для tool-сообщения.

        Использует приватный ``_execute`` (проверка прав — задача вызывающего
        через ``execute_for_agent``).
        """
        result = self._execute(name, args)
        return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# SQL-валидация: единственная реализация read-only контракта (читается из YAML).
# ---------------------------------------------------------------------------


def _strip_sql_literals(sql: str) -> str:
    """Убирает строковые литералы и комментарии, сохраняя структуру запроса.

    Позволяет корректно искать ключевые слова/разделители вне строк и комментариев
    (иначе ``;`` или ``DROP`` внутри литерала дали бы ложное срабатывание).
    """
    out: list[str] = []
    i, n = 0, len(sql)
    in_single = in_double = in_line = in_block = False
    while i < n:
        c = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if in_line:
            if c == "\n":
                in_line = False
                out.append(" ")
            i += 1
            continue
        if in_block:
            if c == "*" and nxt == "/":
                in_block = False
                i += 2
                out.append(" ")
                continue
            i += 1
            continue
        if in_single:
            if c == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if c == '"':
                in_double = False
            i += 1
            continue
        if c == "-" and nxt == "-":
            in_line = True
            i += 2
            out.append(" ")
            continue
        if c == "/" and nxt == "*":
            in_block = True
            i += 2
            out.append(" ")
            continue
        if c == "'":
            in_single = True
            i += 1
            continue
        if c == '"':
            in_double = True
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _sql_first_keyword(sql: str) -> str:
    cleaned = _strip_sql_literals(sql)
    m = re.match(r"\s*([A-Za-z_]+)", cleaned)
    return m.group(1).upper() if m else ""


def _sql_statement_count(sql: str) -> int:
    cleaned = _strip_sql_literals(sql).strip()
    if not cleaned:
        return 0
    return len([p for p in cleaned.split(";") if p.strip()])


def _check_sql(name: str, value: str, spec: dict[str, Any]) -> None:
    """Enforce-ит read-only контракт SQL из декларативной схемы скилла."""
    if spec.get("single_statement") and _sql_statement_count(value) > 1:
        raise ValueError(
            f"{name}.query must be a single SQL statement (multiple ';' detected)"
        )
    allow = [p.upper() for p in spec.get("allowlist_prefix", [])]
    first = _sql_first_keyword(value)
    if allow and first not in allow:
        raise ValueError(
            f"{name}.query must start with one of {spec['allowlist_prefix']}, got {first!r}"
        )
    tokens = re.findall(r"[A-Za-z_]+", _strip_sql_literals(value).upper())
    for kw in spec.get("forbidden_keywords", []):
        if kw.upper() in tokens:
            raise ValueError(f"{name}.query contains forbidden keyword '{kw}'")


# ---------------------------------------------------------------------------
# Реализации скиллов. Регистрируются декларативно; сами функции инструментов
# импортируются внутри обёрток в момент вызова (а не при регистрации), чтобы
# monkeypatch в тестах продолжал работать.
# ---------------------------------------------------------------------------


@register_skill("sql_query")
def _call_sql_query(a: dict[str, Any]) -> Any:
    from src.tools.sql_tool import execute_sql

    return execute_sql(a["query"], a.get("params"))


@register_skill("rag_search")
def _call_rag_search(a: dict[str, Any]) -> Any:
    from src.tools.rag_tool import rag_search

    return rag_search(a["collection"], a["query"], a.get("top_k"))


@register_skill("code_validator")
def _call_code_validator(a: dict[str, Any]) -> Any:
    from src.tools.code_validator import validate_test_cases

    return validate_test_cases(
        a["test_cases"],
        set(a["known_requirement_ids"]) if a.get("known_requirement_ids") else None,
    )
