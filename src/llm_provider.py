import json
import logging
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings, get_settings
from src.tracing import (
    OTEL_AVAILABLE,
    set_llm_output,
    set_span_output,
    set_span_tokens,
    trace_llm,
    trace_tool,
)
from src.tracing import (
    otel_trace as _otel_trace,
)

logger = logging.getLogger(__name__)


def _coerce_tool_results(tool_results: list[str]) -> str:
    """Собирает результаты нескольких вызовов инструментов в единый валидный JSON.

    ``invoke_with_tools`` может сделать несколько раундов tool_calls; результаты
    каждого склеиваются. Простая склейка через ``\\n`` даёт невалидный JSON
    («Extra data»), поэтому здесь каждый результат парсится и объединяется
    в плоский список (для rag_search) либо список объектов.
    """
    combined: list = []
    for tr in tool_results:
        try:
            obj = json.loads(tr)
        except Exception:
            continue
        if isinstance(obj, list):
            combined.extend(obj)
        elif isinstance(obj, dict):
            combined.append(obj)
    return json.dumps(combined, ensure_ascii=False)


def _on_retry(retry_state):
    """Добавляет событие повтора на текущий спан LLM (видно в трейсе)."""
    if not OTEL_AVAILABLE or _otel_trace is None:
        return
    span = _otel_trace.get_current_span()
    if span is None or not span.is_recording():
        return
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    span.add_event(
        "llm_retry",
        {
            "attempt": retry_state.attempt_number,
            "error": str(exc) if exc else "",
        },
    )


class RouterAIProvider:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.routerai_base_url
        self.headers = {
            "Authorization": f"Bearer {self.settings.router_ai_api_key}",
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(get_settings().llm_retry_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=_on_retry,
    )
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        response_format: Optional[dict[str, Any]] = None,
    ) -> str:
        model = model or self.settings.model_senior
        temperature = self.settings.llm_temperature if temperature is None else temperature
        max_tokens = self.settings.llm_max_tokens if max_tokens is None else max_tokens
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}

        with trace_llm(
            model, messages, temperature, max_tokens, bool(response_format or json_mode)
        ) as span:
            try:
                content, usage = self._do_request(payload)
            except Exception:
                # Модель/провайдер может не поддерживать strict json_schema —
                # откатываемся к json_object, затем (при повторном пустом
                # ответе) к обычному режиму без response_format. Пустой
                # контент для больших промптов (напр. Standards) иногда
                # отдаётся именно в json-режимах, а в plain-режиме модель
                # возвращает текст.
                if response_format is not None and response_format.get("type") == "json_schema":
                    payload.pop("response_format", None)
                    payload["response_format"] = {"type": "json_object"}
                    content, usage = self._do_request(payload)
                elif response_format is not None:
                    payload.pop("response_format", None)
                    content, usage = self._do_request(payload)
                else:
                    raise
            set_span_output(span, content, mime_type="text/plain")
            set_llm_output(span, content)
            if usage:
                set_span_tokens(
                    span,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    model=model,
                )
            else:
                set_span_tokens(
                    span,
                    prompt_tokens=len(json.dumps({"messages": messages}, ensure_ascii=False)) // 4,
                    completion_tokens=len(content) // 4,
                    model=model,
                )
            return content

    def _do_request(self, payload: dict[str, Any]) -> tuple[str, dict]:
        try:
            with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"].get("content")
                if content is None or not str(content).strip():
                    # Пустой ответ провайдера. Логируем finish_reason/usage,
                    # чтобы диагностировать причину (обрыв по длине, отказ
                    # модели сгенерировать JSON для слишком большого промпта и т.п.).
                    finish = data["choices"][0].get("finish_reason")
                    logger.warning(
                        "LLM returned empty content (finish_reason=%s, usage=%s, model=%s)",
                        finish,
                        data.get("usage"),
                        payload.get("model"),
                    )
                    # Бросаем, чтобы tenacity сделал повторную попытку и
                    # сработал фоллбэк json_schema -> json_object -> plain.
                    raise ValueError("LLM provider returned empty content")
                usage = data.get("usage", {}) or {}
                return str(content).strip(), usage
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def invoke_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_iterations: int = 5,
        return_tool_results: bool = False,
        tool_choice: Any = "auto",
    ) -> str:
        """LLM с function-calling по скиллам из ToolRegistry.

        Выполняет до ``max_iterations`` раундов вызовов инструментов, прокручивая
        tool_calls через реестр скиллов. При ``return_tool_results=True`` возвращает
        сырые результаты инструментов (для извлечения данных агентами), иначе —
        итоговый ответ ассистента.
        """
        from src.skills import ToolRegistry

        if not tools:
            return self.chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                model=model,
            )

        model = model or self.settings.model_senior
        registry = ToolRegistry()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        tool_results: list[str] = []
        last_content = ""

        for _ in range(max_iterations):
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "temperature": self.settings.llm_temperature,
            }
            try:
                with trace_llm(
                    model, messages, self.settings.llm_temperature, self.settings.llm_max_tokens, False
                ) as span:
                    with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                        response = client.post(
                            f"{self.base_url}/chat/completions",
                            headers=self.headers,
                            json=payload,
                        )
                        response.raise_for_status()
                        data = response.json()
                    msg = data["choices"][0]["message"]
                    content = (msg.get("content") or "").strip()
                    usage = data.get("usage", {}) or {}
                    if usage:
                        set_span_tokens(
                            span,
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=int(usage.get("completion_tokens") or 0),
                            model=model,
                        )
                    set_span_output(
                        span,
                        content or f"[tool_calls={len(msg.get('tool_calls') or [])}]",
                        mime_type="text/plain",
                    )
                    set_llm_output(span, content)

                    tool_calls = msg.get("tool_calls") or []
                    if not tool_calls:
                        if return_tool_results:
                            # Нет вызовов инструментов — возвращаем пустой список,
                            # чтобы вызывающий код получил валидный JSON.
                            return _coerce_tool_results(tool_results) if tool_results else "[]"
                        return content
                    last_content = content
                    messages.append(msg)
                    for tc in tool_calls:
                        fn = tc["function"]
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        with trace_tool(f"tool:{fn['name']}", {"args": args}) as tspan:
                            result = registry.execute_to_json(fn["name"], args)
                            if tspan is not None:
                                tspan.set_attribute("tool.name", fn["name"])
                                set_span_output(tspan, result[:2000])
                        tool_results.append(result)
                        messages.append(
                            {"role": "tool", "content": result, "tool_call_id": tc["id"]}
                        )
            except Exception as e:
                logger.error(f"invoke_with_tools failed: {e}")
                raise

        if return_tool_results and tool_results:
            return _coerce_tool_results(tool_results)
        return last_content
