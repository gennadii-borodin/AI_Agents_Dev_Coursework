import json
import logging
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings, get_settings
from src.tracing import (
    trace_llm,
    set_span_output,
    set_span_tokens,
    set_llm_output,
    OTEL_AVAILABLE,
    otel_trace as _otel_trace,
)

logger = logging.getLogger(__name__)


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
        self.base_url = "https://routerai.ru/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.settings.router_ai_api_key}",
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=_on_retry,
    )
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        model = model or self.settings.model_senior
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 8192,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with trace_llm(model, messages, temperature, max_tokens or 8192, json_mode) as span:
            try:
                content, usage = self._do_request(payload)
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
            except Exception as e:
                raise

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
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {}) or {}
                return content.strip(), usage
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def invoke_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if tools:
            payload: dict[str, Any] = {
                "model": model or self.settings.model_senior,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.1,
            }

            with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                tool_calls = data["choices"][0]["message"].get("tool_calls", [])
                if tool_calls:
                    results = []
                    for tool_call in tool_calls:
                        func = tool_call["function"]
                        tool_name = func["name"]
                        tool_args = json.loads(func["arguments"])
                        tool_result = self._execute_tool(tool_name, tool_args)
                        results.append(f"Tool {tool_name} result: {tool_result}")

                    messages.append(data["choices"][0]["message"])
                    for result in results:
                        messages.append({"role": "tool", "content": result, "tool_call_id": tool_calls[0]["id"]})

                    payload["messages"] = messages
                    response2 = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    response2.raise_for_status()
                    data2 = response2.json()
                    return data2["choices"][0]["message"]["content"].strip()

                return data["choices"][0]["message"]["content"].strip()

        return self.chat_completion(messages, model=model)

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        from src.tools.sql_tool import execute_sql
        from src.tools.rag_tool import rag_search

        if tool_name == "execute_sql":
            return execute_sql(args["query"])
        elif tool_name == "rag_search":
            return rag_search(args["collection"], args["query"], args.get("top_k", 10))
        else:
            return f"Unknown tool: {tool_name}"
