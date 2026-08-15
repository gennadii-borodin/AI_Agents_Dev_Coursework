import os
import time
import uuid
import logging
import socket
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from openinference.semconv.trace import (
        SpanAttributes,
        OpenInferenceSpanKindValues,
    )
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    otel_trace = None
    SpanAttributes = None
    OpenInferenceSpanKindValues = None


if not OTEL_AVAILABLE:
    class _DummyAttrs:
        def __getattr__(self, name):
            return f"unknown.{name}"
    SpanAttributes = _DummyAttrs()
    OpenInferenceSpanKindValues = _DummyAttrs()


_PROVIDER: Optional[TracerProvider] = None
_TRACER = None
_CURRENT_SESSION_ID: Optional[str] = None
_PHOENIX_INITIALIZED = False


def _resolve_phoenix_endpoint() -> str:
    """Определяем доступный gRPC endpoint для Phoenix."""
    grpc_port = int(os.getenv("PHOENIX_GRPC_PORT", "4317"))

    hosts_to_try = ["host.docker.internal", "localhost", "127.0.0.1"]

    for host in hosts_to_try:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, grpc_port))
            sock.close()
            if result == 0:
                return f"{host}:{grpc_port}"
        except Exception:
            pass

    return ""


def init_phoenix(project_name: Optional[str] = None) -> bool:
    """Инициализация Phoenix tracing (опционально).

    Если Phoenix недоступен, приложение продолжает работу без трассировки.
    Каждая сессия ревью создаётся отдельным вызовом run_review и получает
    собственный session.id, поэтому все спаны одного прогона попадают в одну
    сессию/трейс.
    """
    global _PROVIDER, _TRACER, _PHOENIX_INITIALIZED

    if _PHOENIX_INITIALIZED:
        return _PROVIDER is not None

    endpoint = _resolve_phoenix_endpoint()
    if not endpoint:
        logger.debug("Phoenix endpoint not reachable, tracing disabled")
        _PHOENIX_INITIALIZED = True
        return False

    try:
        resource = Resource.create({
            "service.name": project_name or "qa-review-agent",
        })

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True, timeout=3)

        _PROVIDER = TracerProvider(resource=resource)
        _PROVIDER.add_span_processor(SimpleSpanProcessor(exporter))
        otel_trace.set_tracer_provider(_PROVIDER)
        _TRACER = otel_trace.get_tracer("qa-review-agent")

        try:
            from phoenix.trace import wrap_openai
            wrap_openai()
        except Exception:
            pass

        _PHOENIX_INITIALIZED = True
        logger.info("Phoenix tracing initialized")
        logger.info("View traces at http://localhost:6006")
        return True
    except Exception as e:
        logger.debug(f"Phoenix tracing not available: {e}")
        _PHOENIX_INITIALIZED = True
        return False


def get_tracer():
    if _TRACER is not None:
        return _TRACER
    if OTEL_AVAILABLE:
        return otel_trace.get_tracer("qa-review-agent")
    return None


def new_session_id(project_name: str = "qa-review") -> str:
    return f"{project_name}-{uuid.uuid4().hex[:10]}"


def get_current_session_id() -> str:
    return _CURRENT_SESSION_ID or "no-session"


def _flush():
    if _PROVIDER is not None:
        try:
            _PROVIDER.force_flush(timeout_millis=2000)
        except Exception:
            pass


@contextmanager
def trace_span(
    name: str,
    kind: Optional[Any] = None,
    session_id: Optional[str] = None,
    attributes: Optional[dict] = None,
):
    """Универсальный контекст-менеджер спана с авто-родительством и статусом."""
    if not OTEL_AVAILABLE or _TRACER is None:
        yield None
        return

    attrs: dict = dict(attributes or {})
    if kind is not None:
        kind_val = kind.value if hasattr(kind, "value") else kind
        attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] = kind_val

    sid = session_id or _CURRENT_SESSION_ID
    if sid and SpanAttributes.SESSION_ID not in attrs:
        attrs[SpanAttributes.SESSION_ID] = sid

    with _TRACER.start_as_current_span(name) as span:
        for k, v in attrs.items():
            try:
                span.set_attribute(k, v)
            except Exception:
                pass
        start = time.perf_counter()
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            span.set_attribute("duration_ms", round((time.perf_counter() - start) * 1000, 2))


@contextmanager
def trace_run(query: str, scenario: str, agents: list[str], session_id: Optional[str] = None):
    """Корневой спан всего прогона ревью — одна сессия на запуск."""
    global _CURRENT_SESSION_ID
    sid = session_id or new_session_id()
    _CURRENT_SESSION_ID = sid
    with trace_span(
        "QA Review Run",
        kind=OpenInferenceSpanKindValues.AGENT,
        session_id=sid,
        attributes={
            SpanAttributes.INPUT_VALUE: query,
            SpanAttributes.INPUT_MIME_TYPE: "text/plain",
            "qa.scenario": scenario,
            "qa.agents": ",".join(agents),
        },
    ) as span:
        try:
            yield span
        finally:
            _flush()


@contextmanager
def trace_agent(name: str, **attributes):
    with trace_span(
        name,
        kind=OpenInferenceSpanKindValues.AGENT,
        attributes=attributes,
    ) as span:
        yield span


@contextmanager
def trace_llm(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
):
    """Спан LLM-вызова с семантикой OpenInference."""
    invocation = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": "json_object" if json_mode else "text",
    }
    input_value = {
        "messages": [
            {"role": m.get("role"), "content": (m.get("content") or "")[:4000]}
            for m in messages
        ]
    }
    with trace_span(
        f"LLM {model.split('/')[-1]}",
        kind=OpenInferenceSpanKindValues.LLM,
        attributes={
            SpanAttributes.LLM_MODEL_NAME: model,
            SpanAttributes.LLM_PROVIDER: "routerai",
            SpanAttributes.LLM_INVOCATION_PARAMETERS: _json(invocation),
            SpanAttributes.INPUT_VALUE: _json(input_value),
            SpanAttributes.INPUT_MIME_TYPE: "application/json",
        },
    ) as span:
        yield span


@contextmanager
def trace_tool(name: str, parameters: dict, tool_type: str = "TOOL"):
    """Спан вызова инструмента (SQL, RAG)."""
    kind = getattr(OpenInferenceSpanKindValues, tool_type, OpenInferenceSpanKindValues.TOOL)
    with trace_span(
        f"Tool {name}",
        kind=kind,
        attributes={
            SpanAttributes.TOOL_NAME: name,
            SpanAttributes.TOOL_PARAMETERS: _json(parameters),
        },
    ) as span:
        yield span


@contextmanager
def trace_embedding(model: str, text: str):
    """Спан запроса эмбеддинга."""
    with trace_span(
        f"Embedding {model.split('/')[-1]}",
        kind=OpenInferenceSpanKindValues.EMBEDDING,
        attributes={
            SpanAttributes.EMBEDDING_MODEL_NAME: model,
            SpanAttributes.INPUT_VALUE: text[:4000],
            SpanAttributes.INPUT_MIME_TYPE: "text/plain",
            "embedding.input_chars": len(text),
        },
    ) as span:
        yield span


@contextmanager
def trace_retriever(collection: str, query: str, top_k: int):
    """Спан семантического поиска (retriever)."""
    with trace_span(
        f"Retrieve {collection}",
        kind=OpenInferenceSpanKindValues.RETRIEVER,
        attributes={
            SpanAttributes.INPUT_VALUE: query,
            "retriever.collection": collection,
            "retriever.top_k": top_k,
        },
    ) as span:
        yield span


def set_span_output(span, output: Any, mime_type: str = "text/plain"):
    if span is None:
        return
    try:
        if isinstance(output, (dict, list)):
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, _json(output))
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
        else:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(output)[:8000])
            span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, mime_type)
    except Exception:
        pass


def set_span_tokens(span, prompt_tokens: int, completion_tokens: int):
    if span is None:
        return
    try:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL, prompt_tokens + completion_tokens
        )
    except Exception:
        pass


def set_retrieval_documents(span, documents: list[dict]):
    if span is None or not documents:
        return
    try:
        span.set_attribute(
            SpanAttributes.RETRIEVAL_DOCUMENTS,
            _json([
                {
                    "id": d.get("id"),
                    "score": d.get("similarity"),
                    "document": {
                        "content": (d.get("content") or "")[:2000],
                        "metadata": {
                            "title": d.get("title"),
                            "category": d.get("category"),
                            "priority": d.get("priority"),
                        },
                    },
                }
                for d in documents
            ]),
        )
    except Exception:
        pass


def _json(obj) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)
