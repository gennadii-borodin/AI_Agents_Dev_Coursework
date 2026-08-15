"""Synthetic tests for tracing attribute shapes (no live Phoenix required).

Phoenix (<=20.2.1) crashes with React errors when OpenInference attributes
have the wrong shape:

* ``retrieval.documents`` must be an ARRAY of JSON strings (Phoenix does
  ``documents.map(...)``); a single JSON string -> "map is not a function".
* ``llm.input_messages`` / ``llm.output_messages`` must be an ARRAY of JSON
  strings for the same reason.
* ``tool.parameters`` must be a plain string (Phoenix parses objects into a
  React child -> React error #31).

These tests build spans with an in-memory exporter and assert the exported
attribute types, so regressions are caught without a full LLM run.
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import src.tracing as tracing


@pytest.fixture
def memory_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    original = tracing._TRACER
    tracing._TRACER = provider.get_tracer("synthetic-test")
    tracing._PROVIDER = provider
    tracing._PHOENIX_INITIALIZED = True
    yield exporter
    tracing._TRACER = original


def _retriever_span(exporter):
    exporter.force_flush(2000)
    spans = exporter.get_finished_spans()
    return next(s for s in spans if s.name == "Retrieve test_cases")


def test_retrieval_documents_is_array(memory_exporter):
    docs = [
        {
            "id": "TC-STORE-0001",
            "content": "login test",
            "title": "Login",
            "category": "functional",
            "priority": "High",
            "similarity": 0.91,
        }
    ]
    with tracing.trace_retriever("test_cases", "REQ-001", 5) as span:
        tracing.set_retrieval_documents(span, docs)

    attr = _retriever_span(memory_exporter).attributes.get("retrieval.documents")
    assert isinstance(attr, (list, tuple)), f"retrieval.documents must be an array, got {type(attr)}"
    assert all(isinstance(d, str) for d in attr), "each document must be a JSON string"


def test_llm_messages_are_arrays(memory_exporter):
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "cover REQ-001"},
    ]
    with tracing.trace_llm("deepseek/deepseek-v4-pro-0813", messages, 0.1, 100, False) as span:
        tracing.set_llm_output(span, '{"ok": true}')

    memory_exporter.force_flush(2000)
    spans = memory_exporter.get_finished_spans()
    llm = next(s for s in spans if s.name == "LLM deepseek-v4-pro-0813")
    in_msgs = llm.attributes.get("llm.input_messages")
    out_msgs = llm.attributes.get("llm.output_messages")
    assert isinstance(in_msgs, (list, tuple)), f"llm.input_messages must be an array, got {type(in_msgs)}"
    assert isinstance(out_msgs, (list, tuple)), f"llm.output_messages must be an array, got {type(out_msgs)}"
    assert all(isinstance(m, str) for m in in_msgs + out_msgs)


def test_tool_parameters_is_string(memory_exporter):
    with tracing.trace_tool("execute_sql", {"query": "SELECT 1", "params": []}):
        pass

    memory_exporter.force_flush(2000)
    spans = memory_exporter.get_finished_spans()
    tool = next(s for s in spans if s.name == "Tool execute_sql")
    attr = tool.attributes.get("tool.parameters")
    assert isinstance(attr, str), f"tool.parameters must be a string, got {type(attr)}"


def test_mask_sensitive_unit():
    assert tracing.mask_sensitive("password=SuperSecret123") == "password=***"
    assert tracing.mask_sensitive("api_key: abc.DEF-123456") == "api_key=***"
    assert tracing.mask_sensitive("contact victim@example.com") == "contact ***"
    assert tracing.mask_sensitive("card 4111 1111 1111 1111") == "card ***"
    assert tracing.mask_sensitive(123) == 123


def test_secret_masked_in_llm_input(memory_exporter):
    messages = [
        {"role": "system", "content": "analyze"},
        {"role": "user", "content": "password=SuperSecret123 in test_data"},
    ]
    with tracing.trace_llm("deepseek/deepseek-v4-pro-0813", messages, 0.1, 100, False) as span:
        tracing.set_llm_output(span, "email victim@example.com done")

    memory_exporter.force_flush(2000)
    spans = memory_exporter.get_finished_spans()
    llm = next(s for s in spans if s.name == "LLM deepseek-v4-pro-0813")
    joined_in = " ".join(llm.attributes.get("llm.input_messages"))
    assert "SuperSecret123" not in joined_in
    assert "password=***" in joined_in
    joined_out = " ".join(llm.attributes.get("llm.output_messages"))
    assert "victim@example.com" not in joined_out
    assert "***" in joined_out


def test_secret_masked_in_tool_input(memory_exporter):
    with tracing.trace_tool("execute_sql", {"query": "SELECT * FROM t WHERE token=abc123XYZ"}):
        pass

    memory_exporter.force_flush(2000)
    spans = memory_exporter.get_finished_spans()
    tool = next(s for s in spans if s.name == "Tool execute_sql")
    inp = tool.attributes.get(tracing.SpanAttributes.INPUT_VALUE)
    assert "abc123XYZ" not in inp
    assert "token=***" in inp
