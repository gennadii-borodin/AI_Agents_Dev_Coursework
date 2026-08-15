"""Tests that run-level metrics are emitted even when the graph fails (revью §9).

Ранее блок установки метрик (cost/токены) находился ПОСЛЕ finally и был
недостижим при исключении в graph.invoke — стоимость/токены терялись. Теперь
метрики фиксируются внутри finally, включая статус прогона и тип ошибки.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import pytest

import src.report as report_mod
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


class _FakeGraph:
    def invoke(self, state, config=None):
        raise RuntimeError("boom")

    def get_state(self, config):
        return None


def test_run_metrics_emitted_on_error(memory_exporter, monkeypatch):
    monkeypatch.setattr(report_mod, "build_graph", lambda cp: _FakeGraph())
    saved = []
    monkeypatch.setattr(report_mod, "save_reports", lambda state: saved.append(state) or {})

    with pytest.raises(RuntimeError):
        report_mod.run_review("cover REQ-001 requirement")

    memory_exporter.force_flush(2000)
    spans = memory_exporter.get_finished_spans()
    run = next(s for s in spans if s.name == "QA Review Run")
    assert run.attributes.get("qa.run_status") == "error"
    assert run.attributes.get("qa.error_type") == "RuntimeError"
    assert saved, "save_reports должен вызываться в finally даже при ошибке"
