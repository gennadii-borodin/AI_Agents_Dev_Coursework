import os
import logging

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


def init_phoenix():
    try:
        import phoenix.trace as pxtrace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        pxtrace.set_tracer_provider(provider)
        pxtrace.wrap_openai()

        logger.info("Phoenix tracing initialized. Traces sent to localhost:6006")
        return True
    except Exception as e:
        logger.warning(f"Phoenix tracing not available: {e}")
        return False


def main():
    phoenix_ok = init_phoenix()

    from src.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
