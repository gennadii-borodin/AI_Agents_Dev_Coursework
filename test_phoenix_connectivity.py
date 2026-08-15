"""Синтетический тест для проверки записи трейсов в Phoenix."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry import trace
import socket


def check_port(host: str, port: int) -> bool:
    """Проверяет доступность порта."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def main():
    print("=== Phoenix Connectivity Test ===\n")

    hosts = [
        ("host.docker.internal", "Docker Desktop host"),
        ("localhost", "Localhost"),
        ("127.0.0.1", "Loopback"),
        ("172.19.0.2", "Docker container IP"),
    ]

    grpc_port = 4317
    http_port = 6006

    print("Checking ports...\n")
    for host, desc in hosts:
        grpc_ok = check_port(host, grpc_port)
        http_ok = check_port(host, http_port)
        status_grpc = "OK" if grpc_ok else "CLOSED"
        status_http = "OK" if http_ok else "CLOSED"
        print(f"  {desc} ({host}):")
        print(f"    gRPC {grpc_port}: {status_grpc}")
        print(f"    HTTP {http_port}: {status_http}")

    print()

    for host, desc in hosts:
        if check_port(host, grpc_port):
            print(f"\n=== Testing trace export to {host}:{grpc_port} ===\n")

            resource = Resource.create({"service.name": "test-client"})
            exporter = OTLPSpanExporter(
                endpoint=f"{host}:{grpc_port}",
                insecure=True,
                timeout=5,
            )

            provider = TracerProvider(resource=resource)
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            trace.set_tracer_provider(provider)

            tracer = trace.get_tracer(__name__)

            with tracer.start_as_current_span("test_span") as span:
                span.set_attribute("test.attribute", "test_value")
                span.add_event("test_event", {"event_attr": "event_value"})
                print("Span created and closed")

            print("Flushing...")
            flushed = trace.get_tracer_provider().force_flush(timeout_millis=3000)
            print(f"Flush result: {'OK' if flushed else 'FAILED'}")

            print("\nChecking Phoenix API for traces...")
            import httpx
            try:
                r = httpx.get(f"http://{host}:{http_port}/v1/projects/default/traces", timeout=5)
                print(f"Phoenix API response: {r.status_code}")
                print(f"Data: {r.text[:200]}")
            except Exception as e:
                print(f"API check failed: {e}")

            break
    else:
        print("No Phoenix endpoint reachable. Tracing will be disabled.")


if __name__ == "__main__":
    main()
