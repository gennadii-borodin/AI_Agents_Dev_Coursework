"""Примитивы устойчивости для вызовов инструментов (skills).

Реализованы на стандартной библиотеке и предназначены для защиты
инструментов с доступом к внешним сервисам (БД, pgvector, эмбеддинги)
от runaway-вызовов, каскадных сбоев и превышения частоты обращений.

Состояние — in-process (на экземпляр ToolRegistry). Для многопроцессного
деплоя примитивы следует заменить на распределённые (Redis) аналоги.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Tuple


class RateLimiter:
    """Простой in-memory token bucket с окном в 60 секунд.

    ``max_calls_per_minute=None`` отключает ограничение (всегда разрешено).
    """

    def __init__(self, max_calls_per_minute: Optional[int]) -> None:
        self.limit = max_calls_per_minute
        self._lock = threading.Lock()
        self._calls: list[float] = []

    def allow(self) -> bool:
        if not self.limit:
            return True
        now = time.monotonic()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < 60]
            if len(self._calls) >= self.limit:
                return False
            self._calls.append(now)
            return True


class CircuitBreaker:
    """Circuit breaker: после ``failure_threshold`` последовательных сбоев
    переходит в состояние OPEN на ``cooldown_seconds``; в этом состоянии
    вызовы блокируются без обращения к инструменту.

    ``failure_threshold=None`` отключает breaker (всегда разрешено).
    """

    def __init__(
        self, failure_threshold: Optional[int], cooldown_seconds: int
    ) -> None:
        self.threshold = failure_threshold
        self.cooldown = cooldown_seconds
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at = 0.0

    def allow(self) -> bool:
        if not self.threshold:
            return True
        now = time.monotonic()
        with self._lock:
            if self._opened_at and (now - self._opened_at) < self.cooldown:
                return False
            # Cooldown истёк — возвращаемся в нормальное состояние (half-open).
            if self._opened_at:
                self._opened_at = 0.0
                self._failures = 0
            return True

    def record_failure(self) -> None:
        if not self.threshold:
            return
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = time.monotonic()
                self._failures = 0

    def record_success(self) -> None:
        if not self.threshold:
            return
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0


def run_with_timeout(
    func: Callable[..., Any], timeout: Optional[float], *args: Any, **kwargs: Any
) -> Any:
    """Выполняет ``func`` с ограничением по времени.

    ``timeout=None/0`` — выполнение синхронно без дополнительного потока.
    При превышении времени поднимается ``TimeoutError`` (задача в потоке
    помечается на отмену, но блокирующий системный вызов может завершиться
    позже — поэтому для БД рекомендуется также statement/connect_timeout).
    """
    if not timeout:
        return func(*args, **kwargs)

    # Переносим OTel-контекст (contextvars) в воркер-поток. Иначе спаны,
    # созданные внутри func (напр. RETRIEVER/EMBEDDING в rag_search), не видят
    # родительский спан и экспортируются как отдельный «корневой» трейс
    # (в Phoenix это выглядит как второй, висящий трейс). ThreadPoolExecutor
    # контекстvars не пробрасывает, поэтому делаем это явно.
    try:
        from opentelemetry.context import get_current, attach, detach

        otel_ctx = get_current()
        _propagate = True
    except Exception:
        otel_ctx = None
        _propagate = False

    def _run() -> Any:
        if _propagate and otel_ctx is not None:
            token = attach(otel_ctx)
            try:
                return func(*args, **kwargs)
            finally:
                detach(token)
        return func(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=timeout)
        except Exception:
            future.cancel()
            raise


__all__ = ["CircuitBreaker", "RateLimiter", "run_with_timeout"]
