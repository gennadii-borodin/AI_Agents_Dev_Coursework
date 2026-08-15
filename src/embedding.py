import functools
import logging
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings, get_settings
from src.tracing import set_span_output, trace_embedding

logger = logging.getLogger(__name__)

# Ручной кэш эмбеддингов по (text, model). Используется ручной dict (а не
# lru_cache на функции с сетью), чтобы НЕ кэшировать исключения и не ломать
# повторные попытки (§4 ревью: нет кэша эмбеддингов — 21 вызов/run).
_embed_cache: dict[tuple[str, str], list[float]] = {}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _embed_uncached(text: str, model: str, settings: Settings) -> list[float]:
    headers = {
        "Authorization": f"Bearer {settings.router_ai_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=settings.embedding_timeout_seconds) as client:
        response = client.post(
            f"{settings.routerai_base_url}/embeddings",
            headers=headers,
            json={"model": model, "input": text},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


class EmbeddingProvider:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.routerai_base_url
        self.headers = {
            "Authorization": f"Bearer {self.settings.router_ai_api_key}",
            "Content-Type": "application/json",
        }

    def embed_text(self, text: str, model: Optional[str] = None) -> list[float]:
        model = model or self.settings.model_embedding
        cache_key = (text, model)
        cached = _embed_cache.get(cache_key)
        hit = cached is not None
        if not hit:
            cached = _embed_uncached(text, model, self.settings)
            _embed_cache[cache_key] = cached

        try:
            with trace_embedding(model, text) as span:
                set_span_output(
                    span,
                    {"dimensions": len(cached), "cache_hit": hit},
                    mime_type="application/json",
                )
        except Exception:
            pass
        return cached

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self.embed_text(doc) for doc in documents]
