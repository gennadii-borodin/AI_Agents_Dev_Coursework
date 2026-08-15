import logging
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings, get_settings
from src.tracing import set_span_output, trace_embedding

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = "https://routerai.ru/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def embed_text(self, text: str, model: Optional[str] = None) -> list[float]:
        model = model or self.settings.model_embedding
        payload = {
            "model": model,
            "input": text,
        }

        try:
            with trace_embedding(model, text) as span:
                with httpx.Client(timeout=30) as client:
                    response = client.post(
                        f"{self.base_url}/embeddings",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    embedding = data["data"][0]["embedding"]
                    set_span_output(span, {"dimensions": len(embedding)}, mime_type="application/json")
                    return embedding
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self.embed_text(doc) for doc in documents]
