from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    router_ai_api_key: str
    openai_api_key: str
    database_url: str
    phoenix_port: int = 6006
    phoenix_http_endpoint: str = "http://localhost:6006/v1/traces"

    model_senior: str = "deepseek/deepseek-v4-pro-0813"
    model_junior: str = "deepseek/deepseek-v4-flash-0731"
    model_embedding: str = "openai/text-embedding-3-small"

    embedding_dimension: int = 1536
    rag_top_k: int = 10

    llm_retry_attempts: int = 3
    llm_timeout_seconds: int = 60
    max_output_tokens: int = 4096

    # Стоимость токенов (USD за 1M) по моделям. Переопределяется через
    # MODEL_PRICING в .env. Значения по умолчанию — публичные цены DeepSeek V3;
    # замените на актуальные цены RouterAI.
    model_pricing: dict = {
        "deepseek/deepseek-v4-pro-0813": {"input": 0.27, "output": 1.10},
        "deepseek/deepseek-v4-flash-0731": {"input": 0.10, "output": 0.40},
        "openai/text-embedding-3-small": {"input": 0.02, "output": 0.0},
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
