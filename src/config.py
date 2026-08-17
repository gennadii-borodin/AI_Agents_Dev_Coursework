from functools import lru_cache
from typing import Optional

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    router_ai_api_key: str

    # Параметры подключения к БД. Креды вынесены в отдельные переменные
    # .env/окружения; полный DATABASE_URL собирается из них (см. database_url).
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "qa_user"
    db_password: str = "qa_password"
    db_name: str = "qa_review"

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # Базовый URL LLM- и embedding-провайдера (RouterAI). Переопределяется
    # через ROUTERAI_BASE_URL, если нужен прокси/staging/другой шлюз.
    routerai_base_url: str = "https://routerai.ru/api/v1"

    # Phoenix (Arize) — OTLP/gRPC эндпоинт для трассировки.
    phoenix_grpc_port: int = 4317
    phoenix_grpc_endpoint: Optional[str] = None

    model_senior: str = "deepseek/deepseek-v4-pro-0813"
    model_junior: str = "deepseek/deepseek-v4-flash-0731"
    model_embedding: str = "openai/text-embedding-3-small"

    embedding_dimension: int = 1536
    rag_top_k: int = 10

    # Параметры генерации LLM (переопределяются через .env).
    llm_temperature: float = 0.1
    llm_max_tokens: int = 16384
    llm_timeout_seconds: int = 60
    embedding_timeout_seconds: int = 30
    llm_retry_attempts: int = 3

    # Переключатели избыточных LLM-вызовов (revью §2/§3, Этап 3).
    # router_llm_enabled: роутер-LLM избыточен (его scenario отбрасывается,
    #   REQ-IDs и так даёт regex). По умолчанию True (поведение сохранено),
    #   False => чистый детерминированный роутинг без LLM-вызова.
    router_llm_enabled: bool = True
    # rag_enabled: RAG fan-out в Coverage дублирует результат SQL req->test.
    #   По умолчанию True (поведение сохранено); False => только SQL-маппинг.
    rag_enabled: bool = True
    # Лимит токенов вывода для Standards-чанков (T4): вместо общего 16384,
    # чтобы модель не дампила огромный список violations.
    standards_max_tokens: int = 4096
    # Верхняя граница числа чанковых итераций Standards-агента (revью T4,
    # защита от runaway-циклов при сбое пагинации): 0/отрицательное = без лимита.
    standards_max_iterations: int = 200
    # Targeted retry в quality_gate (revью §4, Этап 4): при partial-сбое
    # повторно запускаем только упавшие агенты, а не весь прогон.
    targeted_retry_enabled: bool = True
    max_retry_attempts: int = 2

    # Пороги бизнес-логики агентов.
    agents_chunk_size: int = 50
    sql_max_rows: int = 1000
    priority_weights: dict = {
        "Critical": 3,
        "High": 2,
        "Medium": 1,
        "Low": 0.5,
    }
    coverage_risk_high_threshold: float = 80.0
    coverage_risk_medium_threshold: float = 95.0

    # Стоимость токенов (USD за 1M) по моделям. Переопределяется через
    # MODEL_PRICING в .env (JSON). Значения по умолчанию — публичные цены
    # DeepSeek V3; замените на актуальные цены RouterAI.
    model_pricing: dict = {
        "deepseek/deepseek-v4-pro-0813": {"input": 0.27, "output": 1.10},
        "deepseek/deepseek-v4-flash-0731": {"input": 0.10, "output": 0.40},
        "openai/text-embedding-3-small": {"input": 0.02, "output": 0.0},
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
