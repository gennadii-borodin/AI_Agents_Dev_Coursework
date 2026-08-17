import logging
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row

from src.config import Settings, get_settings
from src.tracing import set_span_output, trace_tool

logger = logging.getLogger(__name__)


def get_connection() -> psycopg.Connection:
    settings = get_settings()
    # Server-side защита от зависших/долгих запросов. Клиентский таймаут
    # (ThreadPoolExecutor) не прерывает блокирующий вызов БД, поэтому
    # statement_timeout — реальный механизм отмены (см. Settings.sql_statement_timeout).
    #
    # Важно: statement_timeout задаётся через опцию подключения `-c`, а не
    # оператором `SET statement_timeout = %s`. У `SET var = $1` psycopg не может
    # вывести тип параметра ("could not determine data type of parameter $1"),
    # команда падает, а т.к. это первый запрос в неявной транзакции — соединение
    # переходит в состояние "current transaction is aborted", и все последующие
    # SELECT на нём проваливаются. Опция `-c` применяется ДО начала транзакции и
    # не может её прервать. Альтернативный вариант — autocommit (см. ниже).
    try:
        conn = psycopg.connect(
            settings.database_url,
            connect_timeout=settings.sql_connect_timeout,
            row_factory=dict_row,
            options=f"-c statement_timeout={settings.sql_statement_timeout}",
        )
    except Exception:  # noqa: BLE001
        # Если опция не поддерживается (старый/проксирующий сервер), подключаемся
        # без неё, чтобы не обрушивать весь пайп БД; защита по таймауту в этом
        # случае ослабляется, но запросы выполняются.
        logger.warning(
            "Не удалось задать statement_timeout через опцию подключения; "
            "подключаемся без неё"
        )
        conn = psycopg.connect(
            settings.database_url,
            connect_timeout=settings.sql_connect_timeout,
            row_factory=dict_row,
        )
    # Read-only нагрузка: autocommit исключает висячие транзакции, поэтому даже
    # при сбое отдельного запроса состояние "transaction aborted" не блокирует
    # следующие вызовы на этом же соединении.
    conn.autocommit = True
    return conn


MAX_SQL_ROWS = 1000

# Явная проекция колонок БЕЗ embedding-вектора (vector(1536)): исключаем
# многомегабайтный мусорный payload из каждого SELECT (T2, §4 ревью).
REQUIREMENT_COLS = (
    "requirement_id, title, requirement_text, category, priority, "
    "qa_requirements_review, rejection_reason"
)
TEST_CASE_COLS = (
    "test_case_id, req, title, description, preconditions, test_data, steps, "
    "expected_result, priority, test_type, design_quality, qa_review, review_comment"
)


def execute_sql(query: str, params: Optional[list] = None, settings: Optional[Settings] = None) -> dict[str, Any]:
    if settings is None:
        settings = get_settings()

    # Read-only контракт (allowlist_prefix/forbidden_keywords/single_statement)
    # enforce-ится в реестре скиллов (src/skills.py:_check_sql) ДО вызова этого
    # метода. Здесь дублировать проверку не нужно — иначе два источника истины
    # расходятся. Внутренние выборки (get_all_*, get_requirements_by_ids, ...)
    # и так формируют только SELECT.

    try:
        with trace_tool("execute_sql", {"query": query, "params": params}) as span:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    if params:
                        cur.execute(query, params)
                    else:
                        cur.execute(query)
                    rows = cur.fetchall()
                    # Ограничение из скилла sql_query.yaml: максимум 1000 строк.
                    truncated = len(rows) > settings.sql_max_rows
                    rows = rows[: settings.sql_max_rows]
                    result = {
                        "results": [dict(row) for row in rows],
                        "row_count": len(rows),
                        "truncated": truncated,
                        "error": None,
                    }
                    if span is not None:
                        span.set_attribute("sql.row_count", len(rows))
                    set_span_output(span, {"row_count": len(rows), "truncated": truncated})
                    return result
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        return {"results": [], "row_count": 0, "error": str(e)}


def get_all_requirements() -> list[dict[str, Any]]:
    result = execute_sql(f"SELECT {REQUIREMENT_COLS} FROM requirements ORDER BY requirement_id")
    return result["results"] if result["error"] is None else []


def get_all_test_cases() -> list[dict[str, Any]]:
    result = execute_sql(f"SELECT {TEST_CASE_COLS} FROM test_cases ORDER BY test_case_id")
    return result["results"] if result["error"] is None else []


def get_requirements_by_ids(requirement_ids: list[str]) -> list[dict[str, Any]]:
    if not requirement_ids:
        return []
    placeholders = ",".join(["%s"] * len(requirement_ids))
    query = f"SELECT {REQUIREMENT_COLS} FROM requirements WHERE requirement_id IN ({placeholders}) ORDER BY requirement_id"
    result = execute_sql(query, requirement_ids)
    return result["results"] if result["error"] is None else []


def get_test_cases_by_reqs(requirement_ids: list[str]) -> list[dict[str, Any]]:
    """Фильтрованная выборка — ровно те кейсы, что относятся к запрошенным REQ.

    Заменяет SELECT * FROM test_cases для целевых запросов (избегаем
    лишних сканирований всей таблицы в каждом агенте).
    """
    if not requirement_ids:
        return []
    placeholders = ",".join(["%s"] * len(requirement_ids))
    query = f"SELECT {TEST_CASE_COLS} FROM test_cases WHERE req IN ({placeholders}) ORDER BY test_case_id"
    result = execute_sql(query, list(requirement_ids))
    return result["results"] if result["error"] is None else []


def get_tests_without_requirements() -> list[dict[str, Any]]:
    query = f"SELECT {TEST_CASE_COLS} FROM test_cases WHERE req IS NULL OR req = '' ORDER BY test_case_id"
    result = execute_sql(query)
    return result["results"] if result["error"] is None else []
