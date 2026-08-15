import logging
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.config import Settings, get_settings
from src.tracing import trace_tool, set_span_output

logger = logging.getLogger(__name__)


def get_connection() -> psycopg.Connection:
    settings = get_settings()
    return psycopg.connect(settings.database_url, row_factory=dict_row)


def execute_sql(query: str, params: Optional[list] = None, settings: Optional[Settings] = None) -> dict[str, Any]:
    if settings is None:
        settings = get_settings()

    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
    query_upper = query.upper().strip()
    for keyword in forbidden_keywords:
        if keyword in query_upper:
            return {"results": [], "row_count": 0, "error": f"Forbidden keyword: {keyword}"}

    try:
        with trace_tool("execute_sql", {"query": query, "params": params}) as span:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    if params:
                        cur.execute(query, params)
                    else:
                        cur.execute(query)
                    rows = cur.fetchall()
                    result = {
                        "results": [dict(row) for row in rows],
                        "row_count": len(rows),
                        "error": None,
                    }
                    if span is not None:
                        span.set_attribute("sql.row_count", len(rows))
                    set_span_output(span, {"row_count": len(rows)})
                    return result
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        return {"results": [], "row_count": 0, "error": str(e)}


def get_all_requirements() -> list[dict[str, Any]]:
    result = execute_sql("SELECT * FROM requirements ORDER BY requirement_id")
    return result["results"] if result["error"] is None else []


def get_all_test_cases() -> list[dict[str, Any]]:
    result = execute_sql("SELECT * FROM test_cases ORDER BY test_case_id")
    return result["results"] if result["error"] is None else []


def get_requirements_by_ids(requirement_ids: list[str]) -> list[dict[str, Any]]:
    if not requirement_ids:
        return []
    placeholders = ",".join(["%s"] * len(requirement_ids))
    query = f"SELECT * FROM requirements WHERE requirement_id IN ({placeholders}) ORDER BY requirement_id"
    result = execute_sql(query, requirement_ids)
    return result["results"] if result["error"] is None else []


def get_test_cases_by_req(requirement_id: str) -> list[dict[str, Any]]:
    query = "SELECT * FROM test_cases WHERE req = %s ORDER BY test_case_id"
    result = execute_sql(query, [requirement_id])
    return result["results"] if result["error"] is None else []


def get_tests_without_requirements() -> list[dict[str, Any]]:
    query = "SELECT * FROM test_cases WHERE req IS NULL OR req = '' ORDER BY test_case_id"
    result = execute_sql(query)
    return result["results"] if result["error"] is None else []


def get_coverage_stats() -> dict[str, Any]:
    query = """
    SELECT
        r.priority,
        COUNT(DISTINCT r.requirement_id) as total_reqs,
        COUNT(DISTINCT tc.test_case_id) as total_tests
    FROM requirements r
    LEFT JOIN test_cases tc ON r.requirement_id = tc.req
    GROUP BY r.priority
    """
    result = execute_sql(query)
    if result["error"] or not result["results"]:
        return {}

    stats = {}
    for row in result["results"]:
        stats[row["priority"]] = {
            "total_reqs": row["total_reqs"],
            "total_tests": row["total_tests"],
        }
    return stats
