import logging
from typing import Any, Optional

from src.config import Settings, get_settings
from src.embedding import EmbeddingProvider
from src.tools.sql_tool import get_connection
from src.tracing import (
    set_retrieval_documents,
    set_span_output,
    trace_retriever,
)

logger = logging.getLogger(__name__)


def rag_search(
    collection: str,
    query: str,
    top_k: int = 10,
    settings: Optional[Settings] = None,
) -> list[dict[str, Any]]:
    if settings is None:
        settings = get_settings()

    embedding_provider = EmbeddingProvider(settings)

    try:
        with trace_retriever(collection, query, top_k) as span:
            try:
                query_embedding = embedding_provider.embed_text(query)
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                return []

            table_name = collection
            if table_name not in ("requirements", "test_cases"):
                logger.warning(f"Unknown collection: {collection}")
                return []

            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            search_query = f"""
            SELECT
                id,
                (1 - (embedding <=> %s::vector)) as sim_score,
                {", ".join([
                    "requirement_id" if table_name == "requirements" else "test_case_id",
                    "title",
                    "requirement_text" if table_name == "requirements" else "description",
                    "category" if table_name == "requirements" else "test_type",
                    "priority",
                ])}
            FROM {table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """

            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(search_query, (embedding_str, embedding_str, top_k))
                        rows = cur.fetchall()
                        results = []
                        for row in rows:
                            row_dict = dict(row)
                            results.append({
                                "id": row_dict.get("requirement_id") or row_dict.get("test_case_id"),
                                "title": row_dict.get("title", ""),
                                "content": row_dict.get("requirement_text") or row_dict.get("description", ""),
                                "category": row_dict.get("category", ""),
                                "priority": row_dict.get("priority", ""),
                                "similarity": float(row_dict.get("sim_score", 0)),
                            })
                        set_retrieval_documents(span, results)
                        set_span_output(span, {"count": len(results)})
                        return results
            except Exception as e:
                logger.error(f"RAG search failed: {e}")
                return []
    except Exception:
        return []


def rag_search_by_requirement(
    requirement_id: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    search_query = """
    SELECT tc.test_case_id, tc.title, tc.description, tc.test_type, tc.priority, tc.req
    FROM test_cases tc
    WHERE tc.req = %s
    ORDER BY tc.test_case_id
    LIMIT %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(search_query, (requirement_id, top_k))
                rows = cur.fetchall()
                return [
                    {
                        "id": row["test_case_id"],
                        "title": row["title"],
                        "content": row["description"],
                        "test_type": row["test_type"],
                        "priority": row["priority"],
                        "req": row["req"],
                    }
                    for row in rows
                ]
    except Exception as e:
        logger.error(f"RAG search by requirement failed: {e}")
        return []


def rag_search_related_requirements(
    requirement_text: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    return rag_search("requirements", requirement_text, top_k)
