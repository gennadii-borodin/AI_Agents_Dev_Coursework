import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.embedding import EmbeddingProvider
from src.tools.sql_tool import get_connection


def backfill():
    settings = get_settings()
    provider = EmbeddingProvider(settings)

    # Requirements
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT requirement_id, title, requirement_text FROM requirements WHERE embedding IS NULL")
            reqs = cur.fetchall()
        print(f"Requirements needing embeddings: {len(reqs)}")
        for r in reqs:
            rid, title, text = r["requirement_id"], r["title"], r["requirement_text"]
            emb = provider.embed_text(f"{title} {text}")
            with get_connection() as c2:
                with c2.cursor() as c2c:
                    c2c.execute(
                        "UPDATE requirements SET embedding = %s::vector WHERE requirement_id = %s",
                        (emb, rid),
                    )
                c2.commit()

    # Test cases
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT test_case_id, title, description FROM test_cases WHERE embedding IS NULL")
            tcs = cur.fetchall()
        print(f"Test cases needing embeddings: {len(tcs)}")
        for tc in tcs:
            tcid, title, desc = tc["test_case_id"], tc["title"], tc["description"]
            emb = provider.embed_text(f"{title} {desc}")
            with get_connection() as c2:
                with c2.cursor() as c2c:
                    c2c.execute(
                        "UPDATE test_cases SET embedding = %s::vector WHERE test_case_id = %s",
                        (emb, tcid),
                    )
                c2.commit()

    print("Backfill complete.")


if __name__ == "__main__":
    backfill()
