import logging
import sys
from pathlib import Path
from typing import Optional

import click
import psycopg
from pgvector.psycopg import register_vector
from rich.console import Console
from rich.progress import Progress

from src.config import get_settings
from src.embedding import EmbeddingProvider
from src.tracing import (
    OpenInferenceSpanKindValues,
    get_current_session_id,
    get_run_stats,
    init_phoenix,
    new_session_id,
    set_span_output,
    trace_run,
    trace_span,
)

logger = logging.getLogger(__name__)
console = Console()


def _read_csv_clean(filepath: Path) -> list[dict[str, str]]:
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()
    content = content.replace('\x00', '')
    lines = content.strip().split("\n")
    if not lines:
        return []
    header_line = lines[0].strip().strip('"')
    headers = [h.strip().strip('"') for h in header_line.split(",")]
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = []
        in_quotes = False
        current = ""
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == "," and not in_quotes:
                values.append(current.strip().strip('"'))
                current = ""
            else:
                current += char
        values.append(current.strip().strip('"'))
        row = {}
        for i, h in enumerate(headers):
            row[h] = values[i] if i < len(values) else ""
        rows.append(row)
    return rows


def run_migration(
    data_dir: Optional[Path] = None,
    skip_embeddings: bool = False,
) -> None:
    settings = get_settings()
    data_dir = data_dir or (Path(__file__).parent.parent / "data")

    # Отдельная сессия мониторинга для процесса миграции (Phoenix),
    # независимая от сессий ревью (qa-review-*). LLM-вызовы (эмбеддинги)
    # попадают в эту же сессию как дочерние спаны.
    session_id = new_session_id("db-migration")
    init_phoenix(project_name="qa-migration")

    console.print(
        f"[bold blue]Запуск миграции БД... (session: {session_id})[/bold blue]"
    )
    console.print(f"[cyan]Директория с данными: {data_dir}[/cyan]")

    with trace_run(
        "Миграция данных в PostgreSQL (requirements, test_cases)",
        scenario="db_migration",
        agents=["embedding", "load_data"],
        session_id=session_id,
    ) as run_span:
        try:
            _do_migration(settings, data_dir, skip_embeddings)
        except Exception as e:
            logger.exception("Migration failed")
            console.print(f"[red]Ошибка: {e}[/red]")
            if run_span is not None:
                run_span.set_attribute("qa.run_status", "error")
                run_span.set_attribute("qa.error_type", type(e).__name__)
            sys.exit(1)
        finally:
            _emit_migration_metrics(run_span, settings)


def _do_migration(settings, data_dir, skip_embeddings) -> None:
    with psycopg.connect(settings.database_url) as conn:
        with trace_span("Создание схемы БД", kind=OpenInferenceSpanKindValues.CHAIN) as span:
            migration_path = Path(__file__).parent.parent / "migrations" / "001_initial.sql"
            console.print(f"[cyan]Файл миграции схемы: {migration_path}[/cyan]")
            migration_sql = migration_path.read_text(encoding="utf-8")
            conn.execute(migration_sql)
            console.print("[green]OK: Таблицы созданы[/green]")

            register_vector(conn)
            if span is not None:
                set_span_output(
                    span,
                    {"tables": ["requirements", "test_cases"]},
                    mime_type="application/json",
                )

        embedding_provider = EmbeddingProvider(settings) if not skip_embeddings else None

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM requirements")
            req_count = cur.fetchone()[0]
            if req_count > 0:
                console.print("[yellow]WARN: Данные уже загружены.[/yellow]")
                if embedding_provider:
                    _backfill_embeddings(conn, embedding_provider)
                else:
                    console.print("[green]OK: Пропуск эмбеддингов.[/green]")
                conn.commit()
                return

        requirements_file = data_dir / "requirements.csv"
        if requirements_file.exists():
            console.print(f"[cyan]Загрузка данных из файла: {requirements_file}[/cyan]")
            with trace_span("Загрузка требований", kind=OpenInferenceSpanKindValues.CHAIN) as span:
                _load_requirements(conn, requirements_file, embedding_provider, skip_embeddings)
                console.print("[green]OK: Требования загружены[/green]")
                if span is not None:
                    set_span_output(span, {"file": str(requirements_file)}, mime_type="text/plain")
        else:
            console.print("[red]ERR: Файл requirements.csv не найден[/red]")

        test_cases_file = data_dir / "online_store_test_cases.csv"
        if test_cases_file.exists():
            console.print(f"[cyan]Загрузка данных из файла: {test_cases_file}[/cyan]")
            with trace_span("Загрузка тест-кейсов", kind=OpenInferenceSpanKindValues.CHAIN) as span:
                _load_test_cases(conn, test_cases_file, embedding_provider, skip_embeddings)
                console.print("[green]OK: Тест-кейсы загружены[/green]")
                if span is not None:
                    set_span_output(span, {"file": str(test_cases_file)}, mime_type="text/plain")
        else:
            console.print("[red]✗ Файл online_store_test_cases.csv не найден[/red]")

        conn.commit()
        console.print("[bold green]OK: Миграция завершена![/bold green]")


def _emit_migration_metrics(run_span, settings) -> None:
    """Фиксирует метрики прогона миграции (LLM-вызовы, токены, $) на run-спане."""
    if run_span is None:
        return
    try:
        stats = get_run_stats()
        run_span.set_attribute("qa.run_status", "success")
        run_span.set_attribute("qa.llm_calls", stats["llm_calls"])
        run_span.set_attribute("qa.prompt_tokens", stats["prompt_tokens"])
        run_span.set_attribute("qa.completion_tokens", stats["completion_tokens"])
        run_span.set_attribute("qa.total_tokens", stats["prompt_tokens"] + stats["completion_tokens"])

        pricing = settings.model_pricing
        est_cost = 0.0
        for model, toks in stats.get("by_model", {}).items():
            price = pricing.get(model, {})
            est_cost += (
                toks["prompt"] / 1_000_000 * price.get("input", 0)
                + toks["completion"] / 1_000_000 * price.get("output", 0)
            )
        run_span.set_attribute("qa.estimated_cost_usd", round(est_cost, 4))

        set_span_output(
            run_span,
            {
                "session_id": get_current_session_id(),
                "llm_calls": stats["llm_calls"],
                "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
            },
            mime_type="application/json",
        )
    except Exception:
        logger.exception("Failed to emit migration metrics")
    console.print(f"[cyan]Monitoring session: {get_current_session_id()}[/cyan]")


def _load_requirements(
    conn,
    filepath: Path,
    embeddings: Optional[EmbeddingProvider],
    skip_embeddings: bool,
):
    rows = _read_csv_clean(filepath)

    with Progress() as progress:
        task = progress.add_task("[cyan]Загрузка требований...", total=len(rows))

        batch = []
        for row in rows:
            embedding = None
            if not skip_embeddings and embeddings:
                embedding_text = f"{row['title']} {row['requirement_text']}"
                try:
                    embedding = embeddings.embed_text(embedding_text)
                except Exception as e:
                    logger.warning(f"Failed to embed requirement {row['requirement_id']}: {e}")

            batch.append((
                row["requirement_id"],
                row["title"],
                row["requirement_text"],
                row["category"],
                row["priority"],
                row.get("qa_requirements_review", ""),
                row.get("rejection_reason", ""),
                embedding,
            ))

            if len(batch) >= 50:
                _insert_requirements(conn, batch)
                batch = []

            progress.update(task, advance=1)

        if batch:
            _insert_requirements(conn, batch)


def _load_test_cases(
    conn,
    filepath: Path,
    embeddings: Optional[EmbeddingProvider],
    skip_embeddings: bool,
):
    rows = _read_csv_clean(filepath)

    with Progress() as progress:
        task = progress.add_task("[cyan]Загрузка test cases...", total=len(rows))

        batch = []
        for row in rows:
            embedding = None
            if not skip_embeddings and embeddings:
                embedding_text = f"{row['title']} {row['description']}"
                try:
                    embedding = embeddings.embed_text(embedding_text)
                except Exception as e:
                    logger.warning(f"Failed to embed test case {row['test_case_id']}: {e}")

            batch.append((
                row["test_case_id"],
                row.get("REQ", ""),
                row["title"],
                row["description"],
                row.get("preconditions", ""),
                row.get("test_data", ""),
                row["steps"],
                row["expected_result"],
                row["priority"],
                row["test_type"],
                row.get("design_quality", ""),
                row.get("qa_review", ""),
                row.get("review_comment", ""),
                embedding,
            ))

            if len(batch) >= 50:
                _insert_test_cases(conn, batch)
                batch = []

            progress.update(task, advance=1)

        if batch:
            _insert_test_cases(conn, batch)


def _insert_requirements(conn, batch: list[tuple]):
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO requirements (
                requirement_id, title, requirement_text, category, priority,
                qa_requirements_review, rejection_reason, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (requirement_id) DO NOTHING
            """,
            batch,
        )


def _insert_test_cases(conn, batch: list[tuple]):
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO test_cases (
                test_case_id, req, title, description, preconditions,
                test_data, steps, expected_result, priority, test_type,
                design_quality, qa_review, review_comment, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (test_case_id) DO NOTHING
            """,
            batch,
        )


def _backfill_embeddings(conn, embeddings: EmbeddingProvider) -> None:
    """Дозаполняет эмбеддинги для строк, где embedding IS NULL (повторный запуск миграции)."""
    with conn.cursor(
        row_factory=psycopg.rows.dict_row,
    ) as cur:
        cur.execute(
            "SELECT requirement_id, title, requirement_text "
            "FROM requirements WHERE embedding IS NULL"
        )
        reqs = cur.fetchall()
    if reqs:
        console.print(f"[cyan]Backfill эмбеддингов требований: {len(reqs)}[/cyan]")
        for r in reqs:
            rid, title, text = r["requirement_id"], r["title"], r["requirement_text"]
            try:
                emb = embeddings.embed_text(f"{title} {text}")
            except Exception as e:
                logger.warning(f"Failed to embed requirement {rid}: {e}")
                continue
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE requirements SET embedding = %s::vector WHERE requirement_id = %s",
                    (emb, rid),
                )
            conn.commit()

    with conn.cursor(
        row_factory=psycopg.rows.dict_row,
    ) as cur:
        cur.execute(
            "SELECT test_case_id, title, description "
            "FROM test_cases WHERE embedding IS NULL"
        )
        tcs = cur.fetchall()
    if tcs:
        console.print(f"[cyan]Backfill эмбеддингов тест-кейсов: {len(tcs)}[/cyan]")
        for tc in tcs:
            tcid, title, desc = tc["test_case_id"], tc["title"], tc["description"]
            try:
                emb = embeddings.embed_text(f"{title} {desc}")
            except Exception as e:
                logger.warning(f"Failed to embed test case {tcid}: {e}")
                continue
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE test_cases SET embedding = %s::vector WHERE test_case_id = %s",
                    (emb, tcid),
                )
            conn.commit()

    console.print("[green]OK: Эмбеддинги актуальны.[/green]")


@click.command()
@click.option("--data-dir", type=click.Path(exists=True), default=None, help="Директория с данными")
@click.option("--skip-embeddings", is_flag=True, help="Пропустить генерацию эмбеддингов")
def migrate(data_dir: Optional[str], skip_embeddings: bool):
    """Загрузка данных в PostgreSQL."""
    run_migration(
        data_dir=Path(data_dir) if data_dir else None,
        skip_embeddings=skip_embeddings,
    )


if __name__ == "__main__":
    migrate()
