"""Интеграционные тесты миграции БД и загрузки данных.

Проверяют:
- парсинг реальных CSV (requirements.csv, online_store_test_cases.csv);
- содержимое миграционного SQL (DDL, pgvector, индексы);
- end-to-end run_migration() с моком psycopg: корректность INSERT (число столбцов,
  кол-во строк, эмбеддинги) и идемпотентность повторного запуска;
- backfill эмбеддингов для строк с NULL.

Внешняя БД / расширение pgvector НЕ требуются - соединение полностью замокировано.
"""

import re
import tempfile
from pathlib import Path

import pytest

from migrations import load_data

pytestmark = [pytest.mark.integration]

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MIGRATION_SQL = PROJECT_ROOT / "migrations" / "001_initial.sql"

REQ_NULL_SQL = (
    "SELECT requirement_id, title, requirement_text "
    "FROM requirements WHERE embedding IS NULL"
)
TC_NULL_SQL = (
    "SELECT test_case_id, title, description "
    "FROM test_cases WHERE embedding IS NULL"
)


# --------------------------------------------------------------------------- #
# Моки psycopg / pgvector / EmbeddingProvider
# --------------------------------------------------------------------------- #
class FakeCursor:
    def __init__(self, conn: "FakeConn"):
        self.conn = conn
        self._fone = None
        self._fall: list = []

    def execute(self, sql: str, params=None):
        self.conn.executed.append((sql, params))
        if "SELECT COUNT(*) FROM requirements" in sql:
            self._fone = (self.conn.req_count,)
        elif REQ_NULL_SQL in sql:
            self._fall = list(self.conn.req_null_rows)
        elif TC_NULL_SQL in sql:
            self._fall = list(self.conn.tc_null_rows)
        else:
            self._fone = None

    def executemany(self, sql: str, batch):
        self.conn.executed.append((sql, list(batch)))

    def fetchone(self):
        return self._fone

    def fetchall(self):
        return self._fall

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, req_count=0, req_null_rows=None, tc_null_rows=None):
        self.req_count = req_count
        self.req_null_rows = req_null_rows or []
        self.tc_null_rows = tc_null_rows or []
        self.executed: list[tuple] = []
        self.committed = False
        self.migration_sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, params=None):
        # migration SQL исполняется напрямую на соединении - просто записываем
        self.migration_sql = sql

    def cursor(self, *args, **kwargs):
        return FakeCursor(self)

    def commit(self):
        self.committed = True


class FakePsycopg:
    def __init__(self):
        self.connections: list[FakeConn] = []
        self.initial_req_count = 0
        self.initial_req_null_rows: list = []
        self.initial_tc_null_rows: list = []
        # минимальный аналог psycopg.rows.dict_row для тестов
        self.rows = type("Rows", (), {"dict_row": object()})()

    def connect(self, url, *args, **kwargs):
        conn = FakeConn(
            req_count=self.initial_req_count,
            req_null_rows=self.initial_req_null_rows,
            tc_null_rows=self.initial_tc_null_rows,
        )
        self.connections.append(conn)
        return conn


class FakeEmbeddingProvider:
    def __init__(self, settings=None):
        self.calls = 0

    def embed_text(self, text, model=None):
        self.calls += 1
        return [0.0] * 1536


@pytest.fixture
def fake_db(monkeypatch):
    """Подменяет psycopg, register_vector и EmbeddingProvider на моки."""
    fake_psycopg = FakePsycopg()
    monkeypatch.setattr(load_data, "psycopg", fake_psycopg)
    monkeypatch.setattr(load_data, "register_vector", lambda conn: None)
    monkeypatch.setattr(load_data, "EmbeddingProvider", FakeEmbeddingProvider)
    return fake_psycopg


def _inserts(executed, table):
    return [(s, b) for s, b in executed if f"INSERT INTO {table}" in s]


# --------------------------------------------------------------------------- #
# 1. Парсинг CSV (чистая логика, без БД)
# --------------------------------------------------------------------------- #
def test_csv_parsing_requirements():
    rows = load_data._read_csv_clean(DATA_DIR / "requirements.csv")
    assert len(rows) == 20
    for row in rows:
        assert "requirement_id" in row and row["requirement_id"]
        assert "title" in row and "requirement_text" in row
        assert "category" in row and "priority" in row


def test_csv_parsing_test_cases():
    rows = load_data._read_csv_clean(DATA_DIR / "online_store_test_cases.csv")
    assert len(rows) == 125
    for row in rows:
        assert "test_case_id" in row and row["test_case_id"]
        assert "REQ" in row
        assert "steps" in row and "expected_result" in row


def test_csv_parsing_handles_quoted_commas():
    """В тест-кейсах шаги/описания содержат запятые внутри кавычей."""
    content = (
        '"test_case_id","REQ","title","description","preconditions",'
        '"test_data","steps","expected_result","priority","test_type",'
        '"design_quality","qa_review","review_comment"\n'
        '"TC-1","REQ-1","Login","Desc","Pre","Data",'
        '"1. Открыть, 2. Нажать","Ok","High","positive","good","pass",""\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", encoding="utf-8", delete=False) as f:
        f.write(content)
        path = Path(f.name)
    try:
        rows = load_data._read_csv_clean(path)
        assert len(rows) == 1
        assert rows[0]["steps"] == "1. Открыть, 2. Нажать"
        assert rows[0]["test_case_id"] == "TC-1"
    finally:
        path.unlink()


# --------------------------------------------------------------------------- #
# 2. Содержимое миграционного SQL
# --------------------------------------------------------------------------- #
def test_migration_sql_ddl():
    sql = MIGRATION_SQL.read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE IF NOT EXISTS requirements" in sql
    assert "CREATE TABLE IF NOT EXISTS test_cases" in sql
    # pgvector-колонки корректной размерности
    assert "embedding vector(1536)" in sql
    # индексы векторного поиска
    assert "ivfflat (embedding vector_cosine_ops)" in sql
    # уникальные ключи, предотвращающие дубли при повторном прогоне
    assert "requirement_id VARCHAR(20) UNIQUE NOT NULL" in sql
    assert "test_case_id VARCHAR(20) UNIQUE NOT NULL" in sql


def test_insert_statements_match_columns():
    """INSERT-ы в коде должны совпадать по числу столбцов с DDL-колонками."""
    sql = MIGRATION_SQL.read_text(encoding="utf-8")
    assert sql.count("embedding vector(1536)") == 2  # requirements + test_cases

    req_cols = [
        "requirement_id", "title", "requirement_text", "category", "priority",
        "qa_requirements_review", "rejection_reason", "embedding",
    ]
    tc_cols = [
        "test_case_id", "req", "title", "description", "preconditions",
        "test_data", "steps", "expected_result", "priority", "test_type",
        "design_quality", "qa_review", "review_comment", "embedding",
    ]
    src = Path(load_data.__file__).read_text(encoding="utf-8")
    req_insert = re.search(r"INSERT INTO requirements \(([^)]+)\) VALUES \(([^)]+)\)", src)
    tc_insert = re.search(r"INSERT INTO test_cases \(([^)]+)\) VALUES \(([^)]+)\)", src)
    assert req_insert, "requirements INSERT не найден"
    assert tc_insert, "test_cases INSERT не найден"
    assert len(req_insert.group(1).split(",")) == len(req_cols) == 8
    assert len(tc_insert.group(1).split(",")) == len(tc_cols) == 14


# --------------------------------------------------------------------------- #
# 3. End-to-end run_migration (мок psycopg)
# --------------------------------------------------------------------------- #
def test_run_migration_loads_all_rows_skip_embeddings(fake_db):
    load_data.run_migration(skip_embeddings=True)
    conn = fake_db.connections[0]

    assert conn.committed is True
    assert "CREATE EXTENSION" in conn.migration_sql

    req_inserts = _inserts(conn.executed, "requirements")
    tc_inserts = _inserts(conn.executed, "test_cases")

    req_rows = sum(len(b) for _, b in req_inserts)
    tc_rows = sum(len(b) for _, b in tc_inserts)
    assert req_rows == 20, f"requirements: {req_rows}"
    assert tc_rows == 125, f"test_cases: {tc_rows}"

    # каждая запись требования - ровно 8 полей, эмбеддинг=None
    for _, batch in req_inserts:
        for row in batch:
            assert len(row) == 8
            assert row[7] is None  # embedding


def test_run_migration_passes_embeddings(fake_db):
    load_data.run_migration(skip_embeddings=False)
    conn = fake_db.connections[0]

    req_inserts = _inserts(conn.executed, "requirements")
    assert req_inserts, "нет INSERT требований"
    for _, batch in req_inserts:
        for row in batch:
            emb = row[7]
            assert isinstance(emb, list) and len(emb) == 1536


def test_run_migration_idempotent_on_rerun(fake_db):
    """Повторный запуск при уже загруженных данных не делает INSERT-ов."""
    fake_db.initial_req_count = 20  # уже загружено
    load_data.run_migration(skip_embeddings=True)
    conn = fake_db.connections[0]
    inserts = [s for s, _ in conn.executed if "INSERT INTO" in s]
    assert inserts == [], "повторный прогон не должен перезаливать данные"


def test_run_migration_backfill_when_rerun_with_embeddings(fake_db):
    fake_db.initial_req_count = 20
    fake_db.initial_req_null_rows = [
        {"requirement_id": "REQ-1", "title": "T", "requirement_text": "Text"},
        {"requirement_id": "REQ-2", "title": "T2", "requirement_text": "Text2"},
    ]
    fake_db.initial_tc_null_rows = [
        {"test_case_id": "TC-1", "title": "T", "description": "D"},
    ]
    load_data.run_migration(skip_embeddings=False)
    conn = fake_db.connections[0]
    # backfill должен сделать UPDATE для каждой NULL-строки
    updates = [
        s for s, _ in conn.executed
        if "UPDATE requirements SET embedding" in s
        or "UPDATE test_cases SET embedding" in s
    ]
    assert len(updates) == 3


# --------------------------------------------------------------------------- #
# 4. Backfill эмбеддингов (прямой вызов)
# --------------------------------------------------------------------------- #
def test_backfill_embeddings_updates_null_rows(fake_db):
    fake_db.initial_req_null_rows = [
        {"requirement_id": "REQ-1", "title": "Title", "requirement_text": "Body"},
    ]
    fake_db.initial_tc_null_rows = []
    conn = fake_db.connect("dummy")
    provider = FakeEmbeddingProvider()
    load_data._backfill_embeddings(conn, provider)

    updates = [
        (s, p) for s, p in conn.executed
        if "UPDATE requirements SET embedding" in s
    ]
    assert len(updates) == 1
    # параметры UPDATE: (embedding_vector, requirement_id)
    _, params = updates[0]
    assert params[1] == "REQ-1"
    assert len(params[0]) == 1536
