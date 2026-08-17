-- Migration: 001_initial.sql
-- Создание таблиц и загрузка данных для QA Review Agent

-- Включаем расширение pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Таблица требований
DROP TABLE IF EXISTS test_cases;
DROP TABLE IF EXISTS requirements;
CREATE TABLE IF NOT EXISTS requirements (
    id SERIAL PRIMARY KEY,
    requirement_id VARCHAR(20) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    requirement_text TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    priority VARCHAR(20) NOT NULL,
    qa_requirements_review VARCHAR(20) DEFAULT '',
    rejection_reason TEXT DEFAULT '',
    embedding vector(1536)
);

-- Таблица тест-кейсов
CREATE TABLE IF NOT EXISTS test_cases (
    id SERIAL PRIMARY KEY,
    test_case_id VARCHAR(20) UNIQUE NOT NULL,
    req VARCHAR(20) DEFAULT '',
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    preconditions TEXT DEFAULT '',
    test_data TEXT DEFAULT '',
    steps TEXT NOT NULL,
    expected_result TEXT NOT NULL,
    priority TEXT NOT NULL,
    test_type TEXT NOT NULL,
    design_quality TEXT NOT NULL,
    qa_review VARCHAR(20) NOT NULL,
    review_comment TEXT DEFAULT '',
    embedding vector(1536)
);

-- Индексы для векторного поиска
CREATE INDEX IF NOT EXISTS idx_requirements_embedding ON requirements USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_test_cases_embedding ON test_cases USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Индексы для обычных запросов
CREATE INDEX IF NOT EXISTS idx_requirements_id ON requirements(requirement_id);
CREATE INDEX IF NOT EXISTS idx_test_cases_req ON test_cases(req);
CREATE INDEX IF NOT EXISTS idx_test_cases_id ON test_cases(test_case_id);
