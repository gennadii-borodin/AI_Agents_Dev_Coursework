# QA Review Agent

AI-агентная система для автоматизированного ревью тест-кейсов на стеке LangGraph + pgvector.

## Возможности

- **Проверка покрытия требований** — построение матрицы Requirement → Test Case, расчёт weighted coverage
- **Оценка качества тест-дизайна** — анализ техник тест-дизайна, выявление слабых и дублирующих тестов
- **Контроль соответствия стандартам QA** — проверка правил оформления тест-кейсов
- **RAG-поиск** — векторный поиск релевантных документов через pgvector
- **SQL-инструмент** — безопасные запросы к базе данных

## Быстрый старт

### 1. Запуск инфраструктуры

```bash
docker compose up -d
```

### 2. Установка зависимостей

```bash
pip install -e ".[dev]"
```

### 3. Настройка переменных окружения

```bash
cp .env.example .env
# Отредактируйте .env, указав API-ключи
```

### 4. Миграция данных

```bash
python -m migrations.load_data
```

### 5. Запуск

```bash
# CLI режим
qa-review run "провести полное ревью"

# Интерактивный режим
qa-review run --interactive

# Просмотр эффективных настроек (без секретов)
qa-review --show-config
```

## Доступные сценарии

| Команда | Описание |
|---------|----------|
| `провести полное ревью` | Анализ всех требований и тестов |
| `проверить покрытие REQ-XXX` | Проверка покрытия конкретного требования |
| `оценить дизайн тестов` | Оценка качества тест-дизайна |
| `проверить соответствие стандартам` | Проверка стандартов QA |
| `найти тесты без требований` | Поиск тестов без привязки к требованиям |

## Архитектура

Пайплайн ревью оркестрируется графом LangGraph (`src/graph.py`):

```
┌─────────┐   ┌──────────────────────────┐   ┌────────────────────────────────────┐
│   CLI   │──▶│ Router (LLM, flash)       │──▶│ Агенты (Conditional Edges)         │
└─────────┘   │ prompts/router.yaml        │   │                                    │
              │ + regex-фоллбэк при сбое   │   │ ┌─────────┐ ┌────────┐ ┌──────────┐ │
              └──────────────────────────┘   │ │Coverage │ │ Design │ │Standards │ │
                                             │ └─────────┘ └────────┘ └──────────┘ │
                                             └────────────────────────────────────┘
                                                      │          │          │
                                                      ▼          ▼          ▼
                                             ┌──────────────────────────────────┐
                                             │ Tools & Skills (ToolRegistry)   │
                                             │ • sql_query  (skills/sql_query.yaml)
                                             │ • rag_search  (skills/rag_search.yaml)
                                             │ • LLM Provider (RouterAI)
                                             └──────────────────────────────────┘
```

- **Промпты** — единый источник истины в `prompts/*.yaml` (`coverage_agent`,
  `design_agent`, `standards_agent`, `router`, `orchestrator`). Загружаются через
  `src/prompts.py`; к промпту добавляются JSON-контракт вывода (из `output_schema`)
  и few-shot примеры (`few_shot`). Агенты и роутер НЕ содержат инлайн-промптов.
- **Скиллы** — `skills/*.yaml` описывают инструменты (схемы ввода/вывода). `src/skills.py`
  (`ToolRegistry`) строит по ним определения function-calling и исполняет вызовы.
- **Router** — классифицирует запрос LLM-моделью по `prompts/router.yaml`; при сбое
  LLM используется детерминированный regex-роутинг (фоллбэк).
- **Агрегаты** (покрытие, % соответствия стандартам, остаточный риск) вычисляются
  в коде по данным, возвращаемым LLM: модель классифицирует, а не считает.
- **Структурированный вывод** — агенты запрашивают `response_format` (JSON Schema,
  построенная из `output_schema`), с безопасным откатом к `json_mode`.

## Технологии

- **Python 3.12+**
- **LangGraph** — оркестрация агентов
- **LangChain** — интеграция с LLM
- **Pydantic v2** — типизация данных
- **PostgreSQL + pgvector** — векторная память для RAG
- **Arize Phoenix** — мониторинг и трассировка
- **RouterAI** (deepseek-v4-pro/flash) — LLM-провайдер

## Структура проекта

```
qa-review-agent/
├── data/              # синтетические данные (CSV, MD)
├── prompts/           # промпты агентов (YAML)
├── skills/            # skills (YAML)
├── src/
│   ├── agents/        # агенты (coverage, design, standards)
│   ├── tools/         # инструменты (SQL, RAG)
│   ├── config.py      # настройки
│   ├── models.py      # Pydantic-модели
│   ├── prompts.py     # загрузчик промптов + сборка JSON-схемы вывода
│   ├── skills.py      # ToolRegistry: скиллы как function-calling инструменты
│   ├── llm_provider.py # кастомный LLM-провайдер + invoke_with_tools
│   ├── embedding.py   # провайдер эмбеддингов
│   ├── graph.py       # LangGraph оркестрация + LLM-роутер
│   ├── cli.py         # CLI интерфейс
│   └── report.py      # генерация отчётов
├── tests/
│   ├── unit/          # юнит-тесты (offline, без LLM/БД)
│   └── integration/   # интеграционные тесты (нужны БД + LLM)
├── migrations/        # миграции БД
└── reports/           # выходные отчёты (Markdown)
```

## Запуск тестов

```bash
# Юнит-тесты (offline, без LLM и БД)
pytest tests/unit -v

# Интеграционные тесты (требуют поднятую БД и доступ к LLM)
pytest tests/integration -v
```

## Трассировка (Arize Phoenix)

Каждый запуск ревью автоматически пишет трейсы в Phoenix (OTLP/gRPC, порт `4317`).
Phoenix поднят в Docker Compose (`localhost:6006`).

**Что попадает в трейсы:**

- **Сессия на прогон** — уникальный `session.id` (`qa-review-xxxxxxxxxx`) на каждый вызов `run_review`. Все спаны одного прогона собраны в одну сессию/трейс.
- **Вложенность спанов:**

  ```
  QA Review Run (AGENT, session.id)
  ├─ Router                         [AGENT]
  ├─ Coverage Agent   [AGENT]
  │  ├─ LLM deepseek-v4-pro-0813    [LLM]   (модель, температура, токены, ввод/вывод)
  │  ├─ Retrieve test_cases         [RETRIEVER]  (top_k, найденные документы + similarity)
  │  │  └─ Embedding text-embedding-3-small  [EMBEDDING]  (модель, входной текст)
  │  └─ Tool execute_sql            [TOOL]  (SQL, row_count)
  ├─ Design Agent     [AGENT] → LLM + SQL
  └─ Standards Agent  [AGENT] → LLM + SQL
  ```

- **Полезные атрибуты:** `qa.scenario`, `qa.requirement_ids`, `qa.agents`, счётчики
  (`requirements.count`, `violations.total`), оценки (`qa.coverage_pct`, `qa.design_score`,
  `qa.standards_compliance_pct`), `duration_ms` на каждом спане, оценки токенов.

**Просмотр:** откройте http://localhost:6006 → список сессий, по одной на каждый запуск.

**Проверка доступности (вне приложения):**

```bash
python test_phoenix_connectivity.py
```

**Эмбеддинги:** генерируются при миграции (`migrations.load_data`). Если данные уже
загружены, но `embedding IS NULL`, миграция дозаполнит их автоматически. RAG-поиск
использует `1 - (embedding <=> query)` (pgvector 0.8).

## Лицензия

MIT

