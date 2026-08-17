# QA Review Agent

AI-агентная система для автоматизированного ревью тест-кейсов на стеке LangGraph + pgvector.

## Возможности

- **Проверка покрытия требований** — построение матрицы Requirement → Test Case, расчёт weighted coverage
- **Оценка качества тест-дизайна** — анализ техник тест-дизайна, выявление слабых и дублирующих тестов
- **Контроль соответствия стандартам QA** — проверка правил оформления тест-кейсов
- **RAG-поиск** — векторный поиск релевантных документов через pgvector
- **SQL-инструмент** — безопасные запросы к базе данных
- **Статический валидатор** — детерминированная проверка структуры тест-кейсов

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
qa-review "провести полное ревью"

# Интерактивный режим
qa-review --interactive

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
| `проверить покрытие` | Проверка покрытия всех требований |

## Архитектура

Пайплайн ревью оркестрируется графом LangGraph (`src/graph.py`):

```
┌─────────┐   ┌──────────────────────────┐   ┌────────────────────────────────────┐
│   CLI   │──▶│ Router (LLM, flash)       │──▶│ load_data_once → fan-out (Send)    │
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
                                             │ • code_validator (skills/code_validator.yaml)
                                             │ • LLM Provider (RouterAI)
                                             └──────────────────────────────────┘
                                                      │
                                                      ▼
                                             ┌──────────────────────────────────┐
                                             │ quality_gate + targeted retry   │
                                             └──────────────────────────────────┘
```

- **Промпты** — единый источник истины в `prompts/*.yaml` (`coverage_agent`,
  `design_agent`, `standards_agent`, `router`). Загружаются через
  `src/prompts.py`; к промпту добавляются JSON-контракт вывода (из `output_schema`)
  и few-shot примеры (`few_shot`). Основные system-promptы агентов и роутера
  вынесены в YAML; вспомогательные inline-промпты (например, для RAG) остаются в коде.
- **Скиллы** — `skills/*.yaml` описывают инструменты (схемы ввода/вывода). `src/skills.py`
  (`ToolRegistry`) строит по ним определения function-calling и исполняет вызовы.
  Реестр навыков заполняется декоратором `@register_skill` из `src/skills.py`.
- **Router** — классифицирует запрос LLM-моделью по `prompts/router.yaml`; при сбое
  LLM используется детерминированный regex-роутинг (фоллбэк). Переключатель
  `router_llm_enabled` позволяет отключить LLM-роутинг полностью.
- **Оркестрация** — граф LangGraph: `router → load_data_once → fan-out агентов →
  finalize → quality_gate`. `quality_gate` выполняет targeted retry: при частичном
  сбое перезапускает только упавших агентов через `Command(goto=Send(...))`.
- **Агрегаты** (покрытие, % соответствия стандартам, остаточный риск) вычисляются
  в коде по данным, возвращаемым LLM: модель классифицирует, а не считает.
- **Структурированный вывод** — агенты запрашивают `response_format` (JSON Schema,
  построенная из `output_schema`), с безопасным откатом к `json_object`/`json_mode`.
- **Checkpointer** — по умолчанию `MemorySaver` (в памяти процесса); можно передать
  свой checkpointer (например, `PostgresSaver`) для resume между перезапусками.

## Технологии

- **Python 3.12+**
- **LangGraph** — оркестрация агентов
- **Pydantic v2** — типизация данных
- **PostgreSQL + pgvector** — векторная память для RAG
- **Arize Phoenix** — мониторинг и трассировка
- **RouterAI** (deepseek-v4-pro/flash) — LLM-провайдер

## Структура проекта

```
qa-review-agent/
├── data/              # синтетические данные (CSV, YAML)
├── prompts/           # промпты агентов (YAML)
├── skills/            # skills (YAML) + манифест доступа агентов
├── src/
│   ├── agents/        # агенты (coverage, design, standards)
│   ├── tools/         # инструменты (SQL, RAG, code_validator)
│   ├── config.py      # настройки
│   ├── models.py      # Pydantic-модели
│   ├── prompts.py     # загрузчик промптов + сборка JSON-схемы вывода
│   ├── skills.py      # ToolRegistry: скиллы как function-calling инструменты
│   ├── tool_resilience.py # RateLimiter / CircuitBreaker / run_with_timeout
│   ├── llm_provider.py # кастомный LLM-провайдер + invoke_with_tools
│   ├── embedding.py   # провайдер эмбеддингов
│   ├── graph.py       # LangGraph оркестрация + LLM-роутер
│   ├── cli.py         # CLI интерфейс
│   ├── report.py      # генерация отчётов
│   └── tracing.py     # OpenTelemetry / Phoenix tracing
├── tests/
│   ├── unit/          # юнит-тесты (offline, без LLM/БД)
│   └── integration/   # интеграционные тесты (нужны БД + LLM)
├── eval/              # регрессионная оценка на golden-датасете
├── migrations/        # миграции БД
└── reports/           # выходные отчёты (Markdown)
```

## Конфигурация

Основные параметры задаются через `.env` (см. `.env.example`) и `src/config.py`:

- `router_llm_enabled` — использовать LLM-роутер (`True`) или детерминированный regex (`False`).
- `rag_enabled` — включить RAG-поиск в Coverage Agent (`True`/`False`).
- `targeted_retry_enabled` / `max_retry_attempts` — targeted retry упавших агентов в `quality_gate`.
- `standards_max_iterations` / `standards_max_tokens` — защита от runaway-циклов в Standards Agent.
- `sql_max_rows` / `sql_statement_timeout` — лимиты SQL-выборок.

Просмотр эффективных настроек (без секретов):

```bash
qa-review --show-config
```

## Запуск тестов

```bash
# Юнит-тесты (offline, без LLM и БД)
pytest tests/unit -v

# Интеграционные тесты (требуют поднятую БД и доступ к LLM)
pytest tests/integration -v

# Регрессионная оценка на golden-датасете
python -m eval.run_eval
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

