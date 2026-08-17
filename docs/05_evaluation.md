# Оценка и безопасность (Требование 5/5)

**Цель документа:** описать метрики (успех, время, стоимость), логирование/трассировку, ограничения (retry / лимит / circuit breaker) и проверку вывода / защиту от некорректных действий.

---

## 1. Метрики

### 1.1 Успех
Фиксируется в корневом спане прогона (`src/report.py:347-396`):
- `qa.run_status` ∈ `success` / `needs_clarification` (провал роутинга) / `partial` (часть чанков Standards не проанализирована) / `error`.
- `qa.error_type` — имя исключения при сбое графа.
- Наличие отчётов: `qa.coverage_pct`, `qa.design_score`, `qa.standards_compliance_pct`, `qa.violations_count`.

Регрессионная оценка — `eval/run_eval.py`: golden-датасет прогоняется через граф, сверяются `scenario`/`agents_to_run`/`errors`/наличие отчётов (`assert_quality`, `eval/run_eval.py:70`). Итог — `eval/results/eval_latest.json` с `summary` (total/errors/reports_missing/latency min-max-mean).

### 1.2 Время (латентность)
- Per-run latency фиксируется в `trace_run` (`src/tracing.py:219`) через `duration_ms` на каждом спане.
- Eval harness замеряет `latency_s` каждого кейса (`eval/run_eval.py:29-31`) и агрегирует min/max/mean.

### 1.3 Стоимость (token $)
В `run_review` (`src/report.py:378-386`) на основе `get_run_stats()` (накопленные токены по моделям) и `settings.model_pricing`:

```python
est_cost += toks["prompt"]/1e6*price["input"] + toks["completion"]/1e6*price["output"]
run_span.set_attribute("qa.estimated_cost_usd", round(est_cost, 4))
```

Метрики токенов (`qa.llm_calls`, `qa.prompt_tokens`, `qa.completion_tokens`, `qa.total_tokens`) записываются **всегда** — блок защищён `finally`, поэтому $/токены не теряются даже при исключении (`src/report.py:338-396`).

---

## 2. Логирование и трассировка

### 2.1 Phoenix / OpenTelemetry
- Инициализация: `init_phoenix` (`src/tracing.py:105`) — OTLP/gRPC экспортер в Phoenix, авто-обнаружение endpoint (`host.docker.internal`/`localhost`/`127.0.0.1`), graceful-degrade если Phoenix недоступен.
- Каждый прогон — отдельная **сессия** (`new_session_id`, `src/tracing.py:171`), все спаны одного запуска попадают в один трейс.
- Контекст-менеджеры спанов: `trace_run` (AGENT run), `trace_agent`, `trace_llm`, `trace_tool`, `trace_embedding`, `trace_retriever` (`src/tracing.py`).

### 2.2 Семантика OpenInference
Спаны несут стандартные атрибуты: `LLM_MODEL_NAME`, `LLM_INVOCATION_PARAMETERS`, `LLM_INPUT_MESSAGES`, `LLM_TOKEN_COUNT_*`, `RETRIEVAL_DOCUMENTS`, `TOOL_NAME`, `TOOL_PARAMETERS`, `OUTPUT_VALUE`. Это даёт готовые к воспроизведению трейсы вызовов LLM/инструментов в UI Phoenix (`http://localhost:6006`).

### 2.3 Application logging
`logging` на уровне INFO в узлах/агентах (`logger.info("Running Coverage Agent")` и т.п.) + `logger.exception` при сбоях.

---

## 3. Ограничения (retry / лимит / circuit breaker)

| Ограничение | Механизм | Где |
|---|---|---|
| Retry LLM-вызовов | `@retry(stop_after_attempt(llm_retry_attempts=3), wait_exponential)` на `chat_completion`/`_do_request` | `src/llm_provider.py:71`, `:139` |
| Retry эмбеддингов | `@retry(stop_after_attempt(3), wait_exponential)` на `_embed_uncached` | `src/embedding.py:22` |
| Retry агентов (targeted) | `quality_gate`: повтор ONLY упавших агентов через `Command(goto=Send)` | `src/graph.py:329-336` |
| Лимит retry-циклов | `max_retry_attempts=2` + `state.iteration` (защита от петли) | `src/config.py:66`, `src/graph.py:325` |
| Лимит чанков Standards | `standards_max_iterations=200` (hard cap runaway-цикла) | `src/agents/standards_agent.py:180` |
| Лимит строк SQL | `sql_max_rows=1000` (truncate + флаг) | `src/tools/sql_tool.py:87` |
| Лимит tool-итераций | `max_iterations` в `invoke_with_tools` | `src/llm_provider.py:176` |
| Circuit-breaker (по сути) | при исчерпании retries LLM — агент не падает графом, а помечается `partial`/`error` и исключается из знаменателя compliance | `src/agents/standards_agent.py:198-203` |

События retry видны в трейсе (`_on_retry` добавляет `llm_retry` event, `src/llm_provider.py:45`).

---

## 4. Проверка вывода и защита от некорректных действий

### 4.1 Валидация/ремонт JSON от LLM
- **Structured output:** запрос через `response_format={"type":"json_schema", ...}` (`build_json_schema`, `src/prompts.py:55`); фоллбэк на `json_object` при неподдерживаемой схеме (`src/llm_provider.py:112-118`).
- **Ремонт:** `json_repair.repair_json` при `JSONDecodeError` (coverage/design/standards агенты); `json_repaired` event в спане.
- **Нормализация:** `_normalize_design`, `_normalize_violations`, `_fix_json` защищают от вложенных списков, отсутствующих ключей, не-dict элементов (adversarial-сбой).

### 4.2 Детерминированный пересчёт (защита от «арифметики LLM»)
- `_recompute_coverage` (`src/agents/coverage_agent.py:36`) пересчитывает `total_coverage`/`critical_coverage`/`residual_risk` в коде по матрице весов из `priority_weights` — модель только классифицирует покрытие.
- `StandardsReport.compliance_percentage` считается в коде: `passed_checks / (analyzed_count * num_rules)` (`src/agents/standards_agent.py:205-216`), знаменатель — число активных правил из YAML (не магическая константа).

### 4.3 Блокировка некорректных действий (SQL/данные)
- **Forbidden keywords:** `execute_sql` отсекает `INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/TRUNCATE` до выполнения (`skills/sql_query.yaml`, `src/skills.py:_check_sql`).
- **Только SELECT:** проекция колонок без вектора исключает утечку тяжёлых payload.
- **Валидация аргументов tool:** `ToolRegistry._validate` (`src/skills.py:279`) — обязательные параметры, приведение `int`.

### 4.4 Безопасность вывода / PII (маскировка)
`mask_sensitive` (`src/tracing.py:23`) вырезает секреты (`api_key`, `token`, `password`...), email и номера карт **перед записью в спаны**. Standards Agent проверяет «secrets в test_data» (QA-TEST), поэтому в БД могут быть секреты — они маскируются до попадания в Phoenix.

### 4.5 Бизнес-валидация (quality gate + правила)
- `quality_gate` не пропускает частично упавший прогон молча — фиксирует `unresolved_questions` (`partial review: missing reports`).
- Реестр `data/standards_rules.yaml` размечает `blocking`/`auto_fixable` правила; `blocking_violations` выделяются отдельно (`src/agents/standards_agent.py:226-230`), что даёт чёткий критерий «не релизить».

---

## 5. Резюме по требованию 5/5

| Пункт | Статус | Реализация |
|---|---|---|
| Метрики (успех/время/стоимость) | ✅ | `qa.run_status`, latency, `qa.estimated_cost_usd` |
| Логирование/трассировка | ✅ | Phoenix OTEL + OpenInference semconv + logging |
| Retry/лимит/circuit breaker (≥1) | ✅ | tenacity retry, max_retry_attempts, standards_max_iterations, sql_max_rows |
| Проверка вывода / защита | ✅ | json_repair, recompute, forbidden SQL, mask_sensitive, quality_gate |

---

## Сквозная схема безопасности и метрик

```mermaid
flowchart TD
    IN[Запрос пользователя] --> TR[trace_run: session_id, mask_sensitive]
    TR --> AGL[LLM/embedding: @retry x3]
    AGL --> TO[Tool: SQL forbidden-kw block, rag [] on fail]
    TO --> OUT[LLM output: json_schema + json_repair]
    OUT --> CALC[Детерминированный пересчёт метрик]
    CALC --> QG[quality_gate: retry/limit]
    QG --> MET[Метрики: status, latency, tokens, $]
    MET --> SPAN[Phoenix spans + mask_sensitive]
    SPAN --> REP[Markdown отчёты + final_answer]
```
