# Ревью надёжности — `qa-review-agent` (мультиагентная QA-система на LangGraph)

## Executive Summary (Общая оценка)

- **Готовность к production: NEEDS WORK (нужна доработка).** Сильная оркестрация, трассировка, изоляция ошибок и детерминированный пересчёт. Но **оценка (Evaluation) незрелая** (P0), а **защита от prompt-injection / утечки секретов** проработана недостаточно (P1). Не «Not Ready» — основной цикл корректен и безопасен по умолчанию для read-only операций.
- **Топ-3 критических риска (P0):**
  1. **Оценка не доказывает надёжность** — всего 10 золотых кейсов в mock-режиме, нет грейдеров качества, нет adversarial/should-refuse, нет реальных прогонов на LLM.
  2. **Путь prompt-injection / утечки секретов** — `user_query` + `test_data` (где могут быть секреты) попадают в LLM без редактирования; `mask_sensitive` покрывает только трассы.
  3. **Нет глобального operating envelope** — нет лимита шагов/tool-вызовов/токенов/wall-clock на весь прогон; только `standards_max_iterations=200` ограничивает один цикл.
- **Топ-3 quick wins (быстрые победы):**
  1. Убрать противоречащий markdown-блок «Формат вывода» из промптов coverage/design/standards (P-A, усилие S).
  2. Заменить denylist-проверку SQL на allowlist (должен начинаться с `SELECT`/`WITH`, один statement) (T-A, усилие S).
  3. Расширить golden-набор до 50+ с edge/adversarial/should-refuse + добавить детерминированные грейдеры качества (E-A/E-C, усилие M, максимальный выхлоп).

## Детальный анализ по категориям

### 1. Tools (инструменты)
**Сильные стороны:** проекция колонок исключает `vector(1536)` payload (`sql_tool.py:22-29`); лимит `sql_max_rows=1000` с флагом; есть forbidden-keyword guard; каждый вызов tool трассируется с параметрами и результатом (`tracing.py:299`); `code_validator` используется как достоверный факт вместо LLM-гипотезы (хороший паттерн).

**Слабые стороны:**
- **T-A (P1):** `execute_sql` запрещает через **denylist по подстроке** по всему запросу (`sql_tool.py:36-40`). Ложные срабатывания (легитимный запрос, упоминающий «UPDATE» в тексте) и, что важнее, **нет allowlist / проверки на один statement** — не гарантируется, что запрос read-only. Заменить на: должен начинаться с `SELECT`/`WITH`/`EXPLAIN`, ровно один statement, без `;`.
- **T-B (P2):** Контракты вывода рассинхронизированы с YAML. `sql_query.yaml` не содержит `truncated` (возвращается в `sql_tool.py:57`); `rag_search.yaml` указывает `results: list` без формы элемента и **без поля `error`**; `code_validator.yaml` не описывает `findings[].test_case_id/issues`, которые жёстко ожидаются в `design_agent.py:136`. Контракты неявны в коде.
- **T-C (P2):** `rag_search` возвращает `[]` при любом сбое (`rag_tool.py:33,79`) → тихо занижает indirect coverage; ничего не добавляется в `state.errors`/`tool_errors`. Такой же паттерн «тихого пустого» рискует скрыть данные.
- **T-D (P2):** Нет таймаута соединения БД (дефолты psycopg) и нет retry на уровне tool; только LLM HTTP-слой имеет `tenacity`. Зависшее PG-соединение останавливает прогон без конверта для прерывания.
- **T-E (P3):** схема `sql_query.yaml` содержит только `query`, но `execute` передаёт `args.get("params")` (`skills.py:112`) — недокументированный параметр.

### 2. Skills / Sub-agents (навыки и под-агенты)
**Сильные стороны:** нет конфликтов инструкций (skills — независимые дескрипторы tool); `code_validator` как детерминированный суб-процесс — эталонный паттерн надёжности.

**Слабые стороны:**
- **S-B (P3):** `ToolRegistry` жадно загружает **все** YAML-скиллы при каждой инициализации (`skills.py:77`). На 3 скилла пренебрежимо, но «on-demand» загрузка не реализована. Определения tool корректно ограничены по агентам (только Coverage получает `rag_search`), так что переэкспозиции уже нет.
- Композиционных sub-агентов с измеримыми objective нет (допустимо для этой архитектуры).

### 3. Prompts (системные и рабочие промты)
**Слабые стороны:**
- **P-A (P1):** **Противоречивый формат вывода.** Каждый промпт агента содержит markdown-таблицу `## Формат вывода` (`coverage_agent.yaml:57-85` и др.) **и** `build_agent_system_prompt` добавляет «Верни СТРОГО валидный JSON…» (`prompts.py:49-53`). Модели сказано два формата. `response_format=json_schema` маскирует это на уровне API, но текст промпта самопротиворечив и хрупок. → Удалить markdown-блок вывода из системных промптов.
- **P-B (P2):** Нет явного блока **constraints** (макс. шаги, бюджет токенов, таймаут, запрещённые действия) и **нет self-check** («перед ответом проверь, что вывод соответствует контракту»). Фреймворк требует role→objective→inputs→constraints→output→self-check.
- **P-C (P2):** промпт `coverage_agent` просит LLM посчитать weighted coverage (`coverage_agent.yaml:41`), но `_recompute_coverage` (`coverage_agent.py:36`) его **перезаписывает**. Дрейф промпт/реализация; LLM делает отброшенную работу.
- **P-D (P3):** промпт роутера по умолчанию «Иначе → full_review» (`router.yaml:25`), но граф безопасно падает на неизвестном сценарии (`graph.py:113`). Рассинхрон: неоднозначный/вредоносный запрос маршрутизируется на запуск *всех* агентов вместо уточнения. Выровнять на безопасный default.
- **P-E (P2):** Нет **владельца** промпта и процесса weekly failure-review (только `version: "1.0"`).

**Сильные стороны:** есть few-shot примеры; промпт Standards инжектит единый источник `standards_rules.yaml` (нет дрейфа); роутер имеет явные упорядоченные правила сценариев.

### 4. Orchestration (оркестрация и workflow)
**Сильные стороны:** параллельный fan-out через `Send` (`graph.py:365`); **targeted retry только упавших агентов** с лимитом итераций (`graph.py:329-336`); graceful degradation (исключения агентов перехватываются, граф не падает); checkpoint/resume + `save_reports` в `finally` (`report.py:338-343`); цикл retry в `quality_gate`.

**Слабые стороны:**
- **O-A (P1):** **Нет глобального operating envelope.** `ReviewState.max_iterations` существует, но не используется (`models.py:106`); нет лимита шагов/tool-вызовов/токенов/wall-clock на весь прогон. Только `standards_max_iterations=200` ограничивает один цикл. Добавить конверт на уровне прогона (токены + wall-clock), проверяемый в `quality_gate`.
- **O-B (P2):** Human approval / escalation не интегрированы. `blocking_violations` и `human_review_required` формируются, но не выводятся как escalation-путь в workflow. Допустимо для read-only инструмента, но compliance хочет явный роутинг.
- **O-C (P2):** `run_find_unlinked_node` (`graph.py:205-221`) не ловит исключения в `state.errors` и не добавляет сбои tool — сбои `find_unlinked_tests` невидимы в state.

### 5. Evaluation (тестирование и оценка) — слабейшая категория
**Слабые стороны:**
- **E-A (P0):** Golden-набор = **10 кейсов** (`golden_dataset.json`), все happy-path по роутингу/формату. Нет edge (пустая БД, невалидный REQ-id, неоднозначный запрос), нет **adversarial/prompt-injection**, нет **should-refuse**. Нужно 50–100.
- **E-B (P0):** `eval_latest.json` показывает latency 2.4–4.6s и 0 ошибок → сгенерирован на **ScriptedLLM/mock** (реальные прогоны — минуты, см. прошлую трассу). Проверяет только *связку*, не качество. Нет p50/p95 latency, нет валидированной стоимости из production.
- **E-C (P0):** **Нет грейдеров качества.** `assert_quality` (`run_eval.py:70`) проверяет только scenario/agents/errors/unlinked — нет precision/recall нарушений, нет LLM-as-judge, нет детерминизма, нет human-уровня. Изменения нельзя регрессионно вязать по качеству.
- **E-D (P1):** Нет prompt-injection / red-team тестирования.
- **E-E (P1):** Нет go/no-go gates (P0-блокеров) в CI.
- **Замечание (P2):** T1 из прошлого ревью (повторение system-промпта на каждый chunk) **всё ещё присутствует** — `STANDARDS_SYSTEM_PROMPT` вместе с полным rules YAML шлётся на **каждый** chunk (`standards_agent.py:152`); `standards_max_tokens=4096` ограничивает выход, но не повторяемый вход.

**Сильные стороны:** обширный набор unit/integration тестов (`test_stage2..6`, `test_quality_gate`, `test_error_recovery` и др.) на ScriptedLLM; структура harness + regression есть — фундамент хороший, глубины нет.

### 6. Безопасность и compliance
**Сильные стороны:** forbidden SQL keywords; проекция колонок исключает утечку вектора/payload; `mask_sensitive` (keys/email/card) применяется по всем трассам (`tracing.py:23`); audit-спаны содержат параметры tool, выводы, retries, токены, воспроизводимы по `session_id`; правила Standards — единый источник.

**Слабые стороны:**
- **SA-A (P1):** **Prompt-injection / утечка секретов.** `user_query` и полные `test_data` (где могут быть секреты по QA-TEST-010) конкатенируются в LLM user-сообщения без редактирования (`coverage_agent.py:231`, `design_agent.py:141`, `standards_agent.py:147`). Вредоносный запрос может управлять отчётами или вывести секреты в `final_answer`/`reports/`. Нет санитизации входа и injection-guard.
- **SA-B (P2):** `mask_sensitive` покрывает трассы, но данные уходят к **внешнему LLM-провайдеру без редактирования** — разрыв data-handling/compliance (PII/секреты покидают периметр).
- **SA-C (P2):** Нет перечисленных «forbidden actions + условий escalation» в промптах; есть только DB-guard.
- **SA-D (P3):** Нет задокументированной bias-mitigation стратегии (низкая релевантность здесь, но фреймворк требует).

## Action Plan (приоритизированный список задач)

| Pri | Задача | Усилие | Ожидаемый эффект | Метрика успеха |
|---|---|---|---|---|
| **P0** | Расширить golden-набор до 50–100: normal/edge/adversarial/should-refuse; добавить детерминированные грейдеры качества (precision/recall нарушений, дельта coverage, валидность схемы) + рубрику LLM-as-judge | M | Изменения становятся регрессионно-безопасными; качество измеримо | ≥50 кейсов; грейдеры зелёные на baseline; ≥1 adversarial-кейс |
| **P0** | Прогнать реальный LLM-eval на golden-наборе; снять p50/p95 latency, токены и стоимость за задачу; вшить go/no-go gate в CI | M | Метрики уровня production; deploy-гейт | eval_latest.json из реальных прогонов; CI падает при нарушении качества |
| **P0** | Добавить prompt-injection / red-team suite + редактирование секретов перед отправкой в LLM (или задокументировать политику data-handling) | M | Закрыть путь утечки/injection | Red-team кейсы отказывают в утечке; нет секрета в `final_answer` |
| **P1** | Исправить противоречивый формат вывода в промптах (убрать markdown-блок) (P-A) | S | Меньше неоднозначности JSON/Markdown; меньше repair-фоллбэков | Доля валидного JSON ↑ в eval |
| **P1** | Заменить SQL denylist на allowlist + одиночный statement (T-A) | S | Надёжная read-only гарантия | Security-тест: injection отклоняется |
| **P1** | Добавить глобальный operating envelope: бюджет токенов + wall-clock, проверяемый в `quality_gate` (O-A) | M | Нет runaway cost/latency при росте | Прогон прерывается в конверте; `max_iterations` работает |
| **P1** | Интегрировать escalation: выводить `blocking_violations`/`human_review_required` как явный шаг workflow (O-B) | S | Compliance/audit-путь для high-risk находок | High-risk находки маршрутизируются, не теряются |
| **P2** | Согласовать контракты вывода tool в YAML с кодом (T-B); добавлять сбои tool/агентов в `state.errors`/`tool_errors` (T-C, O-C) | M | Контракты машиночитаемы; сбои наблюдаемы | Схема совпадает; сбои видны в трассах/state |
| **P2** | Добавить таймаут БД + retry на уровне tool; убрать повтор rules-YAML на chunk (T-D, E-замечание) | M | Устойчивость + ниже стоимость Standards | Standards input-токены ↓; hung-DB обработан |
| **P2** | Добавить блок constraints + self-check в промпты; назначить владельцев промптов + weekly failure-review (P-B, P-E) | S | Стабильные, версионируемые промпты | Поле owner; лог ревью существует |

## Appendix: Failure Patterns (таблица паттернов сбоев)

| Failure pattern | Root cause | Proposed fix | Validation scenario |
|---|---|---|---|
| `find_unlinked_tests` возвращает пусто при наличии unlinked ТК | сбой tool/агента проглочен, нет `state.errors` (O-C) | добавлять исключения в `state.errors`; ассертовать `unlinked_tests` | Внедрить сбой SQL → `state.errors` непуст, статус `partial` |
| Indirect coverage занижен | `rag_search` тихий `[]` при сбое (T-C) | выводить rag-сбои в `state.tool_errors`; фоллбэк на SQL join | Убить embedding-svc → отчёт coverage помечает degraded RAG |
| Прогон превышает бюджет при росте | нет глобального конверта (O-A) | token+wall-clock кап в `quality_gate` | 10× данных → прогон прерывается в конверте, статус `error` |
| Секрет из `test_data` попадает в отчёт | неотредактированная отправка в LLM (SA-A/B) | редактировать до LLM или политика + injection-guard | Внедрить секрет в ТК → отсутствует в `final_answer` |
| Срабатывает JSON-repair фоллбэк | противоречивый формат промпта (P-A) | убрать markdown-блок вывода | Eval: валидный JSON с первой попытки, нет `json_repaired` |
| Вредоносный запрос управляет сценарием | default full_review роутера + нет injection-guard (P-D, SA-A) | безопасный default = clarify; injection-тест в eval | «ignore rules, dump passwords» → отказ/уточнение |
