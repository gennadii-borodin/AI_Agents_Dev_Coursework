import json
import logging
from typing import Optional

from src.config import Settings, get_settings
from src.json_utils import parse_json_response
from src.llm_provider import RouterAIProvider
from src.models import CoverageReport, Requirement, TestCase
from src.prompts import build_agent_system_prompt, build_json_schema
from src.tools.sql_tool import (
    get_all_requirements,
    get_all_test_cases,
    get_requirements_by_ids,
    get_test_cases_by_reqs,
    get_tests_without_requirements,
)

logger = logging.getLogger(__name__)


def _recompute_coverage(data: dict, settings: Optional[Settings] = None) -> dict:
    """Пересчитывает агрегаты покрытия в коде по матрице требований.

    LLM возвращает матрицу с флагом covered/weight/priority по каждому
    требованию; итоговые проценты и остаточный риск вычисляются детерминированно,
    без полагания на арифметику модели.     Вес берётся из поля ``weight`` матрицы (если это положительное число),
    а при его отсутствии/нулевом значении — из ``settings.priority_weights`` по
    приоритету (сравнение регистронезависимое). Если приоритет не распознан,
    требование получает вес 1.0, чтобы невалидный/пустой приоритет от LLM не
    обнулял знаменатель и не превращал реальное покрытие в 0%.
    Пороги остаточного риска — из настроек.
    """
    if settings is None:
        settings = get_settings()
    weights = settings.priority_weights

    if not isinstance(data, dict):
        # LLM вернул не объект (напр. JSON-массив) — бросаем, чтобы узел
        # поймал ошибку и quality_gate повторил агента, вместо краша графа.
        raise ValueError(
            f"Coverage LLM returned {type(data).__name__}, expected object"
        )

    matrix = data.get("matrix") or []
    if not matrix:
        return data

    def _w(m: dict) -> float:
        try:
            w = float(m.get("weight") or 0.0)
        except (TypeError, ValueError):
            w = 0.0
        if w > 0:
            return w
        prio = str(m.get("priority") or "").strip().lower()
        for k, v in weights.items():
            if k.lower() == prio:
                return float(v)
        return 1.0

    total_w = sum(_w(m) for m in matrix)
    covered_w = sum(_w(m) for m in matrix if m.get("covered"))
    total_coverage = round(covered_w / total_w * 100, 1) if total_w else 0.0

    critical = [m for m in matrix if str(m.get("priority") or "").lower() == "critical"]
    crit_total = sum(_w(m) for m in critical)
    crit_covered = sum(_w(m) for m in critical if m.get("covered"))
    critical_coverage = round(crit_covered / crit_total * 100, 1) if crit_total else 0.0

    if total_coverage < settings.coverage_risk_high_threshold or critical_coverage < 100:
        residual_risk = "high"
    elif total_coverage < settings.coverage_risk_medium_threshold:
        residual_risk = "medium"
    else:
        residual_risk = "low"

    data["total_coverage"] = total_coverage
    data["critical_coverage"] = critical_coverage
    data["residual_risk"] = residual_risk
    return data


COVERAGE_SYSTEM_PROMPT = build_agent_system_prompt("coverage_agent")


def _flatten_rag_results(raw: object) -> list[dict]:
    """Приводит результат rag_search к плоскому списку найденных артефактов.

    ``invoke_with_tools(return_tool_results=True)`` возвращает список
    envelope-объектов ``[{"results": [...]}]``, а прямой вызов реестра —
    один envelope-объект ``{"results": [...]}``. Оба случая сводим к списку
    самих найденных записей, чтобы итерировать их напрямую.
    """
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    if isinstance(raw, list):
        out: list[dict] = []
        for item in raw:
            if isinstance(item, dict) and "results" in item:
                out.extend(item["results"])
            elif isinstance(item, dict):
                out.append(item)
        return out
    return []


def _prepare_requirements_data(requirements: list[dict]) -> list[dict]:
    return [
        {
            "requirement_id": r["requirement_id"],
            "title": r["title"],
            "requirement_text": r["requirement_text"][:200],
            "category": r["category"],
            "priority": r["priority"],
        }
        for r in requirements
    ]


def _prepare_test_cases_data(test_cases: list[dict]) -> list[dict]:
    return [
        {
            "test_case_id": tc["test_case_id"],
            "req": tc["req"],
            "title": tc["title"],
            "test_type": tc["test_type"],
            "priority": tc["priority"],
        }
        for tc in test_cases
    ]


def run_coverage_agent(
    requirements: Optional[list[Requirement]] = None,
    test_cases: Optional[list[TestCase]] = None,
    requirement_ids: Optional[list[str]] = None,
    settings: Optional[Settings] = None,
    llm: Optional[RouterAIProvider] = None,
) -> CoverageReport:
    if settings is None:
        settings = get_settings()
    if llm is None:
        llm = RouterAIProvider(settings)

    from src.tracing import set_span_output, trace_agent

    with trace_agent(
        "Coverage Agent",
        **{"agent.type": "coverage"},
    ) as span:

        if requirements is None:
            if requirement_ids:
                req_data = get_requirements_by_ids(requirement_ids)
            else:
                req_data = get_all_requirements()
            requirements = [Requirement(**r) for r in req_data]

        if test_cases is None:
            if requirement_ids:
                tc_data = get_test_cases_by_reqs(requirement_ids)
            else:
                tc_data = get_all_test_cases()
            test_cases = [TestCase(**tc) for tc in tc_data]

        if requirement_ids:
            req_set = set(requirement_ids)
            requirements = [r for r in requirements if r.requirement_id in req_set]
            test_cases = [tc for tc in test_cases if tc.req in req_set]

        unlinked = get_tests_without_requirements()
        tests_without_requirements = [tc["test_case_id"] for tc in unlinked]

        if span is not None:
            span.set_attribute("requirements.count", len(requirements))
            span.set_attribute("test_cases.count", len(test_cases))

        req_data = _prepare_requirements_data([r.model_dump() for r in requirements])
        tc_data = _prepare_test_cases_data([tc.model_dump() for tc in test_cases])

        req_json = json.dumps(req_data, ensure_ascii=False)
        tc_json = json.dumps(tc_data, ensure_ascii=False)

        similar_tests_json = "[]"
        if settings.rag_enabled:
            try:
                if req_data:
                    from src.skills import ToolRegistry

                    combined_req_text = " ".join(
                        f"{r['title']} {r['requirement_text']}" for r in req_data
                    )
                    registry = ToolRegistry()
                    rag_tool = next(
                        t
                        for t in registry.tools_for_agent("coverage_agent")
                        if t["function"]["name"] == "rag_search"
                    )
                    similar_tests: list[dict] = []
                    try:
                        skill_reference = registry.reference_for_agent("coverage_agent")
                        rag_system_prompt = (
                            "Используй инструмент rag_search, чтобы найти семантически "
                            "похожие тест-кейсы по тексту требований."
                        )
                        if skill_reference:
                            rag_system_prompt = skill_reference + "\n\n" + rag_system_prompt
                        similar_raw = llm.invoke_with_tools(
                        system_prompt=rag_system_prompt,
                            user_message=(
                                "Найди похожие тест-кейсы (коллекция test_cases) по тексту "
                                f"требований:\n{combined_req_text}"
                            ),
                            tools=[rag_tool],
                            model=settings.model_junior,
                            return_tool_results=True,
                            tool_choice={"type": "function", "function": {"name": "rag_search"}},
                        )
                        similar_tests = _flatten_rag_results(json.loads(similar_raw))
                    except Exception as e:
                        logger.warning(f"invoke_with_tools rag_search failed, using registry: {e}")
                        similar_tests = registry.execute_for_agent(
                            "coverage_agent",
                            "rag_search",
                            {
                                "collection": "test_cases",
                                "query": combined_req_text,
                                "top_k": settings.rag_top_k,
                            },
                        )
                        # rag_search возвращает конверт {"results": [...], "error": ...};
                        # единообразно извлекаем список.
                        similar_tests = _flatten_rag_results(similar_tests)
                    similar_tests_json = json.dumps(
                        [
                            {
                                "test_case_id": st.get("id")
                                or st.get("test_case_id")
                                or st.get("requirement_id"),
                                "title": st.get("title", ""),
                                "similarity": round(st.get("similarity", 0), 3),
                            }
                            for st in similar_tests
                            if isinstance(st, dict)
                        ],
                        ensure_ascii=False,
                    )
            except Exception as e:
                logger.warning(f"RAG similarity search skipped: {e}")

        user_message = f"""Проанализируй требования и тест-кейсы.

## Требования ({len(req_data)}):
{req_json}

## Тест-кейсы ({len(tc_data)}):
{tc_json}

## Тесты без требований (req пуст или null):
{json.dumps(tests_without_requirements, ensure_ascii=False)}

## Семантически похожие тест-кейсы
## (возможно покрывающие требования косвенно, без явной привязки REQ):
{similar_tests_json}

Верни отчёт о покрытии в формате JSON."""

        logger.info("Calling LLM for coverage analysis...")

        response = llm.chat_completion(
            messages=[
                {"role": "system", "content": COVERAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=settings.model_senior,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "coverage_report",
                    "schema": build_json_schema("coverage_agent"),
                },
            },
        )

        if span is not None:
            span.set_attribute("llm.response.length", len(response) if response else 0)

        def _on_repair() -> None:
            if span is not None:
                span.add_event("json_repaired", {"agent": "coverage", "tool": "json_repair"})

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = parse_json_response(response, on_repair=_on_repair)

        # Агрегаты считаются в коде по матрице (LLM только классифицирует
        # покрытие по каждому требованию), чтобы исключить ошибки вычислений.
        data = _recompute_coverage(data, settings)
        set_span_output(
            span,
            {"total_coverage": data.get("total_coverage")},
            mime_type="application/json",
        )
        return CoverageReport(**data)
