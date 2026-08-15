"""Интеграционные тесты checkpointing и time-travel (LangGraph).

Проверяем, что компилированный граф с checkpointer:
- сохраняет состояние после каждого узла (история чекпоинтов),
- позволяет «вернуться назад» (time-travel) и доисполнить граф с
  изменённым состоянием, не запуская уже пройденные узлы повторно.
"""

from __future__ import annotations

import pytest

from src.models import ReviewState

from .helpers import to_review_state

pytestmark = [pytest.mark.integration]


async def test_checkpoint_records_state_after_each_node(app_graph_ckpt):
    config = {"configurable": {"thread_id": "ckpt-1"}}
    query = "провести полное ревью"
    raw = await app_graph_ckpt.ainvoke(ReviewState(user_query=query), config=config)
    result = to_review_state(raw)
    assert result.scenario == "full_review"

    # Можно прочитать сохранённое состояние.
    snapshot = await app_graph_ckpt.aget_state(config)
    assert snapshot is not None
    assert snapshot.values["scenario"] == "full_review"

    # История содержит чекпоинты для старта + каждого исполненного узла.
    history = [s async for s in app_graph_ckpt.aget_state_history(config)]
    assert len(history) >= 4  # router, coverage, design, standards (+ финальный)


async def test_time_travel_resume_from_router_with_injected_state(app_graph_ckpt):
    config = {"configurable": {"thread_id": "tt-1"}}
    await app_graph_ckpt.ainvoke(ReviewState(user_query="провести полное ревью"), config=config)

    # Находим чекпоинт сразу после router (next указывает на coverage/дальше).
    history = [s async for s in app_graph_ckpt.aget_state_history(config)]
    router_cp = next(s for s in history if s.next and "coverage" in s.next)

    # Time-travel: «отматываем» к router и внедряем другой сценарий (только standards).
    # Используем полный config чекпоинта (включая checkpoint_ns), а не конструируем свой.
    app_graph_ckpt.update_state(
        router_cp.config,
        values={"scenario": "standards_review", "agents_to_run": ["standards"]},
        as_node="router",
    )

    # Доисполняем граф с нового чекпоинта: должен запуститься только standards.
    final_raw = await app_graph_ckpt.ainvoke(None, config={"configurable": {"thread_id": "tt-1"}})
    final = to_review_state(final_raw)

    assert final.standards_report is not None
    assert final.coverage_report is None
    assert final.design_report is None


async def test_checkpoint_is_isolated_per_thread(app_graph_ckpt):
    c1 = {"configurable": {"thread_id": "iso-a"}}
    c2 = {"configurable": {"thread_id": "iso-b"}}
    q1 = "оценить дизайн тестов"
    q2 = "проверить соответствие стандартам"
    await app_graph_ckpt.ainvoke(ReviewState(user_query=q1), config=c1)
    await app_graph_ckpt.ainvoke(ReviewState(user_query=q2), config=c2)

    s1 = await app_graph_ckpt.aget_state(c1)
    s2 = await app_graph_ckpt.aget_state(c2)
    assert s1.values["scenario"] == "design_review"
    assert s2.values["scenario"] == "standards_review"
