"""
Тест движка диалога агентов на офлайн-клиенте (без сети).

Проверяет:
    1. run_dialogue() с OfflineRuleBasedClient() на 3-4 черновых задачах
       возвращает валидный Plan.
    2. Plan.cells непустой, и все PaintTask в нём проходят __post_init__
       валидацию (создаются без исключений).
    3. Стенограмма (Plan.transcript) содержит реплики всех 4 личностей
       минимум по одному разу.
    4. Весь прогон укладывается менее чем в 5 секунд — офлайн-клиент не
       должен искусственно тормозить диалог.

Запуск: `python3 tests/test_dialogue_engine_offline.py`
Совместим с pytest (использует обычные функции test_* и assert).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.schema import PaintTask, Plan  # noqa: E402

from agents.broadcast import Broadcaster  # noqa: E402
from agents.dialogue_engine import PERSONA_ORDER, run_dialogue  # noqa: E402
from agents.llm_client import OfflineRuleBasedClient  # noqa: E402


def _make_draft_tasks() -> list[PaintTask]:
    return [
        PaintTask(cell="A1", color="black", duration_s=1.5, passes=1),
        PaintTask(cell="B2", color="red", duration_s=1.0, passes=1),
        PaintTask(cell="C3", color="blue", duration_s=2.0, passes=1),
        PaintTask(cell="D4", color="yellow", duration_s=1.2, passes=1),
    ]


def test_run_dialogue_returns_valid_plan_within_time_budget() -> None:
    draft_tasks = _make_draft_tasks()
    llm = OfflineRuleBasedClient(seed=123)
    # Не пишем лог на диск в тесте, чтобы не засорять workspace и не зависеть от ФС.
    broadcaster = Broadcaster(log_path=None)

    start = time.monotonic()
    plan = run_dialogue(
        prompt="нарисуй сокола",
        draft_tasks=draft_tasks,
        llm=llm,
        broadcaster=broadcaster,
        rounds=2,
        time_budget_s=180.0,
    )
    elapsed = time.monotonic() - start

    assert isinstance(plan, Plan), "run_dialogue должен вернуть объект Plan"
    assert elapsed < 5.0, f"Диалог на офлайн-клиенте занял слишком много времени: {elapsed:.2f} c"

    assert plan.cells, "Plan.cells не должен быть пустым"
    for task in plan.cells:
        assert isinstance(task, PaintTask), "Каждый элемент cells должен быть PaintTask"
        # PaintTask.__post_init__ уже выполнился при создании — если бы данные
        # были некорректны, конструктор бросил бы исключение ещё внутри движка.
        # Дополнительно инвариант перепроверяем явно:
        from common.schema import COLORS

        assert task.color in COLORS
        assert task.duration_s > 0
        assert task.passes >= 1

    assert plan.transcript, "Стенограмма диалога не должна быть пустой"

    agents_in_transcript = {turn.agent for turn in plan.transcript}
    for persona_name in PERSONA_ORDER:
        assert persona_name in agents_in_transcript, (
            f"В стенограмме отсутствуют реплики персонажа @{persona_name}"
        )

    # Каждая личность должна была высказаться минимум один раз за диалог.
    counts = {name: 0 for name in PERSONA_ORDER}
    for turn in plan.transcript:
        if turn.agent in counts:
            counts[turn.agent] += 1
    for persona_name, count in counts.items():
        assert count >= 1, f"@{persona_name} должен высказаться минимум 1 раз, получено {count}"

    print(f"OK: run_dialogue вернул валидный Plan за {elapsed:.3f} c, "
          f"{len(plan.cells)} ячеек, {len(plan.transcript)} реплик.")


def test_run_dialogue_reproducible_with_same_seed() -> None:
    """Дополнительная проверка: офлайн-клиент с одинаковым seed воспроизводим."""
    draft_tasks_a = _make_draft_tasks()
    draft_tasks_b = _make_draft_tasks()
    broadcaster = Broadcaster(log_path=None)

    plan_a = run_dialogue(
        prompt="нарисуй сокола",
        draft_tasks=draft_tasks_a,
        llm=OfflineRuleBasedClient(seed=99),
        broadcaster=broadcaster,
        rounds=1,
        time_budget_s=60.0,
    )
    plan_b = run_dialogue(
        prompt="нарисуй сокола",
        draft_tasks=draft_tasks_b,
        llm=OfflineRuleBasedClient(seed=99),
        broadcaster=broadcaster,
        rounds=1,
        time_budget_s=60.0,
    )

    texts_a = [t.text for t in plan_a.transcript]
    texts_b = [t.text for t in plan_b.transcript]
    assert texts_a == texts_b, "При одинаковом seed офлайн-диалог должен быть воспроизводим"
    print("OK: офлайн-диалог воспроизводим при одинаковом seed.")


def _run_all() -> None:
    test_run_dialogue_returns_valid_plan_within_time_budget()
    test_run_dialogue_reproducible_with_same_seed()
    print("\nВСЕ ТЕСТЫ test_dialogue_engine_offline.py ПРОЙДЕНЫ.")


if __name__ == "__main__":
    _run_all()
