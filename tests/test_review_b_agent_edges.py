"""Регрессии review B для LLM/диалога/трансляции.

Запуск: ``python3 tests/test_review_b_agent_edges.py`` или через pytest.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.broadcast import Broadcaster  # noqa: E402
from agents.dialogue_engine import _apply_patch, run_dialogue  # noqa: E402
from agents.llm_client import LLMClient, OfflineRuleBasedClient  # noqa: E402
from agents.personas import COORDINATOR_PROMPT, PERSONAS  # noqa: E402
from common.schema import DialogueTurn, PaintTask  # noqa: E402


def test_offline_coordinator_preserves_empty_plan() -> None:
    client = OfflineRuleBasedClient(seed=3)
    reply = client.chat(
        COORDINATOR_PROMPT,
        "Финальные задачи после обсуждения (JSON): []\nВерни JSON.",
    )
    assert json.loads(reply)["cells"] == []


def test_offline_persona_does_not_repeat_consecutively() -> None:
    # seed=2 воспроизводил подряд одинаковую реплику до исправления.
    client = OfflineRuleBasedClient(seed=2)
    system_prompt = PERSONAS["Академист"]["system_prompt"]
    replies = [client.chat(system_prompt, "Текущий план (JSON): []") for _ in range(8)]
    assert all(left != right for left, right in zip(replies, replies[1:]))


class _FixedCoordinatorClient(LLMClient):
    def __init__(self, cells: list[dict]):
        self.cells = cells

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if "Координатор" in system_prompt:
            return json.dumps({"cells": self.cells, "notes": "test"})
        return "Реплика без патча."


def test_unknown_agent_patch_and_invalid_recolor_are_ignored() -> None:
    tasks = [PaintTask("A1", "red")]
    broadcaster = Broadcaster(log_path=None, to_stdout=False)
    _apply_patch(
        tasks,
        {"suggest": [{"cell": "Z99", "action": "remove"}]},
        "Тест",
        broadcaster,
    )
    _apply_patch(
        tasks,
        {"suggest": [{"cell": "A1", "action": "recolor", "color": "purple"}]},
        "Тест",
        broadcaster,
    )
    assert len(tasks) == 1
    assert tasks[0].color == "red"


def test_empty_coordinator_plan_is_accepted_and_unknown_cells_filtered() -> None:
    broadcaster = Broadcaster(log_path=None, to_stdout=False)
    empty = run_dialogue(
        "test",
        [PaintTask("A1", "red")],
        _FixedCoordinatorClient([]),
        broadcaster,
        rounds=0,
        time_budget_s=1.0,
    )
    assert empty.cells == []

    unknown = run_dialogue(
        "test",
        [PaintTask("A1", "red")],
        _FixedCoordinatorClient([{"cell": "Z99", "color": "red"}]),
        broadcaster,
        rounds=0,
        time_budget_s=1.0,
    )
    assert unknown.cells == []


def test_priorities_are_assigned_with_outline_first() -> None:
    draft = [
        PaintTask("B2", "red", note="заливка внутренней области"),
        PaintTask("A1", "black", note="контурная ячейка — обводит Академист"),
        PaintTask("C3", "blue", note="заливка внутренней области"),
    ]
    plan = run_dialogue(
        "test",
        draft,
        _FixedCoordinatorClient([
            {"cell": task.cell, "color": task.color, "priority": 0, "note": task.note}
            for task in draft
        ]),
        Broadcaster(None, False),
        rounds=0,
        time_budget_s=1.0,
    )
    assert len({task.priority for task in plan.cells}) > 1
    outline_priority = next(task.priority for task in plan.cells if task.color == "black")
    assert outline_priority == min(task.priority for task in plan.cells)


class _SlowClient(LLMClient):
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        time.sleep(1.0)
        return "слишком поздно"


def test_dialogue_honors_tiny_budget_even_if_client_blocks() -> None:
    start = time.monotonic()
    plan = run_dialogue(
        "test",
        [PaintTask("A1", "black"), PaintTask("B2", "red")],
        _SlowClient(),
        Broadcaster(None, False),
        rounds=3,
        time_budget_s=0.15,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 0.5
    assert len(plan.cells) == 2


def test_broadcaster_nested_path_and_none() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        nested = os.path.join(root, "a", "b", "c", "dialogue.jsonl")
        Broadcaster(nested, False).emit(DialogueTurn("test", "ok"))
        assert os.path.isfile(nested)
    Broadcaster(None, False).emit(DialogueTurn("test", "ok"))


def _run_all() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")


if __name__ == "__main__":
    _run_all()
