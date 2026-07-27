"""Регрессии review B для полного vision-пайплайна.

Запуск: ``python3 tests/test_review_b_vision_edges.py`` или через pytest.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.schema import PaintTask  # noqa: E402
from vision.bitmap_to_plan import bitmap_to_tasks, merge_adjacent_same_color  # noqa: E402
from vision.prompt_to_bitmap import PALETTE, generate_bitmap, quantize_to_palette  # noqa: E402


PROMPTS = [
    "нарисуй солнце над горами",
    "нарисуй сердце",
    "нарисуй звезду",
    "abc xyz непонятный промпт без ключевых слов",
    "",
    "нарисуй кота и собаку",
]


def test_prompt_matrix_keeps_colored_fill_and_valid_tasks() -> None:
    for size in (4, 8, 16, 20):
        for prompt in PROMPTS:
            bitmap = generate_bitmap(prompt, size, size)
            quantized = quantize_to_palette(bitmap, PALETTE)
            tasks = bitmap_to_tasks(quantized)
            merged = merge_adjacent_same_color(tasks)

            assert len({task.color for task in tasks}) > 1, (
                f"План для {prompt!r} на {size}x{size} потерял цветную заливку"
            )
            for task in tasks + merged:
                # Повторно запускаем __post_init__ на всех полях задачи.
                PaintTask(**asdict(task))


def _run_all() -> None:
    test_prompt_matrix_keeps_colored_fill_and_valid_tasks()
    print("OK: матрица vision 6 промптов x 4 размера пройдена.")


if __name__ == "__main__":
    _run_all()
