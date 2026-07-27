"""
Тесты для полного офлайн-пайплайна vision: prompt_to_bitmap -> quantize ->
bitmap_to_tasks -> merge_adjacent_same_color.

Запуск напрямую: python3 tests/test_bitmap_to_plan.py
Запуск через pytest: pytest tests/test_bitmap_to_plan.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.schema import COLORS, PaintTask
from vision.bitmap_to_plan import bitmap_to_tasks, merge_adjacent_same_color
from vision.prompt_to_bitmap import (
    detect_outline_cells,
    generate_bitmap,
    quantize_to_palette,
)

PALETTE = {
    "black": (20, 20, 20),
    "red": (210, 30, 30),
    "blue": (30, 60, 210),
    "yellow": (230, 200, 30),
}


def _full_pipeline(prompt: str, cols: int, rows: int):
    bmp = generate_bitmap(prompt, cols, rows)
    quant = quantize_to_palette(bmp, PALETTE)
    outline = detect_outline_cells(quant)
    tasks = bitmap_to_tasks(quant, base_duration_s=1.2)
    merged = merge_adjacent_same_color(tasks)
    return quant, outline, tasks, merged


def test_generate_bitmap_shape_and_determinism():
    """Проверка размерности и детерминированности генерации растра."""
    prompt = "нарисуй солнце над горами"
    bmp1 = generate_bitmap(prompt, cols=20, rows=15)
    bmp2 = generate_bitmap(prompt, cols=20, rows=15)

    assert len(bmp1) == 15, "число строк должно совпадать с rows"
    assert len(bmp1[0]) == 20, "число столбцов должно совпадать с cols"
    assert bmp1 == bmp2, "генерация должна быть детерминированной для одного промпта"

    for row in bmp1:
        for pixel in row:
            assert len(pixel) == 3
            for channel in pixel:
                assert 0 <= channel <= 255


def test_quantize_to_palette_values_are_valid():
    """Проверка, что quantize_to_palette возвращает либо None, либо ключ палитры."""
    bmp = generate_bitmap("звезда", cols=16, rows=16)
    quant = quantize_to_palette(bmp, PALETTE)

    assert len(quant) == 16
    assert len(quant[0]) == 16
    for row in quant:
        for value in row:
            assert value is None or value in PALETTE


def test_pipeline_produces_at_least_one_task():
    """Полный пайплайн должен произвести хотя бы одну задачу покраски."""
    _, _, tasks, merged = _full_pipeline("нарисуй солнце над горами", cols=20, rows=20)
    assert len(tasks) >= 1, "должна быть хотя бы одна задача"
    assert len(merged) >= 1, "после объединения должна остаться хотя бы одна задача"
    assert len(merged) <= len(tasks), "объединение не должно увеличивать число задач"


def test_all_tasks_pass_post_init_validation():
    """Все PaintTask из пайплайна должны быть валидны согласно __post_init__ схемы."""
    for prompt in ["нарисуй солнце над горами", "heart", "cat and flower", "случайный узор xyz"]:
        _, _, tasks, merged = _full_pipeline(prompt, cols=18, rows=18)
        for t in tasks + merged:
            assert isinstance(t, PaintTask)
            # __post_init__ уже вызван при создании; здесь дополнительно
            # пересоздаём объект с теми же полями, чтобы явно спровоцировать
            # повторный прогон валидации и убедиться, что она проходит.
            PaintTask(cell=t.cell, color=t.color, duration_s=t.duration_s, passes=t.passes)
            assert t.color in COLORS
            assert t.duration_s > 0
            assert t.passes >= 1


def test_outline_cells_get_black_color():
    """Контурные ячейки в итоговых задачах должны иметь color == 'black'."""
    quant, outline, tasks, _ = _full_pipeline("нарисуй солнце над горами", cols=20, rows=20)
    assert len(outline) > 0, "в этой сцене должны быть контурные ячейки"

    # Строим множество id контурных ячеек в терминах letters+digits (как в cell).
    from vision.bitmap_to_plan import _cell_id

    outline_ids = {_cell_id(r, c) for (r, c) in outline}

    tasks_by_cell = {t.cell: t for t in tasks}
    checked = 0
    for cell_id in outline_ids:
        if cell_id in tasks_by_cell:
            assert tasks_by_cell[cell_id].color == "black", (
                f"контурная ячейка {cell_id} должна иметь цвет 'black', "
                f"получено {tasks_by_cell[cell_id].color!r}"
            )
            checked += 1
    assert checked > 0, "должна быть проверена хотя бы одна контурная ячейка"


def test_max_tasks_limit_respected_when_outline_small():
    """
    Если контурных ячеек меньше лимита, итоговое число задач не превышает лимит
    (небольшой квадрат 8x8 внутри поля 10x10 даёт контур меньше лимита 40,
    оставляя бюджет для прореживания внутренних ячеек).
    """
    rows, cols = 10, 10
    quant = [[None] * cols for _ in range(rows)]
    for r in range(1, 9):
        for c in range(1, 9):
            quant[r][c] = "red"

    outline = detect_outline_cells(quant)
    assert len(outline) < 40, "контур должен быть меньше лимита для проверки прореживания"

    tasks = bitmap_to_tasks(quant, max_tasks=40)
    assert len(tasks) <= 40
    black_count = sum(1 for t in tasks if t.color == "black")
    assert black_count == len(outline), "все контурные ячейки должны быть сохранены"


def test_outline_always_kept_even_if_it_exceeds_limit():
    """Даже если контурных ячеек больше лимита, все они должны быть сохранены."""
    rows, cols = 20, 20
    quant = [[None] * cols for _ in range(rows)]
    for r in range(1, 19):
        for c in range(1, 19):
            quant[r][c] = "red"

    outline = detect_outline_cells(quant)
    tasks = bitmap_to_tasks(quant, max_tasks=40)
    black_count = sum(1 for t in tasks if t.color == "black")
    assert black_count == len(outline), "контурные ячейки не должны прореживаться, даже если их больше лимита"


def test_merge_does_not_increase_task_count():
    """merge_adjacent_same_color не должна увеличивать число задач."""
    _, _, tasks, merged = _full_pipeline("дом с ёлкой", cols=22, rows=22)
    assert len(merged) <= len(tasks)


def _run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  OK: {t.__name__}")
    print(f"Пройдено тестов: {passed}/{len(tests)}")


if __name__ == "__main__":
    _run_all()
