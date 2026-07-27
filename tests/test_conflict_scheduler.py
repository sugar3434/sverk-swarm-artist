"""
Тесты для swarm/conflict_scheduler.py.

Запуск напрямую: python3 tests/test_conflict_scheduler.py
Запуск через pytest: pytest tests/test_conflict_scheduler.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.schema import PaintTask
from swarm.canvas_grid import CanvasGrid
from swarm.conflict_scheduler import schedule, validate_no_conflicts

COLOR_TO_DRONE = {
    "black": "drone_academist",
    "red": "drone_expressionist",
    "blue": "drone_minimalist",
    "yellow": "drone_detailer",
}


def _make_grid() -> CanvasGrid:
    # 6x6 сетка на полотне 2.4x2.4 м -> ячейка 0.4x0.4 м, min_separation=0.4 м.
    return CanvasGrid(width_m=2.4, height_m=2.4, cols=6, rows=6, origin=(0.0, 1.5, 0.5), wall_normal="y")


def _close_conflict_scenario():
    """
    ~10 задач с намеренно близкими ячейками у РАЗНЫХ цветов (соседние клетки),
    чтобы гарантированно спровоцировать конфликты, если бы планировщик их
    не разрешал.
    """
    tasks = [
        PaintTask(cell="A1", color="red"),
        PaintTask(cell="A2", color="blue"),      # соседняя с A1 (red) -> конфликт
        PaintTask(cell="B1", color="yellow"),     # соседняя с A1 (red) -> конфликт
        PaintTask(cell="C3", color="black"),
        PaintTask(cell="C4", color="red"),        # соседняя с C3 (black) -> конфликт
        PaintTask(cell="D3", color="blue"),       # соседняя с C3 (black) -> конфликт
        PaintTask(cell="E5", color="yellow"),
        PaintTask(cell="E6", color="black"),      # соседняя с E5 (yellow) -> конфликт
        PaintTask(cell="F5", color="red"),        # соседняя с E5 (yellow) -> конфликт
        PaintTask(cell="F6", color="blue"),       # соседняя с E6 (black) и F5 (red) -> конфликт
    ]
    return tasks


def test_schedule_returns_scheduled_task_for_each_input_task():
    """Число ScheduledTask должно совпадать с числом входных PaintTask."""
    grid = _make_grid()
    tasks = _close_conflict_scenario()
    result = schedule(tasks, grid, COLOR_TO_DRONE, min_separation_m=grid.min_cell_separation_m())
    assert len(result) == len(tasks)


def test_schedule_no_conflicts_on_close_cells_scenario():
    """
    Основной тест: сценарий с намеренно близкими ячейками разных цветов —
    validate_no_conflicts должна вернуть пустой список.
    """
    grid = _make_grid()
    tasks = _close_conflict_scenario()
    min_sep = grid.min_cell_separation_m()

    result = schedule(tasks, grid, COLOR_TO_DRONE, min_separation_m=min_sep, drone_speed_mps=0.4)
    problems = validate_no_conflicts(result, grid, min_sep)

    assert problems == [], f"обнаружены конфликты в расписании: {problems}"


def test_same_drone_tasks_are_sequential():
    """Задачи одного дрона никогда не должны пересекаться по времени."""
    grid = _make_grid()
    tasks = [
        PaintTask(cell="A1", color="red"),
        PaintTask(cell="B2", color="red"),
        PaintTask(cell="C3", color="red"),
        PaintTask(cell="D4", color="red"),
    ]
    result = schedule(tasks, grid, COLOR_TO_DRONE, min_separation_m=grid.min_cell_separation_m())

    result_sorted = sorted(result, key=lambda s: s.start_offset_s)
    for i in range(len(result_sorted) - 1):
        a, b = result_sorted[i], result_sorted[i + 1]
        assert a.end_offset_s <= b.start_offset_s + 1e-9, (
            f"задачи одного дрона пересекаются по времени: {a.task.cell} и {b.task.cell}"
        )


def test_far_apart_different_colors_can_overlap_in_time():
    """
    Далёкие друг от друга ячейки разных дронов НЕ обязаны быть
    последовательными — планировщик не должен искусственно сериализовать всё.
    """
    grid = _make_grid()
    tasks = [
        PaintTask(cell="A1", color="red"),
        PaintTask(cell="F6", color="blue"),  # максимально далеко от A1
    ]
    result = schedule(tasks, grid, COLOR_TO_DRONE, min_separation_m=grid.min_cell_separation_m())

    by_cell = {s.task.cell: s for s in result}
    a1 = by_cell["A1"]
    f6 = by_cell["F6"]
    # Оба должны стартовать в t=0, так как это первая задача каждого дрона
    # и расстояние между ячейками намного больше порога конфликта.
    assert abs(a1.start_offset_s - 0.0) < 1e-9
    assert abs(f6.start_offset_s - 0.0) < 1e-9


def test_validate_no_conflicts_detects_injected_conflict():
    """
    Проверка, что validate_no_conflicts действительно обнаруживает конфликт,
    если он искусственно внедрён (гарантия того, что тест не «пустой»).
    """
    from common.schema import ScheduledTask

    grid = _make_grid()
    min_sep = grid.min_cell_separation_m()

    task_a = PaintTask(cell="A1", color="red")
    task_b = PaintTask(cell="A2", color="blue")  # соседняя ячейка -> близко

    bad_schedule = [
        ScheduledTask(task=task_a, drone_id="drone_expressionist", start_offset_s=0.0, end_offset_s=2.0),
        ScheduledTask(task=task_b, drone_id="drone_minimalist", start_offset_s=1.0, end_offset_s=3.0),
    ]

    problems = validate_no_conflicts(bad_schedule, grid, min_sep)
    assert len(problems) > 0, "искусственно внедрённый конфликт должен быть обнаружен"


def test_schedule_raises_on_unknown_color():
    """Если у задачи цвет не найден в color_to_drone — должна быть внятная ошибка."""
    grid = _make_grid()
    tasks = [PaintTask(cell="A1", color="red")]
    incomplete_map = {"blue": "drone_minimalist"}

    raised = False
    try:
        schedule(tasks, grid, incomplete_map, min_separation_m=grid.min_cell_separation_m())
    except ValueError:
        raised = True
    assert raised


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
