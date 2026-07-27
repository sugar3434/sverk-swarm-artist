"""
Тесты для swarm/canvas_grid.py.

Запуск напрямую: python3 tests/test_canvas_grid.py
Запуск через pytest: pytest tests/test_canvas_grid.py
Используются только `assert` — без зависимости от pytest в коде тестов.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarm.canvas_grid import CanvasGrid


def test_cell_to_world_basic():
    """Проверка базового вычисления мировых координат центра ячейки."""
    grid = CanvasGrid(width_m=2.4, height_m=2.4, cols=6, rows=6, origin=(0.0, 1.5, 0.5), wall_normal="y")

    x, y, z = grid.cell_to_world("A1")
    # A1 -> col=0, row=0 -> центр первой ячейки: 0.5*cell_size
    assert abs(x - 0.2) < 1e-9
    assert abs(y - 1.5) < 1e-9  # фиксированная дистанция от стены
    assert abs(z - 0.7) < 1e-9

    x2, y2, z2 = grid.cell_to_world("F6")
    assert abs(x2 - 2.2) < 1e-9
    assert abs(y2 - 1.5) < 1e-9
    assert abs(z2 - 2.7) < 1e-9


def test_cell_to_world_wall_normal_x():
    """Проверка ориентации полотна вдоль другой нормали (wall_normal='x')."""
    grid = CanvasGrid(width_m=2.0, height_m=2.0, cols=4, rows=4, origin=(3.0, 0.0, 0.0), wall_normal="x")
    x, y, z = grid.cell_to_world("A1")
    assert abs(x - 3.0) < 1e-9  # фиксирована по x
    assert y > 0
    assert z > 0


def test_col_row_cell_id_roundtrip():
    """Проверка обратимости col_row_to_cell_id <-> cell_id_to_col_row."""
    grid = CanvasGrid(width_m=2.0, height_m=2.0, cols=10, rows=10)

    for col in range(10):
        for row in range(10):
            cell_id = grid.col_row_to_cell_id(col, row)
            back_col, back_row = grid.cell_id_to_col_row(cell_id)
            assert (col, row) == (back_col, back_row), f"round-trip fail for ({col},{row}) -> {cell_id}"


def test_cell_id_letters_multi_char():
    """Проверка, что при >26 столбцов буквенная нумерация продолжается (AA, AB, ...)."""
    grid = CanvasGrid(width_m=30.0, height_m=2.0, cols=30, rows=2)
    cell_id = grid.col_row_to_cell_id(26, 0)  # 27-й столбец -> 'AA'
    assert cell_id == "AA1", f"ожидалось AA1, получено {cell_id}"
    col, row = grid.cell_id_to_col_row("AA1")
    assert (col, row) == (26, 0)


def test_out_of_bounds_raises_value_error():
    """Проверка, что выход за границы сетки кидает ValueError с понятным сообщением."""
    grid = CanvasGrid(width_m=2.4, height_m=2.4, cols=6, rows=6)

    raised = False
    try:
        grid.cell_id_to_col_row("G1")  # столбец 6 >= cols=6
    except ValueError as exc:
        raised = True
        assert "столбец" in str(exc).lower() or "column" in str(exc).lower() or "6" in str(exc)
    assert raised, "ожидался ValueError при выходе за границы столбцов"

    raised = False
    try:
        grid.cell_id_to_col_row("A7")  # строка 6 >= rows=6
    except ValueError as exc:
        raised = True
    assert raised, "ожидался ValueError при выходе за границы строк"

    raised = False
    try:
        grid.col_row_to_cell_id(6, 0)
    except ValueError:
        raised = True
    assert raised, "ожидался ValueError в col_row_to_cell_id при выходе за границы"

    raised = False
    try:
        grid.cell_id_to_col_row("bad_id")
    except ValueError:
        raised = True
    assert raised, "ожидался ValueError на некорректном формате id"


def test_min_cell_separation_m():
    """Проверка вычисления безопасного расстояния между ячейками."""
    grid = CanvasGrid(width_m=2.4, height_m=1.2, cols=6, rows=6)
    # cell_width = 0.4, cell_height = 0.2 -> min = 0.2
    assert abs(grid.min_cell_separation_m() - 0.2) < 1e-9


def test_invalid_constructor_args():
    """Проверка валидации конструктора (размеры, wall_normal)."""
    raised = False
    try:
        CanvasGrid(width_m=-1.0, height_m=2.0, cols=4, rows=4)
    except ValueError:
        raised = True
    assert raised

    raised = False
    try:
        CanvasGrid(width_m=2.0, height_m=2.0, cols=4, rows=4, wall_normal="q")
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
