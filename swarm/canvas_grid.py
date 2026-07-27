"""
Сетка вертикального полотна для покраски дронами («Рой дронов-художников»).

Полотно — плоская вертикальная стена (Регламент, п. 2.7: не менее 2х2 м),
разбитая на сетку cols x rows ячеек. Модуль переводит id ячейки ("B3") в
мировые координаты точки в центре ячейки и обратно, а также вычисляет
безопасное расстояние между соседними ячейками (для anti-collision
планировщика в swarm/conflict_scheduler.py).

Модуль не использует rclpy/sverk_interfaces — работает офлайн (см.
docs/ARCHITECTURE_CONTRACT.md).
"""
from __future__ import annotations

from typing import Tuple

# Оси мировой системы координат, вдоль которых может быть ориентирована
# нормаль стены (в какую сторону "смотрит" полотно).
_VALID_WALL_NORMALS = ("x", "y", "z")


def _col_to_letter(col: int) -> str:
    """Переводит номер столбца (0-based) в буквенный id (0->A, 25->Z, 26->AA, ...)."""
    col += 1
    letters = ""
    while col > 0:
        col, rem = divmod(col - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def _letter_to_col(letters: str) -> int:
    """Переводит буквенный id столбца в номер (0-based)."""
    col = 0
    for ch in letters.upper():
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Некорректная буква столбца: {ch!r} в {letters!r}")
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return col - 1


def _split_cell_id(cell_id: str) -> Tuple[str, str]:
    """Разбивает id ячейки на буквенную (столбец) и цифровую (строка) части."""
    letters = ""
    digits = ""
    for ch in cell_id:
        if ch.isalpha():
            if digits:
                raise ValueError(f"Некорректный формат id ячейки: {cell_id!r} (буквы после цифр)")
            letters += ch
        elif ch.isdigit():
            digits += ch
        else:
            raise ValueError(f"Некорректный символ {ch!r} в id ячейки {cell_id!r}")
    if not letters or not digits:
        raise ValueError(
            f"Некорректный id ячейки {cell_id!r}: ожидался формат «буква(ы)+число», напр. 'B3'"
        )
    return letters, digits


class CanvasGrid:
    """
    Сетка вертикального полотна cols x rows на плоскости стены.

    Полотно располагается в плоскости, перпендикулярной оси `wall_normal`
    (по умолчанию "y" — дрон подлетает вдоль оси Y на фиксированное
    расстояние `origin[1]` от стены). Ячейка (col=0, row=0) — левый нижний
    угол полотна, id "A1".

    Параметры:
        width_m:    ширина полотна, м (Регламент 2.7: не менее 2 м).
        height_m:   высота полотна, м (не менее 2 м).
        cols:       число столбцов сетки.
        rows:       число строк сетки.
        origin:     (x0, y0, z0) — мировые координаты левого нижнего угла
                    полотна (соответствует центру ячейки A1 по плоским осям,
                    а по нормали стены — фиксированное расстояние подлёта).
        wall_normal: ось, вдоль которой "смотрит" полотно ("x", "y" или "z");
                    по этой оси координата всех ячеек фиксирована = origin
                    по соответствующей координате (дрон держит одно и то же
                    расстояние от стены при покраске любой ячейки).
    """

    def __init__(
        self,
        width_m: float,
        height_m: float,
        cols: int,
        rows: int,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        wall_normal: str = "y",
    ) -> None:
        if width_m <= 0 or height_m <= 0:
            raise ValueError("width_m и height_m должны быть положительными")
        if cols <= 0 or rows <= 0:
            raise ValueError("cols и rows должны быть положительными целыми числами")
        if wall_normal not in _VALID_WALL_NORMALS:
            raise ValueError(f"wall_normal должен быть одним из {_VALID_WALL_NORMALS}, получено {wall_normal!r}")

        self.width_m = width_m
        self.height_m = height_m
        self.cols = cols
        self.rows = rows
        self.origin = origin
        self.wall_normal = wall_normal

        self.cell_width_m = width_m / cols
        self.cell_height_m = height_m / rows

    # ------------------------------------------------------------------
    # Преобразования id ячейки <-> (col, row)
    # ------------------------------------------------------------------

    def col_row_to_cell_id(self, col: int, row: int) -> str:
        """Переводит (col, row) (0-based) в id ячейки, напр. (1, 2) -> 'B3'."""
        self._validate_col_row(col, row)
        return f"{_col_to_letter(col)}{row + 1}"

    def cell_id_to_col_row(self, cell_id: str) -> Tuple[int, int]:
        """Переводит id ячейки (напр. 'B3') в (col, row) (0-based), с валидацией границ."""
        letters, digits = _split_cell_id(cell_id)
        col = _letter_to_col(letters)
        row = int(digits) - 1
        self._validate_col_row(col, row, source=cell_id)
        return col, row

    def _validate_col_row(self, col: int, row: int, source: str = "") -> None:
        origin_msg = f" (из id ячейки {source!r})" if source else ""
        if not (0 <= col < self.cols):
            raise ValueError(
                f"Столбец {col} вне границ сетки{origin_msg}: допустимо 0..{self.cols - 1} "
                f"(всего столбцов: {self.cols})"
            )
        if not (0 <= row < self.rows):
            raise ValueError(
                f"Строка {row} вне границ сетки{origin_msg}: допустимо 0..{self.rows - 1} "
                f"(всего строк: {self.rows})"
            )

    # ------------------------------------------------------------------
    # Мировые координаты
    # ------------------------------------------------------------------

    def cell_to_world(self, cell_id: str) -> Tuple[float, float, float]:
        """
        Переводит id ячейки в мировые координаты (x, y, z) её центра.

        Координата вдоль плоскости стены вычисляется как origin + смещение
        центра ячейки; координата вдоль wall_normal фиксирована и равна
        соответствующей компоненте origin (дрон подлетает на фиксированное
        расстояние от полотна независимо от ячейки).
        """
        col, row = self.cell_id_to_col_row(cell_id)
        return self._col_row_to_world(col, row)

    def _col_row_to_world(self, col: int, row: int) -> Tuple[float, float, float]:
        x0, y0, z0 = self.origin
        # Координаты центра ячейки в плоскости стены (0-based индексация,
        # +0.5 — смещение к центру ячейки, а не к её левому/нижнему краю).
        along_width = (col + 0.5) * self.cell_width_m
        along_height = (row + 0.5) * self.cell_height_m

        if self.wall_normal == "y":
            # Стена в плоскости XZ, дрон подлетает вдоль Y на фикс. дистанцию y0.
            x = x0 + along_width
            y = y0
            z = z0 + along_height
        elif self.wall_normal == "x":
            # Стена в плоскости YZ, подлёт вдоль X на фикс. дистанцию x0.
            x = x0
            y = y0 + along_width
            z = z0 + along_height
        else:  # wall_normal == "z"
            # Стена в плоскости XY (напр. "потолок"/пол-полотно), подлёт вдоль Z.
            x = x0 + along_width
            y = y0 + along_height
            z = z0

        return (x, y, z)

    # ------------------------------------------------------------------
    # Параметры безопасности
    # ------------------------------------------------------------------

    def min_cell_separation_m(self) -> float:
        """
        Минимальное «безопасное» расстояние между соседними ячейками, м.

        Равно минимуму из ширины и высоты одной ячейки — используется
        планировщиком коллизий (swarm/conflict_scheduler.py) как порог,
        ниже которого две задачи разных дронов не должны выполняться
        параллельно по времени.
        """
        return min(self.cell_width_m, self.cell_height_m)

    def __repr__(self) -> str:
        return (
            f"CanvasGrid(width_m={self.width_m}, height_m={self.height_m}, "
            f"cols={self.cols}, rows={self.rows}, origin={self.origin}, "
            f"wall_normal={self.wall_normal!r})"
        )


# --------------------------------------------------------------------------
# Демонстрация
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Регламент 2.7: полотно не менее 2x2 м — берём 2.4x2.4 м, сетка 6x6.
    grid = CanvasGrid(
        width_m=2.4,
        height_m=2.4,
        cols=6,
        rows=6,
        origin=(0.0, 1.5, 0.5),  # подлёт на 1.5 м от стены, низ полотна на высоте 0.5 м
        wall_normal="y",
    )

    print(grid)
    print(f"Размер ячейки: {grid.cell_width_m:.3f} x {grid.cell_height_m:.3f} м")
    print(f"min_cell_separation_m = {grid.min_cell_separation_m():.3f} м")
    print()

    for cell_id in ("A1", "F6", "C3"):
        x, y, z = grid.cell_to_world(cell_id)
        col, row = grid.cell_id_to_col_row(cell_id)
        print(f"{cell_id} (col={col}, row={row}) -> world=({x:.3f}, {y:.3f}, {z:.3f})")

    print()
    try:
        grid.cell_id_to_col_row("G1")
    except ValueError as exc:
        print(f"Ожидаемая ошибка на выходе за границы: {exc}")
