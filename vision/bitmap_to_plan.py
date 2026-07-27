"""
Преобразование квантованного растра в список задач покраски (PaintTask).

Модуль берёт результат `vision.prompt_to_bitmap.quantize_to_palette`
(и `detect_outline_cells`) и строит черновой список `PaintTask` — по одной
задаче на закрашенную ячейку сетки полотна. Мировые координаты (x/y/z) и
итоговый приоритет НЕ заполняются здесь — это делают `swarm/canvas_grid.py`
и `agents/dialogue_engine.py` на следующих этапах пайплайна (см.
docs/ARCHITECTURE_CONTRACT.md, раздел «Разбиение работы»).

Правила:
- id ячейки формируется как буква-столбец (A, B, C, ...) + номер строки
  с 1 (A1, B3, ...), т.е. col=0 -> 'A', row=0 -> '1'.
- Контурные ячейки (см. detect_outline_cells) принудительно окрашиваются
  в "black" — по регламенту чёрный дрон-Академист обводит контур поверх/
  вместо исходного цвета — и получают чуть большую длительность (+0.3с),
  чтобы контурный проход был выразительнее.
- Итоговое число задач ограничено (по умолчанию 60) под 15-минутное окно
  соревнования: если ячеек больше, лишние НЕ-контурные ячейки прореживаются
  равномерно, все контурные ячейки сохраняются.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from common.schema import PaintTask

# Разумный верхний предел числа задач под 15-минутное окно полёта.
MAX_TASKS_DEFAULT = 60

# Дополнительная длительность контурного прохода (Академист обводит контур).
OUTLINE_EXTRA_DURATION_S = 0.3


def _col_to_letter(col: int) -> str:
    """Переводит номер столбца (0-based) в буквенный id (0->A, 25->Z, 26->AA, ...)."""
    col += 1
    letters = ""
    while col > 0:
        col, rem = divmod(col - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def _cell_id(row: int, col: int) -> str:
    """Строит id ячейки "<буква_столбца><номер_строки>", напр. (row=2, col=1) -> 'B3'."""
    return f"{_col_to_letter(col)}{row + 1}"


def bitmap_to_tasks(
    quantized: List[List[Optional[str]]],
    base_duration_s: float = 1.2,
    max_tasks: int = MAX_TASKS_DEFAULT,
) -> List[PaintTask]:
    """
    Строит список PaintTask из квантованного растра.

    Параметры:
        quantized:       результат quantize_to_palette (rows x cols, цвет|None).
        base_duration_s: базовая длительность распыления одной ячейки, с.
        max_tasks:       верхняя граница числа задач (прореживание не-контурных
                          ячеек при превышении, все контурные ячейки сохраняются).

    Возвращает:
        список PaintTask без заполненных x/y/z (их заполнит CanvasGrid) и
        без приоритета (заполнит диалог агентов) — только cell/color/duration/passes.
    """
    from vision.prompt_to_bitmap import detect_outline_cells

    outline_cells: Set[Tuple[int, int]] = detect_outline_cells(quantized)

    filled_cells: List[Tuple[int, int, str]] = []
    for r, row in enumerate(quantized):
        for c, color in enumerate(row):
            if color is not None:
                filled_cells.append((r, c, color))

    if not filled_cells:
        return []

    outline_filled = [(r, c, color) for (r, c, color) in filled_cells if (r, c) in outline_cells]
    non_outline_filled = [(r, c, color) for (r, c, color) in filled_cells if (r, c) not in outline_cells]

    # Прореживание: если суммарно ячеек больше лимита, равномерно выбрасываем
    # часть НЕ-контурных ячеек, оставляя все контурные (контур важен для
    # распознаваемости VLM-судьёй — см. регламент, критерий similarity score).
    budget_for_non_outline = max(0, max_tasks - len(outline_filled))
    if len(non_outline_filled) > budget_for_non_outline and budget_for_non_outline > 0:
        step = len(non_outline_filled) / budget_for_non_outline
        thinned = []
        idx = 0.0
        while len(thinned) < budget_for_non_outline and int(idx) < len(non_outline_filled):
            thinned.append(non_outline_filled[int(idx)])
            idx += step
        non_outline_filled = thinned
    elif budget_for_non_outline == 0:
        non_outline_filled = []

    tasks: List[PaintTask] = []
    for r, c, color in outline_filled:
        tasks.append(
            PaintTask(
                cell=_cell_id(r, c),
                color="black",
                duration_s=base_duration_s + OUTLINE_EXTRA_DURATION_S,
                passes=1,
                note="контурная ячейка — обводит Академист (чёрный)",
            )
        )
    for r, c, color in non_outline_filled:
        tasks.append(
            PaintTask(
                cell=_cell_id(r, c),
                color=color,
                duration_s=base_duration_s,
                passes=1,
                note="заливка внутренней области",
            )
        )

    return tasks


def _cell_id_to_col_row_local(cell_id: str) -> Tuple[int, int]:
    """Локальный разбор id ячейки в (col, row), 0-based, без зависимости от CanvasGrid."""
    letters = ""
    digits = ""
    for ch in cell_id:
        if ch.isalpha():
            letters += ch
        else:
            digits += ch
    if not letters or not digits:
        raise ValueError(f"Некорректный id ячейки: {cell_id!r}")
    col = 0
    for ch in letters.upper():
        col = col * 26 + (ord(ch) - ord("A") + 1)
    col -= 1
    row = int(digits) - 1
    return col, row


def merge_adjacent_same_color(tasks: List[PaintTask]) -> List[PaintTask]:
    """
    Объединяет соседние по сетке ячейки одного цвета в одну задачу.

    Простая эвристика (не претендует на оптимальность): строит горизонтальные
    цепочки соседних ячеек одинакового цвета (одна строка, соседние столбцы)
    и объединяет их в одну задачу — cell первой ячейки цепочки, duration_s —
    сумма длительностей объединённых ячеек, note дополняется списком
    объединённых id. Контурные ячейки (у которых в note упомянут Академист)
    объединяются только с такими же контурными ячейками того же цвета,
    чтобы не потерять маркировку контура.

    Возвращает список PaintTask не длиннее исходного.
    """
    if not tasks:
        return []

    # Группируем по (row, color, is_outline), сортируем по col, чтобы находить
    # горизонтальные цепочки соседних ячеек.
    by_row_color: Dict[Tuple[int, str, bool], List[Tuple[int, PaintTask]]] = {}
    for t in tasks:
        col, row = _cell_id_to_col_row_local(t.cell)
        is_outline = "Академист" in t.note or "контурн" in t.note
        key = (row, t.color, is_outline)
        by_row_color.setdefault(key, []).append((col, t))

    merged: List[PaintTask] = []
    for (row, color, is_outline), items in by_row_color.items():
        items.sort(key=lambda pair: pair[0])
        chain: List[Tuple[int, PaintTask]] = []
        for col, task in items:
            if chain and col == chain[-1][0] + 1:
                chain.append((col, task))
            else:
                if chain:
                    merged.append(_merge_chain(chain))
                chain = [(col, task)]
        if chain:
            merged.append(_merge_chain(chain))

    # Гарантия контракта: не больше исходного числа задач.
    if len(merged) > len(tasks):
        return tasks
    return merged


def _merge_chain(chain: List[Tuple[int, "PaintTask"]]) -> PaintTask:
    """Сворачивает цепочку соседних одноцветных ячеек в одну PaintTask."""
    first_task = chain[0][1]
    if len(chain) == 1:
        return first_task

    total_duration = sum(t.duration_s * t.passes for _, t in chain)
    merged_cells = ",".join(t.cell for _, t in chain)
    return PaintTask(
        cell=first_task.cell,
        color=first_task.color,
        duration_s=total_duration,
        passes=1,
        priority=first_task.priority,
        note=f"{first_task.note}; объединено с [{merged_cells}]",
    )


# --------------------------------------------------------------------------
# Демонстрация
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from vision.prompt_to_bitmap import generate_bitmap, quantize_to_palette

    PALETTE = {
        "black": (20, 20, 20),
        "red": (210, 30, 30),
        "blue": (30, 60, 210),
        "yellow": (230, 200, 30),
    }

    demo_prompt = "нарисуй солнце над горами"
    bmp = generate_bitmap(demo_prompt, 20, 20)
    quant = quantize_to_palette(bmp, PALETTE)

    tasks = bitmap_to_tasks(quant, base_duration_s=1.2)
    print(f"Промпт: {demo_prompt!r}")
    print(f"Задач до объединения: {len(tasks)}")

    merged = merge_adjacent_same_color(tasks)
    print(f"Задач после объединения: {len(merged)}")
    print()
    for t in merged[:15]:
        print(f"  {t.cell:>4}  цвет={t.color:<6}  duration={t.duration_s:.2f}s  passes={t.passes}  note={t.note}")
    if len(merged) > 15:
        print(f"  ... и ещё {len(merged) - 15} задач")
