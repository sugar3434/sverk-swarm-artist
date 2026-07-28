"""Bitmap to Plan Task Conversion and Merging module for Sverk PikoClaw Swarm.

Converts 2D color matrix into draft PaintTask objects and merges adjacent cells along rows
without exceeding maximum spray nozzle duration (<= 5.0s) and passes (<= 3).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from common.schema import COLORS, PaintTask

logger = logging.getLogger("vision.bitmap_to_plan")


def coords_to_cell_id(col_idx: int, row_idx: int) -> str:
    """Converts 0-indexed column and row numbers into cell ID (e.g. 0,0 -> 'A1', 1,2 -> 'B3')."""
    if col_idx < 0 or row_idx < 0:
        raise ValueError(f"Coordinate indices must be non-negative: col={col_idx}, row={row_idx}")
    
    alpha_part = ""
    c = col_idx
    while True:
        alpha_part = chr(ord("A") + (c % 26)) + alpha_part
        c = c // 26 - 1
        if c < 0:
            break
            
    num_part = str(row_idx + 1)
    return f"{alpha_part}{num_part}"


def _parse_col_row_from_cell(cell_id: str) -> Tuple[int, int]:
    clean = cell_id.strip().upper()
    idx = 0
    while idx < len(clean) and clean[idx].isalpha():
        idx += 1
    alpha = clean[:idx]
    num = clean[idx:]
    
    col_idx = 0
    for ch in alpha:
        col_idx = col_idx * 26 + (ord(ch) - ord("A"))
    row_idx = max(0, int(num) - 1) if num.isdigit() else 0
    return col_idx, row_idx


def merge_adjacent_cells(
    tasks: List[PaintTask],
    max_duration_s: float = 5.0,
    max_passes: int = 3,
) -> List[PaintTask]:
    """Merges adjacent same-color cells horizontally within row without exceeding duration or passes bounds."""
    if not tasks:
        return []

    merged: List[PaintTask] = []
    current: Optional[PaintTask] = None
    current_row: int = -1
    current_col: int = -1
    merged_cells: List[str] = []

    for task in tasks:
        col_idx, row_idx = _parse_col_row_from_cell(task.cell)

        if current is None:
            current = PaintTask(
                cell=task.cell,
                color=task.color,
                duration_s=task.duration_s,
                passes=task.passes,
                priority=task.priority,
                x=task.x,
                y=task.y,
                z=task.z,
                note=task.note,
            )
            current_row = row_idx
            current_col = col_idx
            merged_cells = [task.cell]
            continue

        same_color = (task.color == current.color)
        is_adjacent = (row_idx == current_row and col_idx == (current_col + 1))
        new_duration = float(current.duration_s + task.duration_s)
        duration_ok = (new_duration <= max_duration_s + 1e-6)
        passes_ok = (max(current.passes, task.passes) <= max_passes)

        if same_color and is_adjacent and duration_ok and passes_ok:
            current.duration_s = min(max_duration_s, float(new_duration))
            current.passes = max(current.passes, task.passes)
            current_col = col_idx
            merged_cells.append(task.cell)
            if len(merged_cells) > 1:
                current.note = f"Merged segment {current.color}: cells " + ", ".join(merged_cells)
        else:
            merged.append(current)
            current = PaintTask(
                cell=task.cell,
                color=task.color,
                duration_s=task.duration_s,
                passes=task.passes,
                priority=task.priority,
                x=task.x,
                y=task.y,
                z=task.z,
                note=task.note,
            )
            current_row = row_idx
            current_col = col_idx
            merged_cells = [task.cell]

    if current is not None:
        merged.append(current)

    logger.info("Merged %d initial cells into %d tasks.", len(tasks), len(merged))
    return merged


merge_adjacent_tasks = merge_adjacent_cells


def bitmap_to_tasks(
    bitmap: List[List[str]],
    merge_adjacent: bool = True,
    default_duration_s: float = 2.0,
    default_passes: int = 1,
    max_duration_s: float = 5.0,
    max_passes: int = 3,
) -> List[PaintTask]:
    """Converts 2D bitmap color matrix into draft PaintTask list."""
    raw_tasks: List[PaintTask] = []
    if not bitmap or not bitmap[0]:
        logger.warning("Empty bitmap matrix passed to bitmap_to_tasks.")
        return raw_tasks

    rows = len(bitmap)
    cols = len(bitmap[0])
    priority_counter = 0

    for r in range(rows):
        for c in range(cols):
            color = str(bitmap[r][c]).strip().lower()
            if not color or color in ("none", "empty", "white", "null"):
                continue
            
            if color not in COLORS:
                logger.warning("Color %r at (%d, %d) not in palette, defaulting to 'black'.", color, r, c)
                color = "black"

            cell_id = coords_to_cell_id(c, r)
            task = PaintTask(
                cell=cell_id,
                color=color,
                duration_s=float(default_duration_s),
                passes=int(default_passes),
                priority=priority_counter,
                note=f"Bitmap cell {cell_id} ({color})",
            )
            raw_tasks.append(task)
            priority_counter += 1

    if merge_adjacent:
        return merge_adjacent_cells(raw_tasks, max_duration_s=max_duration_s, max_passes=max_passes)

    return raw_tasks


bitmap_to_draft_tasks = bitmap_to_tasks
convert_bitmap_to_plan = bitmap_to_tasks
