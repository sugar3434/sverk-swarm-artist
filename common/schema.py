"""
Общие структуры данных проекта «Рой дронов-художников».

Единый источник истины для всех модулей (vision, agents, swarm, openclaw,
mission). См. docs/ARCHITECTURE_CONTRACT.md — не менять сигнатуры без
согласования, иначе сборка развалится на стыках.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

# Цвета красок ровно как в Регламенте (табл. 2.1): 4 дрона = 4 цвета.
COLORS = ("black", "red", "blue", "yellow")

# Соответствие цвет -> личность агента (Регламент, п. 2.1, таблица ролей).
COLOR_TO_AGENT = {
    "black": "Академист",
    "red": "Экспрессионист",
    "blue": "Минималист",
    "yellow": "Детализатор",
}


@dataclass
class PaintTask:
    """Одна покраска одной ячейки сетки полотна одним цветом."""

    cell: str                    # id ячейки, напр. "B3" (столбец-буква, строка-цифра)
    color: str                   # один из COLORS
    duration_s: float = 1.5      # длительность распыления форсунки, с
    passes: int = 1              # число проходов (Детализатор просит больше)
    priority: int = 0            # порядок после диалога агентов; меньше = раньше
    x: Optional[float] = None    # мировые координаты ячейки (м), заполняет CanvasGrid
    y: Optional[float] = None
    z: Optional[float] = None
    note: str = ""                # почему агент так решил (для лога/трансляции)

    def __post_init__(self) -> None:
        if self.color not in COLORS:
            raise ValueError(f"Неизвестный цвет '{self.color}', ожидался один из {COLORS}")
        if self.duration_s <= 0:
            raise ValueError("duration_s должен быть положительным")
        if self.passes < 1:
            raise ValueError("passes должен быть >= 1")


@dataclass
class DialogueTurn:
    """Одна реплика в публичном диалоге агентов."""

    agent: str
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class Plan:
    """Итог диалога агентов: финальный план покраски + стенограмма."""

    prompt: str
    cells: List[PaintTask]
    transcript: List[DialogueTurn]
    outline_color: str = "black"
    notes: str = ""

    def tasks_by_color(self, color: str) -> List[PaintTask]:
        return [t for t in self.cells if t.color == color]


@dataclass
class ScheduledTask:
    """Задача, назначенная конкретному дрону с временным окном (анти-коллизии)."""

    task: PaintTask
    drone_id: str
    start_offset_s: float
    end_offset_s: float
