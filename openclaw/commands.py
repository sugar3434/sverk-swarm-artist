"""
Команды OpenCLaw — единый словарь высокоуровневых действий, которые
координатор роя (`swarm/fleet_coordinator.py`) отдаёт мосту `middleware.py`.

Соответствие Регламенту (п. 2.1): «принять высокоуровневую команду агента
(например, "закрасить зону B3 красным, 2 секунды распыления") и
транслировать её в последовательность низкоуровневых инструкций».
`PaintCommand` — это ровно такая высокоуровневая команда.
"""
from __future__ import annotations

from dataclasses import dataclass

# ЕДИНАЯ скорость перелёта между ячейками во время покраски, м/с.
# Критично, чтобы ОДНО и то же значение использовали И планировщик
# (`swarm.conflict_scheduler.schedule(drone_speed_mps=...)`), И исполнитель
# (`OpenClaw.paint_zone`): иначе реальные перелёты длиннее расчётных,
# рой отстаёт от расписания и гарантия бесконфликтности ломается
# (судейский критерий «синхронность и взаимодействие... отсутствие конфликтов»).
PAINT_TRAVEL_SPEED_MPS = 0.35


@dataclass
class TakeoffCommand:
    drone_id: str
    altitude_m: float = 1.5


@dataclass
class NavigateCommand:
    drone_id: str
    x: float
    y: float
    z: float
    speed_mps: float = 0.4


@dataclass
class PaintCommand:
    """«Закрасить зону {cell} цветом {color}, {duration_s} секунд распыления»."""

    drone_id: str
    cell: str
    x: float
    y: float
    z: float
    duration_s: float
    passes: int = 1
    speed_mps: float = PAINT_TRAVEL_SPEED_MPS


@dataclass
class LandCommand:
    drone_id: str


@dataclass
class KillCommand:
    drone_id: str
    also_land: bool = True


@dataclass
class StatusQuery:
    drone_id: str
