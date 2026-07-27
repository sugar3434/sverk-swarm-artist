"""
Планировщик анти-коллизий для роя дронов-художников.

По регламенту («синхронность и взаимодействие дронов при рисовании,
отсутствие конфликтов» — 25 баллов) нужно гарантировать, что если задачи
ДВУХ РАЗНЫХ дронов затрагивают ячейки, расположенные ближе, чем
`min_separation_m * 1.5` (реальное расстояние между центрами ячеек, через
`CanvasGrid.cell_to_world`), то по времени они не должны выполняться
параллельно — их временные интервалы должны быть развёрстаны последовательно.

Задачи ОДНОГО дрона всегда последовательны (дрон физически один и не может
рисовать в двух местах одновременно).

Алгоритм (жадный, не претендует на оптимальность по общему времени, но
гарантирует отсутствие конфликтов):
1. Задачи группируются по drone_id через `color_to_drone`.
2. Внутри дрона задачи сортируются по (priority, cell) для стабильности.
3. Для каждой задачи дрона считается длительность = duration_s*passes +
   время перелёта от предыдущей точки этого же дрона (расстояние между
   центрами ячеек / drone_speed_mps).
4. Задачи назначаются по очереди («раунд-робин» по дронам в порядке их
   следующей свободной задачи), причём перед фиксацией времени старта
   задачи проверяется пересечение по времени с уже размещёнными задачами
   ДРУГИХ дронов, чьи ячейки находятся ближе порога. При конфликте старт
   сдвигается на конец конфликтующего интервала (и так до тех пор, пока
   конфликтов не останется — цикл, а не одно сравнение).

СЕМАНТИКА ВРЕМЕННОГО ОКНА (важно для исполнителя!):
  `start_offset_s` — момент, когда дрон НАЧИНАЕТ перелёт к ячейке
  (именно так его трактует `swarm/fleet_coordinator.py`: дождаться
  `start_offset_s` -> navigate_wait -> paint);
  `end_offset_s`   — момент окончания распыления, т.е.
  `start + travel_time + duration_s * passes`.
Таким образом окно [start, end) ПОКРЫВАЕТ и перелёт, и распыление, а значит
физическое присутствие дрона у ячейки — подмножество этого окна. Раньше
время перелёта прибавлялось ДО start (start = free_at + travel), из-за чего
(а) исполнитель систематически отставал от расписания на суммарное время
перелётов и гарантия бесконфликтности переставала действовать в реальном
времени, (б) при сдвиге старта из-за конфликта дрон прилетал к ячейке
раньше и ВИСЕЛ над ней, пока рядом красил другой дрон.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from common.schema import PaintTask, ScheduledTask

# Множитель порога безопасного расстояния (см. docs задачи): конфликтом
# считается близость строго ближе min_separation_m * CONFLICT_DISTANCE_FACTOR.
CONFLICT_DISTANCE_FACTOR = 1.5


def _distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def _group_tasks_by_drone(
    tasks: List[PaintTask], color_to_drone: Dict[str, str]
) -> Dict[str, List[PaintTask]]:
    by_drone: Dict[str, List[PaintTask]] = {}
    for task in tasks:
        if task.color not in color_to_drone:
            raise ValueError(
                f"Для цвета {task.color!r} не найден дрон в color_to_drone ({list(color_to_drone)})"
            )
        drone_id = color_to_drone[task.color]
        by_drone.setdefault(drone_id, []).append(task)
    return by_drone


def _sort_key(task: PaintTask) -> Tuple[int, str]:
    return (task.priority, task.cell)


def schedule(
    tasks: List[PaintTask],
    grid: "CanvasGrid",  # noqa: F821 — только для тайп-хинта, без импорта модуля во избежание циклов
    color_to_drone: Dict[str, str],
    min_separation_m: float,
    drone_speed_mps: float = 0.4,
    settle_s: float = 0.0,
) -> List[ScheduledTask]:
    """
    Строит бесконфликтное расписание задач покраски для роя дронов.

    Параметры:
        tasks:            список PaintTask (черновой план покраски).
        grid:             CanvasGrid — для перевода id ячейки в мировые координаты.
        color_to_drone:   отображение цвет -> id дрона (напр. {"black": "drone1", ...}).
        min_separation_m: минимальное безопасное расстояние между ячейками, м
                           (обычно CanvasGrid.min_cell_separation_m()).
        drone_speed_mps:  скорость перелёта дрона между точками, м/с — ДОЛЖНА
                           совпадать со скоростью, которой реально летит
                           исполнитель (openclaw.commands.PAINT_TRAVEL_SPEED_MPS),
                           иначе расписание расходится с реальностью.
        settle_s:         запас на стабилизацию/арм перед распылением, с.

    Возвращает:
        список ScheduledTask с непересекающимися по времени интервалами для
        любых двух задач разных дронов, чьи ячейки ближе min_separation_m*1.5.
        Окно [start_offset_s, end_offset_s) включает и перелёт, и распыление.
    """
    if not tasks:
        return []
    if drone_speed_mps <= 0:
        raise ValueError(f"drone_speed_mps должен быть положительным, получено {drone_speed_mps}")
    if min_separation_m < 0:
        raise ValueError("min_separation_m не может быть отрицательным")

    conflict_threshold = min_separation_m * CONFLICT_DISTANCE_FACTOR
    by_drone = _group_tasks_by_drone(tasks, color_to_drone)
    for drone_id in by_drone:
        by_drone[drone_id].sort(key=_sort_key)

    # Позиции центров ячеек всех задач заранее (избегаем повторных вызовов).
    # Ключ — id ЯЧЕЙКИ (а не id() объекта задачи: id() переиспользуется
    # интерпретатором и не устойчив, если задачи создаются/удаляются на ходу).
    world_pos: Dict[str, Tuple[float, float, float]] = {}
    for drone_tasks in by_drone.values():
        for task in drone_tasks:
            if task.cell not in world_pos:
                world_pos[task.cell] = grid.cell_to_world(task.cell)

    # Индекс текущей необработанной задачи для каждого дрона + время,
    # доступное для его следующей задачи (конец предыдущей задачи ЭТОГО дрона).
    next_idx: Dict[str, int] = {d: 0 for d in by_drone}
    drone_free_at: Dict[str, float] = {d: 0.0 for d in by_drone}
    drone_last_pos: Dict[str, Tuple[float, float, float]] = {d: None for d in by_drone}

    scheduled: List[ScheduledTask] = []
    # Для быстрой проверки конфликтов храним уже размещённые интервалы:
    # список (drone_id, cell_pos, start, end).
    placed: List[Tuple[str, Tuple[float, float, float], float, float]] = []

    drone_ids = list(by_drone.keys())
    remaining = sum(len(v) for v in by_drone.values())

    while remaining > 0:
        # Раунд-робин: берём дрона с наименьшим drone_free_at среди тех, у
        # кого остались задачи — это естественным образом продвигает во
        # времени того, кто освободился раньше, и распределяет работу.
        candidates = [d for d in drone_ids if next_idx[d] < len(by_drone[d])]
        drone_id = min(candidates, key=lambda d: drone_free_at[d])

        task = by_drone[drone_id][next_idx[drone_id]]
        pos = world_pos[task.cell]

        # Полная длительность окна = перелёт от предыдущей точки ЭТОГО дрона
        # + стабилизация + распыление. start_offset_s = момент НАЧАЛА перелёта
        # (ровно так его исполняет fleet_coordinator), поэтому перелёт входит
        # внутрь окна, а не прибавляется до него.
        prev_pos = drone_last_pos[drone_id]
        travel_time = 0.0
        if prev_pos is not None:
            travel_time = _distance(prev_pos, pos) / drone_speed_mps
        paint_duration = travel_time + settle_s + task.duration_s * task.passes

        start = drone_free_at[drone_id]

        # Разрешение конфликтов: пока есть пересечение по времени с задачей
        # ДРУГОГО дрона на близкой ячейке — сдвигаем старт на конец этого
        # конфликтующего интервала. Цикл, т.к. сдвиг может создать новый
        # конфликт с другой ранее размещённой задачей.
        moved = True
        while moved:
            moved = False
            end = start + paint_duration
            for other_drone, other_pos, other_start, other_end in placed:
                if other_drone == drone_id:
                    continue
                if _distance(pos, other_pos) >= conflict_threshold:
                    continue
                # Пересечение полуоткрытых интервалов [start, end) и [other_start, other_end).
                if start < other_end and other_start < end:
                    start = other_end
                    moved = True
                    end = start + paint_duration

        end = start + paint_duration
        scheduled.append(
            ScheduledTask(task=task, drone_id=drone_id, start_offset_s=start, end_offset_s=end)
        )
        placed.append((drone_id, pos, start, end))

        drone_free_at[drone_id] = end
        drone_last_pos[drone_id] = pos
        next_idx[drone_id] += 1
        remaining -= 1

    return scheduled


def validate_no_conflicts(
    scheduled: List[ScheduledTask],
    grid: "CanvasGrid",  # noqa: F821
    min_separation_m: float,
) -> List[str]:
    """
    Проверяет расписание на отсутствие конфликтов.

    Конфликт — пара ScheduledTask разных дронов, чьи ячейки ближе
    min_separation_m * 1.5, и их временные интервалы [start, end)
    пересекаются.

    Дополнительно проверяется физически обязательный инвариант: задачи ОДНОГО
    дрона не могут пересекаться по времени (дрон один и не умеет быть в двух
    местах сразу) — такое пересечение означает баг планировщика.

    Возвращает список текстовых описаний найденных конфликтов
    (пустой список = расписание бесконфликтно).
    """
    conflict_threshold = min_separation_m * CONFLICT_DISTANCE_FACTOR
    problems: List[str] = []

    positions = [grid.cell_to_world(s.task.cell) for s in scheduled]

    n = len(scheduled)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = scheduled[i], scheduled[j]
            if a.drone_id == b.drone_id:
                # Один дрон не может выполнять две задачи одновременно.
                if a.start_offset_s < b.end_offset_s - 1e-9 and b.start_offset_s < a.end_offset_s - 1e-9:
                    problems.append(
                        f"Самопересечение: дрон {a.drone_id} должен красить {a.task.cell} "
                        f"[{a.start_offset_s:.2f}, {a.end_offset_s:.2f}] и {b.task.cell} "
                        f"[{b.start_offset_s:.2f}, {b.end_offset_s:.2f}] одновременно"
                    )
                continue
            dist = _distance(positions[i], positions[j])
            if dist >= conflict_threshold:
                continue
            overlap = a.start_offset_s < b.end_offset_s and b.start_offset_s < a.end_offset_s
            if overlap:
                problems.append(
                    f"Конфликт: дрон {a.drone_id} (ячейка {a.task.cell}, "
                    f"[{a.start_offset_s:.2f}, {a.end_offset_s:.2f}]) и дрон {b.drone_id} "
                    f"(ячейка {b.task.cell}, [{b.start_offset_s:.2f}, {b.end_offset_s:.2f}]) "
                    f"— расстояние {dist:.3f} м < порога {conflict_threshold:.3f} м, "
                    f"интервалы времени пересекаются"
                )
    return problems


# --------------------------------------------------------------------------
# Демонстрация
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from swarm.canvas_grid import CanvasGrid

    grid = CanvasGrid(width_m=2.4, height_m=2.4, cols=6, rows=6, origin=(0.0, 1.5, 0.5), wall_normal="y")

    color_to_drone = {
        "black": "drone_academist",
        "red": "drone_expressionist",
        "blue": "drone_minimalist",
        "yellow": "drone_detailer",
    }

    # 8 задач разных цветов, часть — намеренно на соседних (близких) ячейках,
    # чтобы продемонстрировать разрешение конфликтов.
    demo_tasks = [
        PaintTask(cell="A1", color="red", duration_s=1.2),
        PaintTask(cell="A2", color="blue", duration_s=1.2),   # близко к A1 (red) — конфликт
        PaintTask(cell="C3", color="yellow", duration_s=1.5),
        PaintTask(cell="C4", color="black", duration_s=1.5),  # близко к C3 (yellow) — конфликт
        PaintTask(cell="F6", color="red", duration_s=1.0),
        PaintTask(cell="F5", color="blue", duration_s=1.0),   # близко к F6 (red) — конфликт
        PaintTask(cell="B1", color="yellow", duration_s=1.3),
        PaintTask(cell="D4", color="black", duration_s=1.1),
    ]

    min_sep = grid.min_cell_separation_m()
    result = schedule(demo_tasks, grid, color_to_drone, min_separation_m=min_sep, drone_speed_mps=0.4)

    print(f"min_cell_separation_m = {min_sep:.3f} м, порог конфликта = {min_sep * CONFLICT_DISTANCE_FACTOR:.3f} м")
    print()
    print("Итоговое расписание:")
    for s in sorted(result, key=lambda x: x.start_offset_s):
        print(
            f"  дрон={s.drone_id:<20} ячейка={s.task.cell:<3} цвет={s.task.color:<6} "
            f"[{s.start_offset_s:6.2f} ; {s.end_offset_s:6.2f}] с"
        )

    print()
    problems = validate_no_conflicts(result, grid, min_sep)
    if not problems:
        print("validate_no_conflicts: OK, конфликтов не найдено")
    else:
        print(f"validate_no_conflicts: найдено {len(problems)} конфликтов:")
        for p in problems:
            print(f"  - {p}")
