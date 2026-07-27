"""Временный стресс-тест conflict_scheduler на случайных наборах задач."""
import random
import sys

from common.schema import PaintTask
from swarm.canvas_grid import CanvasGrid
from swarm.conflict_scheduler import schedule, validate_no_conflicts, CONFLICT_DISTANCE_FACTOR

COLORS = ("black", "red", "blue", "yellow")
C2D = {c: f"drone_{c}" for c in COLORS}


def run_case(seed: int, n_tasks: int, verbose: bool = False) -> int:
    rng = random.Random(seed)
    grid = CanvasGrid(width_m=2.4, height_m=2.4, cols=10, rows=10,
                      origin=(0.0, 1.5, 0.5), wall_normal="y")
    tasks = []
    for _ in range(n_tasks):
        col = rng.randrange(10)
        row = rng.randrange(10)
        cell = grid.col_row_to_cell_id(col, row)
        tasks.append(PaintTask(cell=cell, color=rng.choice(COLORS),
                               duration_s=round(rng.uniform(0.5, 2.5), 2),
                               passes=rng.randint(1, 3),
                               priority=rng.randint(0, 3)))
    min_sep = grid.min_cell_separation_m()
    sch = schedule(tasks, grid, C2D, min_separation_m=min_sep,
                   drone_speed_mps=rng.choice([0.2, 0.35, 0.4, 1.0]))
    problems = validate_no_conflicts(sch, grid, min_sep)
    # инвариант: у каждого дрона свои задачи не пересекаются
    per = {}
    for s in sch:
        per.setdefault(s.drone_id, []).append(s)
    self_overlaps = []
    for d, lst in per.items():
        lst.sort(key=lambda s: s.start_offset_s)
        for a, b in zip(lst, lst[1:]):
            if b.start_offset_s < a.end_offset_s - 1e-9:
                self_overlaps.append((d, a.task.cell, b.task.cell,
                                      a.end_offset_s, b.start_offset_s))
    if len(sch) != len(tasks):
        print(f"seed={seed}: ПОТЕРЯНЫ ЗАДАЧИ {len(sch)} != {len(tasks)}")
    if problems:
        print(f"seed={seed} n={n_tasks}: КОНФЛИКТЫ {len(problems)}")
        for p in problems[:3]:
            print("   ", p)
    if self_overlaps:
        print(f"seed={seed} n={n_tasks}: САМОПЕРЕСЕЧЕНИЯ дрона {self_overlaps[:3]}")
    return len(problems) + len(self_overlaps) + (len(sch) != len(tasks))


bad = 0
for seed in range(400):
    n = random.Random(seed * 7).randint(30, 40)
    bad += run_case(seed, n)
print("всего проблемных случаев:", bad)
sys.exit(1 if bad else 0)
