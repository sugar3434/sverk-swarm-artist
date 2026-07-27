#!/usr/bin/env python3
"""
Регрессионные тесты ревизии A (безопасность и соответствие Регламенту).

Каждый тест соответствует РЕАЛЬНОМУ багу, найденному и исправленному в ходе
ревизии критического пути (openclaw/, swarm/, mission_runner.py).

Запускается и как pytest, и напрямую: python3 tests/test_review_a_safety.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.schema import PaintTask, ScheduledTask  # noqa: E402
from openclaw.commands import PAINT_TRAVEL_SPEED_MPS, PaintCommand, TakeoffCommand  # noqa: E402
from openclaw.drone_link import DroneLinkError, SimDroneLink  # noqa: E402
from openclaw.middleware import OpenClaw  # noqa: E402
from openclaw.safety import (  # noqa: E402
    LANDING_RESERVE_S,
    MAX_ALTITUDE_M,
    SafetyMonitor,
    SafetyViolation,
)
from swarm.canvas_grid import CanvasGrid  # noqa: E402
from swarm.conflict_scheduler import schedule, validate_no_conflicts  # noqa: E402
from swarm.fleet_coordinator import FleetCoordinator  # noqa: E402

LOG = "logs/test_review_a_events.jsonl"


class BrokenLink(SimDroneLink):
    """Дрон, который отказывается останавливаться (эмуляция отказа связи)."""

    def emergency_stop(self, land: bool = True) -> None:
        raise DroneLinkError("нет связи с дроном")

    def land(self) -> None:
        raise DroneLinkError("нет связи с дроном")


def _fleet(n=4, broken_first=False):
    ids = ["drone_black", "drone_red", "drone_blue", "drone_yellow"][:n]
    colors = ["black", "red", "blue", "yellow"][:n]
    fleet = {}
    for i, (d, c) in enumerate(zip(ids, colors)):
        cls = BrokenLink if (broken_first and i == 0) else SimDroneLink
        fleet[d] = cls(d, c, battery_pct=90.0)
        fleet[d].connect()
    return fleet


# ---------------------------------------------------------------- п. 2.2
def test_time_budget_reserves_landing():
    """check_time_budget обязан прерывать попытку ДО дедлайна, оставляя
    резерв на обязательную посадку (баг: раньше срабатывал ровно в 15:00,
    т.е. посадка гарантированно выходила за лимит)."""
    safety = SafetyMonitor(budget_s=LANDING_RESERVE_S + 5.0)
    safety.check_time_budget()  # 5с полётного времени ещё есть
    assert abs(safety.remaining_flight_s() - 5.0) < 1.0

    tight = SafetyMonitor(budget_s=LANDING_RESERVE_S)
    raised = False
    try:
        tight.check_time_budget()
    except SafetyViolation:
        raised = True
    assert raised, "бюджет без запаса на посадку должен приводить к SafetyViolation"

    # extra_s: задача, которая НЕ успеет закончиться, не должна начинаться
    safety2 = SafetyMonitor(budget_s=LANDING_RESERVE_S + 10.0)
    raised2 = False
    try:
        safety2.check_time_budget(extra_s=30.0)
    except SafetyViolation:
        raised2 = True
    assert raised2, "задача длиннее остатка времени не должна допускаться к старту"


# -------------------------------------------------------------- п. 2.6.2
def test_check_altitude_rejects_bad_values():
    safety = SafetyMonitor(budget_s=900.0)
    safety.check_altitude(MAX_ALTITUDE_M)  # ровно лимит — допустимо
    for bad in (MAX_ALTITUDE_M + 0.01, -0.5, None, float("nan"), float("inf")):
        raised = False
        try:
            safety.check_altitude(bad)
        except SafetyViolation:
            raised = True
        assert raised, f"высота {bad!r} должна быть отклонена"


def test_middleware_blocks_every_navigation_above_limit():
    """Лимит высоты проверяется перед КАЖДОЙ командой, а не только при взлёте."""
    fleet = _fleet()
    safety = SafetyMonitor(budget_s=900.0)
    oc = OpenClaw(fleet, log_path=LOG, safety=safety)
    assert oc.takeoff(TakeoffCommand("drone_black", 1.5)) is True

    cmd = PaintCommand("drone_black", "Z9", 0.5, 1.5, MAX_ALTITUDE_M + 1.0, duration_s=0.01)
    assert oc.paint_zone(cmd) is False, "покраска выше 4 м должна быть отклонена"
    # и дрон при этом НЕ должен был улететь наверх
    assert fleet["drone_black"].z <= MAX_ALTITUDE_M

    assert oc.takeoff(TakeoffCommand("drone_red", 10.0)) is False


def test_middleware_unknown_drone_returns_false():
    """Незнакомый drone_id раньше давал KeyError внутри потока координатора."""
    oc = OpenClaw(_fleet(), log_path=LOG, safety=SafetyMonitor(budget_s=900.0))
    assert oc.takeoff(TakeoffCommand("drone_ghost", 1.0)) is False
    assert oc.paint_zone(PaintCommand("drone_ghost", "A1", 0.0, 1.5, 1.0, duration_s=0.01)) is False


# ------------------------------------------------- п. 2.6.11 / 2.8.5 KILL
def test_kill_switch_stops_all_drones_even_if_one_fails():
    """Цикл KILL SWITCH не должен прерываться на первом исключении."""
    fleet = _fleet(broken_first=True)
    safety = SafetyMonitor(budget_s=900.0)
    failed = safety.kill_switch(list(fleet.values()), "тест KILL SWITCH")
    assert failed == ["drone_black"], f"ожидали список несработавших дронов, получили {failed}"
    for drone_id in ("drone_red", "drone_blue", "drone_yellow"):
        st = fleet[drone_id].get_status()
        assert st.armed is False, f"{drone_id} должен быть разоружён после KILL SWITCH"


def test_live_link_raises_when_not_connected():
    """LiveDroneLink без connect() должен бросать DroneLinkError, а не
    AttributeError: 'NoneType' object has no attribute ..."""
    from openclaw.drone_link import LiveDroneLink

    link = LiveDroneLink("drone_x", "black")
    for call in (lambda: link.takeoff(1.0), lambda: link.land(), lambda: link.get_status()):
        raised = False
        try:
            call()
        except DroneLinkError:
            raised = True
        except Exception as exc:  # noqa: BLE001
            raised = False
            assert False, f"ожидали DroneLinkError, получили {type(exc).__name__}: {exc}"
        assert raised


# ----------------------------------------------------- расписание/конфликты
def test_schedule_window_includes_travel():
    """Семантика окна: start_offset_s — начало ПЕРЕЛЁТА, а конец окна включает
    перелёт + распыление (баг: перелёт прибавлялся ДО start, рой отставал)."""
    grid = CanvasGrid(width_m=2.4, height_m=2.4, cols=4, rows=4,
                      origin=(0.0, 1.5, 0.4), wall_normal="y")
    tasks = [PaintTask(cell="A1", color="black", duration_s=1.0),
             PaintTask(cell="D4", color="black", duration_s=1.0)]
    sch = schedule(tasks, grid, {"black": "drone_black"}, min_separation_m=grid.min_cell_separation_m(),
                   drone_speed_mps=PAINT_TRAVEL_SPEED_MPS)
    assert len(sch) == 2
    first, second = sorted(sch, key=lambda s: s.start_offset_s)
    assert abs(first.start_offset_s) < 1e-9, "первая задача стартует в t=0"
    # вторая начинает перелёт сразу после окончания первой
    assert abs(second.start_offset_s - first.end_offset_s) < 1e-6
    # окно второй задачи длиннее её распыления — включает перелёт
    assert second.end_offset_s - second.start_offset_s > 1.0


def test_schedule_random_no_conflicts():
    """Свойство бесконфликтности на случайных наборах (30-40 задач, 10x10)."""
    import random

    grid = CanvasGrid(width_m=3.0, height_m=3.0, cols=10, rows=10,
                      origin=(0.0, 1.5, 0.4), wall_normal="y")
    min_sep = grid.min_cell_separation_m()
    colors = ["black", "red", "blue", "yellow"]
    mapping = {c: f"drone_{c}" for c in colors}
    cells = [f"{chr(ord('A') + c)}{r + 1}" for c in range(10) for r in range(10)]
    for seed in range(40):
        rnd = random.Random(seed)
        chosen = rnd.sample(cells, rnd.randint(30, 40))
        tasks = [PaintTask(cell=cell, color=rnd.choice(colors),
                           duration_s=round(rnd.uniform(0.5, 3.0), 2),
                           passes=rnd.choice([1, 1, 2]))
                 for cell in chosen]
        sch = schedule(tasks, grid, mapping, min_separation_m=min_sep,
                       drone_speed_mps=PAINT_TRAVEL_SPEED_MPS)
        assert len(sch) == len(tasks), f"seed={seed}: потеряны задачи"
        problems = validate_no_conflicts(sch, grid, min_sep)
        assert problems == [], f"seed={seed}: {problems[:3]}"


# ----------------------------------------------------------- координатор
def test_coordinator_watchdog_aborts_on_time_budget():
    """Сторожевой таймер должен прервать попытку, даже если дроны заняты
    длинной задачей и сами бюджет не проверяют (п. 2.2)."""
    fleet = _fleet()
    safety = SafetyMonitor(budget_s=LANDING_RESERVE_S + 0.4, landing_reserve_s=LANDING_RESERVE_S)
    oc = OpenClaw(fleet, log_path=LOG, safety=safety)
    coord = FleetCoordinator(fleet, oc, safety, takeoff_altitude_m=1.0)
    coord.synchronized_takeoff()
    sch = [ScheduledTask(PaintTask(cell="A1", color="black", duration_s=5.0, x=0.2, y=1.5, z=1.0),
                         "drone_black", 3.0, 8.0)]
    t0 = time.monotonic()
    report = coord.run_schedule(sch)
    assert report.killed is True, "истечение бюджета должно приводить к прерыванию"
    assert report.kill_reason, "причина прерывания должна быть зафиксирована в отчёте"
    assert time.monotonic() - t0 < 3.0, "прерывание должно быть быстрым, а не после задачи"


def test_coordinator_skips_tasks_of_unknown_drone_and_none_coords():
    fleet = _fleet(n=2)
    safety = SafetyMonitor(budget_s=900.0)
    oc = OpenClaw(fleet, log_path=LOG, safety=safety)
    coord = FleetCoordinator(fleet, oc, safety, takeoff_altitude_m=1.0)
    coord.synchronized_takeoff()
    sch = [
        ScheduledTask(PaintTask(cell="A1", color="black", duration_s=0.01, x=0.2, y=1.5, z=1.0),
                      "drone_black", 0.0, 0.01),
        ScheduledTask(PaintTask(cell="B1", color="black", duration_s=0.01), "drone_black", 0.0, 0.01),
        ScheduledTask(PaintTask(cell="C1", color="yellow", duration_s=0.01, x=0.2, y=1.5, z=1.0),
                      "drone_ghost", 0.0, 0.01),
    ]
    report = coord.run_schedule(sch)
    assert report.skipped_unknown_drone == 1
    assert report.skipped_unsafe == 1, "задача без мировых координат не должна выполняться"
    assert report.per_drone["drone_black"].tasks_done == 1


def test_abort_is_idempotent_and_records_reason():
    fleet = _fleet()
    safety = SafetyMonitor(budget_s=900.0)
    coord = FleetCoordinator(fleet, OpenClaw(fleet, log_path=LOG, safety=safety), safety)
    coord.abort("первая причина")
    coord.abort("вторая причина")
    assert coord.abort_reason == "первая причина"
    coord.reset()
    assert coord.abort_reason == ""


# --------------------------------------------------------------- build_fleet
def test_build_fleet_survives_partial_connect_failure():
    """Если connect() одного дрона упал, остальные должны продолжить попытку,
    а сбойный канал — быть закрыт (не оставлять включённое серво)."""
    import mission_runner

    closed = []
    original = mission_runner.SimDroneLink

    class Flaky(original):
        def connect(self):
            if self.drone_id == "drone_blue":
                raise DroneLinkError("эмуляция отказа ROS-сервиса")
            return super().connect()

        def close(self):
            closed.append(self.drone_id)
            return super().close()

    mission_runner.SimDroneLink = Flaky
    try:
        fleet, failures = mission_runner.build_fleet(sim=True, namespaces={})
    finally:
        mission_runner.SimDroneLink = original

    assert set(fleet) == {"drone_black", "drone_red", "drone_yellow"}
    assert "drone_blue" in failures
    assert "drone_blue" in closed, "сбойный канал должен быть закрыт"


def test_build_fleet_raises_when_nobody_connects():
    import mission_runner

    original = mission_runner.SimDroneLink

    class Dead(original):
        def connect(self):
            raise DroneLinkError("нет связи")

    mission_runner.SimDroneLink = Dead
    raised = False
    try:
        mission_runner.build_fleet(sim=True, namespaces={})
    except DroneLinkError:
        raised = True
    finally:
        mission_runner.SimDroneLink = original
    assert raised, "если не подключился ни один дрон — попытка невозможна"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("OK: все тесты ревизии A прошли")
