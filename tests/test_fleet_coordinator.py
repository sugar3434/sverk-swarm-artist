"""
Интеграционный тест координатора роя (SimDroneLink x4, без ROS/железа).
Проверяет: синхронный взлёт всех 4 «дронов», выполнение бесконфликтного
расписания, синхронную посадку, и аварийную остановку по SafetyMonitor.
Запуск: python3 tests/test_fleet_coordinator.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.schema import PaintTask, ScheduledTask
from openclaw.drone_link import SimDroneLink
from openclaw.middleware import OpenClaw
from openclaw.safety import SafetyMonitor
from swarm.fleet_coordinator import FleetCoordinator

COLOR_TO_DRONE = {"black": "drone_black", "red": "drone_red", "blue": "drone_blue", "yellow": "drone_yellow"}


def make_fleet():
    fleet = {}
    for color, drone_id in COLOR_TO_DRONE.items():
        link = SimDroneLink(drone_id, color, battery_pct=90.0)
        link.connect()
        fleet[drone_id] = link
    return fleet


def test_full_mission_happy_path():
    fleet = make_fleet()
    safety = SafetyMonitor(budget_s=120.0)
    safety.preflight(list(fleet.values()))  # заряд 90% > 40% -> не должно бросить

    oc = OpenClaw(fleet, log_path="logs/test_fleet_events.jsonl")
    coord = FleetCoordinator(fleet, oc, safety, takeoff_altitude_m=1.5)

    failed_takeoff = coord.synchronized_takeoff()
    assert failed_takeoff == [], f"взлёт не удался у: {failed_takeoff}"
    for drone_id in fleet:
        assert oc.status(drone_id).armed is True

    # Мировые координаты ОБЯЗАТЕЛЬНЫ: координатор отклоняет задачи с x/y/z=None
    # (раньше они молча превращались в полёт в точку (0,0,0)).
    scheduled = [
        ScheduledTask(PaintTask(cell="A1", color="black", duration_s=0.05, x=0.2, y=1.5, z=0.6),
                      "drone_black", 0.0, 0.05),
        ScheduledTask(PaintTask(cell="B2", color="red", duration_s=0.05, x=0.6, y=1.5, z=1.0),
                      "drone_red", 0.0, 0.05),
        ScheduledTask(PaintTask(cell="C3", color="blue", duration_s=0.05, x=1.0, y=1.5, z=1.4),
                      "drone_blue", 0.0, 0.05),
        ScheduledTask(PaintTask(cell="D4", color="yellow", duration_s=0.05, x=1.4, y=1.5, z=1.8),
                      "drone_yellow", 0.0, 0.05),
        ScheduledTask(PaintTask(cell="A2", color="black", duration_s=0.05, x=0.2, y=1.5, z=1.0),
                      "drone_black", 0.1, 0.15),
    ]
    report = coord.run_schedule(scheduled)
    assert report.killed is False
    total_done = sum(r.tasks_done for r in report.per_drone.values())
    total_failed = sum(r.tasks_failed for r in report.per_drone.values())
    assert total_done == 5, f"ожидали 5 успешных задач, получили {total_done} (failed={total_failed})"
    assert report.per_drone["drone_black"].tasks_done == 2

    failed_land = coord.synchronized_land()
    assert failed_land == []
    for drone_id in fleet:
        assert oc.status(drone_id).armed is False
        fleet[drone_id].close()


def test_time_budget_triggers_abort():
    fleet = make_fleet()
    # Бюджет времени практически нулевой -> первая же проверка бросит SafetyViolation
    safety = SafetyMonitor(budget_s=0.0)
    oc = OpenClaw(fleet, log_path="logs/test_fleet_events.jsonl")
    coord = FleetCoordinator(fleet, oc, safety, takeoff_altitude_m=1.5)
    coord.synchronized_takeoff()

    scheduled = [
        ScheduledTask(PaintTask(cell="A1", color="black", duration_s=0.05, x=0.2, y=1.5, z=0.6),
                      "drone_black", 0.0, 0.05),
    ]
    report = coord.run_schedule(scheduled)
    assert report.killed is True
    assert report.per_drone["drone_black"].aborted is True
    for drone_id in fleet:
        fleet[drone_id].close()


def test_low_battery_rejected_at_preflight():
    fleet = make_fleet()
    fleet["drone_yellow"].battery_pct = 10.0  # ниже порога 40% (Регламент п. 2.6.7)
    safety = SafetyMonitor(budget_s=120.0)
    try:
        safety.preflight(list(fleet.values()))
        raised = False
    except Exception:
        raised = True
    assert raised, "preflight должен отклонить попытку при заряде ниже 40%"
    for drone_id in fleet:
        fleet[drone_id].close()


if __name__ == "__main__":
    test_full_mission_happy_path()
    test_time_budget_triggers_abort()
    test_low_battery_rejected_at_preflight()
    print("OK: все тесты fleet_coordinator прошли")
