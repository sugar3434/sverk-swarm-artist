"""
Тест моста OpenCLaw и SimDroneLink (полностью офлайн, без ROS/железа).
Запуск: python3 tests/test_openclaw_middleware.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.commands import KillCommand, LandCommand, NavigateCommand, PaintCommand, TakeoffCommand
from openclaw.drone_link import SimDroneLink
from openclaw.middleware import OpenClaw


def test_basic_sequence():
    link = SimDroneLink("drone_black", "black")
    link.connect()
    oc = OpenClaw({"drone_black": link}, log_path="logs/test_openclaw_events.jsonl")

    assert oc.takeoff(TakeoffCommand("drone_black", 1.5)) is True
    assert oc.navigate(NavigateCommand("drone_black", 0.5, 0.0, 1.5)) is True
    assert oc.paint_zone(PaintCommand("drone_black", "B3", 0.5, 0.0, 1.5, duration_s=1.0, passes=1)) is True
    status = oc.status("drone_black")
    assert status.armed is True
    assert oc.land(LandCommand("drone_black")) is True
    status = oc.status("drone_black")
    assert status.armed is False
    link.close()


def test_retry_and_recover_from_transient_failure():
    """Имитируем сбой связи ПОСЛЕ взлёта: retry должен попытаться, но так как
    сбой постоянный (fail_after_s уже наступил) — команда в итоге вернёт False,
    и это не должно бросить исключение наружу (мягкая деградация)."""
    link = SimDroneLink("drone_red", "red", fail_after_s=0.05)
    link.connect()
    oc = OpenClaw({"drone_red": link}, log_path="logs/test_openclaw_events.jsonl", max_retries=1)

    import time
    time.sleep(0.1)  # гарантируем, что fail_after_s уже наступил
    ok = oc.navigate(NavigateCommand("drone_red", 1.0, 1.0, 1.5))
    assert ok is False  # команда провалилась, но НЕ бросила исключение
    link.close()


def test_kill_command():
    link = SimDroneLink("drone_blue", "blue")
    link.connect()
    oc = OpenClaw({"drone_blue": link}, log_path="logs/test_openclaw_events.jsonl")
    oc.takeoff(TakeoffCommand("drone_blue", 1.5))
    assert oc.status("drone_blue").armed is True
    assert oc.kill(KillCommand("drone_blue", also_land=True)) is True
    assert oc.status("drone_blue").armed is False
    link.close()


if __name__ == "__main__":
    test_basic_sequence()
    test_retry_and_recover_from_transient_failure()
    test_kill_command()
    print("OK: все тесты openclaw middleware прошли")
