"""Unit tests for FleetCoordinator, persona speed calculations, and yield_wait airspace hold."""
from __future__ import annotations

import pytest

from common.schema import FlightCommand, PaintTask, Plan, PERSONA_SPEEDS
from openclaw.drone_link import SimulatedDroneLink
from openclaw.safety import SafetyViolationError
from swarm.canvas_grid import CanvasGrid
from swarm.fleet_coordinator import FleetCoordinator


def test_persona_speed_mapping_and_travel_distance() -> None:
    """Tests persona velocity values and Euclidean distance calculation in ARUCO map frame."""
    coordinator = FleetCoordinator()
    
    assert PERSONA_SPEEDS["black"] == pytest.approx(0.9)
    assert PERSONA_SPEEDS["red"] == pytest.approx(1.8)
    assert PERSONA_SPEEDS["blue"] == pytest.approx(1.2)
    assert PERSONA_SPEEDS["yellow"] == pytest.approx(0.6)

    p1 = (0.0, 0.0, 1.5)
    p2 = (3.0, 4.0, 1.5)
    dist = coordinator.calculate_travel_distance(p1, p2)
    assert dist == pytest.approx(5.0)

    # Transit time for Red (speed 1.8 m/s): 5.0 / 1.8 + 0.5 = 3.277s
    t_red = coordinator.calculate_transit_time(dist, "red")
    assert t_red == pytest.approx(5.0 / 1.8 + 0.5)

    # Transit time for Yellow (speed 0.6 m/s): 5.0 / 0.6 + 0.5 = 8.833s
    t_yellow = coordinator.calculate_transit_time(dist, "yellow")
    assert t_yellow == pytest.approx(5.0 / 0.6 + 0.5)

    # Test drone ID lookup ("drone_black", "drone_red", etc.)
    t_drone_black = coordinator.calculate_transit_time(dist, "drone_black")
    assert t_drone_black == pytest.approx(5.0 / 0.9 + 0.5)


def test_priority_hierarchy_task_sorting() -> None:
    """Tests that schedule_plan prioritizes tasks using PRIORITY_HIERARCHY when priorities are equal."""
    grid = CanvasGrid(cols=3, rows=3, width_m=3.0, height_m=3.0, origin_z=2.0)
    coordinator = FleetCoordinator(grid=grid)
    plan = Plan(
        prompt="Priority test",
        cells=[
            PaintTask("C3", "yellow", priority=0),
            PaintTask("A1", "black", priority=0),
        ],
    )
    scheduled = coordinator.schedule_plan(plan)
    assert len(scheduled) == 2
    assert scheduled[0].drone_id == "drone_black"
    assert scheduled[1].drone_id == "drone_yellow"


def test_yield_wait_airspace_hold_execution() -> None:
    """Tests yield_wait airspace hold execution and position hold logging in ARUCO frame."""
    drone_blue = SimulatedDroneLink("drone_blue", initial_battery_pct=90.0)
    drone_red = SimulatedDroneLink("drone_red", initial_battery_pct=90.0)
    
    grid = CanvasGrid(cols=3, rows=3, width_m=3.0, height_m=3.0, origin_z=2.0)
    coordinator = FleetCoordinator(fleet={"blue": drone_blue, "red": drone_red}, grid=grid)

    cmd_yield = FlightCommand(
        drone_id="drone_blue",
        action="yield_wait",
        x=1.5,
        y=1.5,
        z=2.0,
        duration_s=0.2,
        note="Yielding airspace hold for priority red drone",
    )

    res = coordinator.execute_command(cmd_yield)
    assert res is True

    # Verify yield_wait event was logged in execution_log
    log_actions = [e["action"] for e in coordinator.execution_log]
    assert "yield_wait" in log_actions
    yield_entry = next(e for e in coordinator.execution_log if e["action"] == "yield_wait")
    assert yield_entry["drone_id"] == "drone_blue"
    assert yield_entry["duration_s"] == 0.2


def test_fleet_coordinator_preflight_battery_check() -> None:
    """Tests that FleetCoordinator enforces battery check (>= 40.0%) during plan execution."""
    low_battery_drone = SimulatedDroneLink("drone_black", initial_battery_pct=30.0)
    grid = CanvasGrid(cols=2, rows=2, width_m=2.0, height_m=2.0, origin_z=2.0)
    coordinator = FleetCoordinator(fleet={"black": low_battery_drone}, grid=grid)

    plan = Plan(
        prompt="Test preflight battery check",
        cells=[PaintTask("A1", "black", duration_s=1.0)],
    )

    with pytest.raises(SafetyViolationError) as exc_info:
        coordinator.execute_plan(plan)
    assert "40.0%" in str(exc_info.value) or "battery" in str(exc_info.value).lower() or "заряд" in str(exc_info.value).lower()


def test_fleet_coordinator_emergency_kill_all() -> None:
    """Tests emergency kill switch across all fleet drones."""
    d1 = SimulatedDroneLink("drone_black", initial_battery_pct=90.0)
    d2 = SimulatedDroneLink("drone_red", initial_battery_pct=90.0)
    coordinator = FleetCoordinator(fleet={"black": d1, "red": d2})

    d1.takeoff(z=2.0)
    d2.takeoff(z=2.0)
    assert d1.get_status().armed is True
    assert d2.get_status().armed is True

    coordinator.emergency_stop_all()

    assert d1.get_status().armed is False
    assert d1.is_killed is True
    assert d2.get_status().armed is False
    assert d2.is_killed is True
