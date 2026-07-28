"""Комплексные юнит-тесты подсистемы связи, координации и безопасности роя (Сверх v2).

Проверяемые аспекты:
1. Преобразование координат сетки CanvasGrid и ограничение по высоте <= 4.0 м.
2. Предполётная проверка заряда батареи (отказ при < 40.0%).
3. Контроль высоты полёта во время манёвров (отказ при > 4.0 м).
4. Воспроизведение симулированных траекторий полёта и работа LocalDroneLink (API sverk_interfaces).
5. Последовательность событий и управление сервоприводом покрасочной форсунки.
6. Использование Middleware для контроля латентности и устойчивости к сбоям.
7. Исполнение структурированных команд LLM, включая уступку воздушного пространства ("yield_wait").
8. Отключение и запрет офлайн-генераторов при пустых LLM-планах.
9. Экстренное отключение всего роя (kill switch / emergency_stop_all).
"""
from __future__ import annotations

import time
import pytest

from common.schema import FlightCommand, PaintTask, Plan
from openclaw.drone_link import DroneStatus, DroneTimeoutError, LocalDroneLink, SimulatedDroneLink
from openclaw.middleware import DroneMiddleware
from openclaw.safety import SafetyMonitor, SafetyViolationError
from swarm.canvas_grid import CanvasGrid
from swarm.fleet_coordinator import FleetCoordinator


def test_grid_conversion_and_altitude_limits() -> None:
    """Тестирование корректного расчёта координат холста и блокировки превышения высоты 4.0м."""
    # Горизонтальный холст на высоте 1.0м
    grid_xy = CanvasGrid(cols=4, rows=4, width_m=4.0, height_m=4.0, origin_x=0.0, origin_y=0.0, origin_z=1.0, orientation="horizontal")
    x, y, z = grid_xy.cell_to_world("B3")
    # col B -> индекс 1 -> x = (1 + 0.5) * 1.0 = 1.5
    # row 3 -> индекс 2 -> y = (2 + 0.5) * 1.0 = 2.5
    assert x == pytest.approx(1.5)
    assert y == pytest.approx(2.5)
    assert z == pytest.approx(1.0)

    # Вертикальный холст в плоскости XZ (где строки изменяются по высоте Z)
    grid_xz = CanvasGrid(cols=4, rows=4, width_m=4.0, height_m=6.0, origin_x=0.0, origin_y=0.0, origin_z=1.0, orientation="vertical")
    # Ячейка A1 (нижняя строка row=0): z = 1.0 + (0.5) * 1.5 = 1.75 м (<= 4.0 м, норма)
    x1, y1, z1 = grid_xz.cell_to_world("A1")
    assert z1 <= 4.0

    # Ячейка A4 (верхняя строка row=3): z = 1.0 + (3.5) * 1.5 = 6.25 м (> 4.0 м, должно быть отклонено!)
    with pytest.raises(SafetyViolationError) as exc_info:
        grid_xz.cell_to_world("A4")
    assert "превышает" in str(exc_info.value).lower() or "4.0" in str(exc_info.value)

    # Проверка отклонения сетки с начальной высотой > 4.0 м при инициализации
    with pytest.raises(SafetyViolationError):
        CanvasGrid(cols=2, rows=2, width_m=2.0, height_m=2.0, origin_z=4.5)

    # Проверка устойчивости к сложным суффиксным обозначениям от LLM
    xb, yb, zb = grid_xy.cell_to_world("B1_accent")
    assert xb == pytest.approx(1.5) and yb == pytest.approx(0.5)



def test_preflight_battery_verification() -> None:
    """Тест проверки уровня заряда перед полётом (отказ при заряде < 40%)."""
    monitor = SafetyMonitor()

    # Дрон с зарядом ниже 40.0% должен отклоняться
    low_battery_drone = SimulatedDroneLink("drone_red", initial_battery_pct=35.0)
    middleware = DroneMiddleware(low_battery_drone, safety_monitor=monitor, drone_id="drone_red")

    with pytest.raises(SafetyViolationError) as exc_info:
        middleware.takeoff(z=2.0)
    assert "40.0%" in str(exc_info.value) or "заряд" in str(exc_info.value).lower()

    # Дрон с достаточным зарядом (>= 40%) проходит проверку успешно
    good_drone = SimulatedDroneLink("drone_red", initial_battery_pct=45.0)
    middleware_good = DroneMiddleware(good_drone, safety_monitor=monitor, drone_id="drone_red")
    assert middleware_good.takeoff(z=2.0) is True


def test_altitude_checks_in_maneuvers() -> None:
    """Тест проверки высоты во время выполнения полётных инструкций (блокировка при z > 4.0 м)."""
    drone = SimulatedDroneLink("drone_blue", initial_battery_pct=90.0)
    middleware = DroneMiddleware(drone, drone_id="drone_blue")

    # Взлёт на допустимую высоту 3.5 м
    assert middleware.takeoff(z=3.5) is True

    # Попытка навигации на высоту 4.5 м должна завершиться отказом безопасности
    with pytest.raises(SafetyViolationError):
        middleware.navigate_wait(x=1.0, y=1.0, z=4.5)

    # Попытка повторного взлёта на высоту > 4.0 м должна отклониться
    with pytest.raises(SafetyViolationError):
        middleware.takeoff(z=5.0)


def test_simulated_trajectory_and_latency_logging() -> None:
    """Проверка выполнения траекторий полёта, записи истории и замеров латентности."""
    sim_link = SimulatedDroneLink("drone_yellow", initial_battery_pct=80.0)
    middleware = DroneMiddleware(sim_link, drone_id="drone_yellow")

    assert middleware.connect() is True
    assert middleware.takeoff(z=2.0, speed=1.5) is True
    assert middleware.navigate_wait(x=3.0, y=4.0, z=2.0, speed=2.0) is True
    assert middleware.land() is True

    # Проверка телеметрии и истории
    telemetry = middleware.get_telemetry()
    assert telemetry.x == pytest.approx(3.0)
    assert telemetry.y == pytest.approx(4.0)
    assert telemetry.z == pytest.approx(0.0)  # После посадки z=0
    assert len(sim_link.trajectory_history) == 3

    # Верификация логов латентности middleware
    assert len(middleware.latency_log) >= 4  # connect, takeoff, navigate_wait, land
    for entry in middleware.latency_log:
        assert "action" in entry
        assert "latency_s" in entry
        assert entry["latency_s"] >= 0.0


def test_local_drone_link_api_compliance() -> None:
    """Тест совместимости LocalDroneLink со спецификацией API sverk_interfaces."""
    local_link = LocalDroneLink(node_name="test_local_drone")
    assert local_link.connect() is True

    # Проверка методов навигации и получения статуса/телеметрии
    assert local_link.takeoff(z=2.0, speed=1.0) is True
    assert local_link.navigate_wait(x=1.0, y=2.0, z=3.0, yaw=45.0, tolerance=0.05) is True

    status = local_link.get_status()
    assert status.armed is True
    assert getattr(status, "battery_pct", 0) >= 40.0

    telemetry = local_link.get_telemetry()
    assert getattr(telemetry, "x", 0.0) == pytest.approx(1.0)

    # Работа форсунки и посадка
    assert local_link.paint_zone(duration_s=0.1, passes=1, angle_deg=30.0) is True
    assert local_link.land() is True
    local_link.close()


def test_servo_paint_sequencing() -> None:
    """Тестирование корректной последовательности событий работы сервопривода (enable -> angle -> center -> disable)."""
    sim_link = SimulatedDroneLink("drone_black", initial_battery_pct=95.0)
    middleware = DroneMiddleware(sim_link, drone_id="drone_black")

    middleware.takeoff(z=2.0)
    middleware.paint_zone(duration_s=0.2, passes=2, angle_deg=60.0)

    # Проверка журнала событий сервопривода
    events = [e["event"] for e in sim_link.servo_events]
    expected = [
        "servo_enable",
        "servo_set_angle",
        "servo_center",
        "servo_set_angle",
        "servo_center",
        "servo_disable",
    ]
    assert events == expected

    # Проверка, что после покраски клапан закрыт и питание отключено
    assert sim_link.servo_enabled is False
    assert sim_link.servo_angle == 0.0
    assert len(sim_link.paint_history) == 1


def test_fleet_coordinator_llm_execution_and_yield_wait() -> None:
    """Тест исполнения плана от LLM-координатора с задачами покраски и командой уступки 'yield_wait'."""
    drone_red = SimulatedDroneLink("drone_red", initial_battery_pct=90.0)
    drone_blue = SimulatedDroneLink("drone_blue", initial_battery_pct=90.0)
    grid = CanvasGrid(cols=3, rows=3, width_m=3.0, height_m=3.0, origin_x=0.0, origin_y=0.0, origin_z=2.0)

    coordinator = FleetCoordinator(fleet={"red": drone_red, "blue": drone_blue}, grid=grid)

    plan = Plan(
        prompt="Сложный манёвр роя",
        cells=[
            PaintTask(cell="A1", color="red", duration_s=0.1, passes=1, priority=1),
        ],
        flight_commands=[
            # Синий дрон уступает воздушное пространство красному дрону
            FlightCommand(drone_id="drone_blue", action="takeoff", z=2.0, speed_mps=1.0),
            FlightCommand(
                drone_id="drone_blue",
                action="yield_wait",
                duration_s=0.2,
                note="Зависание на месте: уступка пространства красному дрону",
            ),
            FlightCommand(drone_id="drone_blue", action="land"),
        ],
    )

    result = coordinator.execute_plan(plan)
    assert result["status"] == "success"
    assert result["executed_paint_tasks"] == 1
    assert result["executed_llm_commands"] == 3

    # Проверка наличия записи об уступке воздушного пространства в журнале исполнения
    actions_logged = [entry["action"] for entry in coordinator.execution_log]
    assert "yield_wait" in actions_logged
    assert "paint_zone" in actions_logged


def test_offline_fallback_rejection() -> None:
    """Проверка полного запрета офлайн-генерации правил при получении пустого плана от LLM."""
    coordinator = FleetCoordinator()
    empty_plan = Plan(prompt="Пустой запрос от пользователя")

    with pytest.raises(RuntimeError) as exc_info:
        coordinator.execute_plan(empty_plan)
    assert "заказ" in str(exc_info.value).lower() or "офлайн" in str(exc_info.value).lower() or "запрещ" in str(exc_info.value).lower()


def test_emergency_kill_switch() -> None:
    """Тест экстренной остановки моторов (kill switch / emergency_kill) всего роя дронов."""
    drone_black = SimulatedDroneLink("drone_black", initial_battery_pct=90.0)
    drone_yellow = SimulatedDroneLink("drone_yellow", initial_battery_pct=90.0)
    coordinator = FleetCoordinator(fleet={"black": drone_black, "yellow": drone_yellow})

    # Армирование и взлёт
    coordinator.get_drone("drone_black").takeoff(z=3.0)
    coordinator.get_drone("drone_yellow").takeoff(z=2.5)
    assert drone_black.get_status().armed is True

    # Активация экстренной остановки
    coordinator.emergency_stop_all()

    # Проверка полного отключения моторов и сброса флагов арминга
    assert drone_black.get_status().armed is False
    assert drone_black.is_killed is True
    assert drone_black.get_telemetry().z == 0.0

    assert drone_yellow.get_status().armed is False
    assert drone_yellow.is_killed is True

    # Попытка выполнения команд после kill должна блокироваться
    with pytest.raises(RuntimeError):
        drone_black.takeoff(z=2.0)
