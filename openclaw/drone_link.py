"""
DroneLink — единый интерфейс «одна команда OpenCLaw = одно физическое действие
дрона», за которым скрыты детали sverk_interfaces (реальный полёт) или
имитация (сухой прогон без железа).

Почему так: судейский критерий «качество интеграции (надёжность, latency,
обработка ошибок)» требует, чтобы верхний уровень (middleware/координатор роя)
не падал из-за деталей ROS/PX4 и чтобы весь стек можно было протестировать
без физического дрона. `sverk_interfaces` (см.
https://github.com/sverk-tech/sverk-ros2/blob/main/sverk_interfaces/sverk_interfaces/__init__.py)
импортируется ЛЕНИВО — только внутри `LiveDroneLink`, поэтому весь остальной
код проекта (vision/agents/swarm/openclaw.middleware) можно запускать и
тестировать на ноутбуке разработчика без ROS 2 и без дронов.
"""
from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("openclaw.drone_link")


class DroneLinkError(RuntimeError):
    """Любая ошибка связи/выполнения команды дроном."""


@dataclass
class DroneStatus:
    """Снимок состояния дрона — общий для sim и live реализаций."""

    drone_id: str
    battery_pct: float
    armed: bool
    x: float
    y: float
    z: float
    last_abort_reason: str = ""


class DroneLink(abc.ABC):
    """Абстрактный канал управления одним дроном.

    Реализации: `SimDroneLink` (сухой прогон, без железа и без ROS) и
    `LiveDroneLink` (реальный полёт через sverk_interfaces / offboard_control
    / fmu_control / servo_control).
    """

    drone_id: str
    color: str

    @abc.abstractmethod
    def connect(self) -> None:
        """Установить соединение (создать ROS-ноду / клиентов сервисов)."""

    @abc.abstractmethod
    def preflight_check(self, min_battery_pct: float = 40.0) -> DroneStatus:
        """Проверка перед допуском к попытке (Регламент, п. 2.6.7: заряд >= 40%)."""

    @abc.abstractmethod
    def takeoff(self, z: float, timeout: float = 30.0) -> None:
        ...

    @abc.abstractmethod
    def navigate_wait(self, x: float, y: float, z: float, speed: float = 0.4,
                       timeout: float = 60.0) -> None:
        ...

    @abc.abstractmethod
    def paint(self, duration_s: float, passes: int = 1) -> None:
        """Открыть клапан форсунки на duration_s, повторить passes раз."""

    @abc.abstractmethod
    def land(self, timeout: float = 15.0) -> None:
        ...

    @abc.abstractmethod
    def emergency_stop(self, land: bool = True) -> None:
        """Программный KILL SWITCH (дублирует штатный RC-перехват offboard_control)."""

    @abc.abstractmethod
    def get_status(self) -> DroneStatus:
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# SimDroneLink — сухой прогон без железа и без ROS (для разработки/тестов/CI)
# ---------------------------------------------------------------------------


class SimDroneLink(DroneLink):
    """Имитация одного дрона: не требует ROS/железа, моделирует тайминги
    (перелёт по расстоянию/скорости, распыление по duration_s*passes) и
    расход заряда, чтобы весь стек (middleware, координатор, mission_runner)
    можно было прогнать и проверить целиком офлайн.
    """

    def __init__(self, drone_id: str, color: str, start_xyz: tuple = (0.0, 0.0, 0.0),
                 battery_pct: float = 95.0, speed_mps: float = 0.4,
                 fail_after_s: Optional[float] = None) -> None:
        self.drone_id = drone_id
        self.color = color
        self.x, self.y, self.z = start_xyz
        self.battery_pct = battery_pct
        self._speed = speed_mps
        self._armed = False
        self._t0 = time.monotonic()
        self._fail_after_s = fail_after_s  # для тестов отказоустойчивости
        self._last_abort_reason = ""
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        logger.info("[SIM %s] соединение установлено (имитация)", self.drone_id)

    def _elapsed(self) -> float:
        return time.monotonic() - self._t0

    def _maybe_fail(self) -> None:
        if self._fail_after_s is not None and self._elapsed() >= self._fail_after_s:
            self._last_abort_reason = "имитация сбоя связи (fail_after_s)"
            raise DroneLinkError(
                f"[SIM {self.drone_id}] имитированный сбой связи на {self._elapsed():.1f}с"
            )

    def preflight_check(self, min_battery_pct: float = 40.0) -> DroneStatus:
        if not self._connected:
            raise DroneLinkError(f"[SIM {self.drone_id}] preflight без connect()")
        if self.battery_pct < min_battery_pct:
            raise DroneLinkError(
                f"[SIM {self.drone_id}] заряд {self.battery_pct:.0f}% < {min_battery_pct:.0f}% "
                "(Регламент п. 2.6.7)"
            )
        return self.get_status()

    def takeoff(self, z: float, timeout: float = 30.0) -> None:
        self._maybe_fail()
        self._armed = True
        self.z = z
        self.battery_pct = max(0.0, self.battery_pct - 0.5)
        logger.info("[SIM %s] взлёт на z=%.2f", self.drone_id, z)

    def navigate_wait(self, x: float, y: float, z: float, speed: float = 0.4,
                       timeout: float = 60.0) -> None:
        self._maybe_fail()
        dist = ((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2) ** 0.5
        eta = dist / max(speed, 0.05)
        if eta > timeout:
            self._last_abort_reason = "Timeout"
            raise TimeoutError(
                f"[SIM {self.drone_id}] navigate_wait: расчётное время {eta:.1f}с > timeout {timeout:.1f}с"
            )
        self.x, self.y, self.z = x, y, z
        self.battery_pct = max(0.0, self.battery_pct - 0.05 * dist)

    def paint(self, duration_s: float, passes: int = 1) -> None:
        self._maybe_fail()
        total = duration_s * passes
        self.battery_pct = max(0.0, self.battery_pct - 0.02 * total)
        logger.info(
            "[SIM %s] распыление %.1fс x%d проходов в точке (%.2f,%.2f,%.2f)",
            self.drone_id, duration_s, passes, self.x, self.y, self.z,
        )

    def land(self, timeout: float = 15.0) -> None:
        self._armed = False
        self.z = 0.0
        logger.info("[SIM %s] посадка", self.drone_id)

    def emergency_stop(self, land: bool = True) -> None:
        self._last_abort_reason = "emergency_stop"
        logger.warning("[SIM %s] KILL SWITCH: emergency_stop(land=%s)", self.drone_id, land)
        if land:
            self.land()
        else:
            self._armed = False  # зависание на месте (сим упрощённо разоружает)

    def get_status(self) -> DroneStatus:
        return DroneStatus(
            drone_id=self.drone_id, battery_pct=self.battery_pct, armed=self._armed,
            x=self.x, y=self.y, z=self.z, last_abort_reason=self._last_abort_reason,
        )

    def close(self) -> None:
        self._connected = False
        logger.info("[SIM %s] соединение закрыто", self.drone_id)


# ---------------------------------------------------------------------------
# LiveDroneLink — реальный полёт через sverk_interfaces (см. sverk-ros2)
# ---------------------------------------------------------------------------


class LiveDroneLink(DroneLink):
    """Реальное управление дроном «Сверх» через sverk_interfaces.

    Один процесс/нода на дрон в мультидрон-конфигурации (см.
    docs/ARCHITECTURE_CONTRACT.md): каждый дрон запускает свой стек
    offboard_control/fmu_control/servo_control в собственном ROS-namespace
    (напр. `/drone_black`, `/drone_red`, ...), поэтому в конструктор передаём
    namespace-параметры один в один как принимает `sverk_interfaces.init(...)`.

    ВАЖНО про безопасность: у offboard_control уже есть штатный RC-перехват
    (kill switch с пульта, параметр `check_kill_switch`) — Регламент п. 2.6.11.
    `emergency_stop()` здесь — ДОПОЛНИТЕЛЬНЫЙ программный канал, не замена
    аппаратному.
    """

    def __init__(self, drone_id: str, color: str, *,
                 offboard_namespace: str = "", fcu_namespace: str = "/fmu_control",
                 servo_enable: str = "/servo_control/enable",
                 servo_angle_topic: str = "/servo_control/target_angle_deg",
                 servo_center: str = "/servo_control/center",
                 spray_open_deg: float = 90.0, spray_closed_deg: float = 0.0) -> None:
        self.drone_id = drone_id
        self.color = color
        self._ns_kwargs = dict(
            offboard_namespace=offboard_namespace,
            fcu_namespace=fcu_namespace,
            servo_enable=servo_enable,
            servo_angle_topic=servo_angle_topic,
            servo_center=servo_center,
        )
        self._spray_open_deg = spray_open_deg
        self._spray_closed_deg = spray_closed_deg
        self._drone = None  # sverk_interfaces.DroneInterfaces, создаётся в connect()

    def _require_link(self, action: str):
        """Гарантирует, что connect() был успешен.

        Без этой проверки любой вызов после провалившегося connect() (или
        после close()) падал с AttributeError: 'NoneType' — а этот тип НЕ ловится
        обработчиками (DroneLinkError, TimeoutError, RuntimeError) в middleware,
        т.е. ронял поток дрона в координаторе вместо мягкой деградации.
        """
        if self._drone is None:
            raise DroneLinkError(
                f"[{self.drone_id}] {action}: соединение не установлено "
                "(connect() не вызван, провалился или уже вызван close())"
            )
        return self._drone

    def connect(self) -> None:
        try:
            import sverk_interfaces  # импорт лениво: ROS не нужен вне LiveDroneLink
        except ImportError as exc:
            raise DroneLinkError(
                f"[{self.drone_id}] sverk_interfaces не установлен — LiveDroneLink "
                "требует запуска внутри ROS 2 окружения (см. sverk-ros2)."
            ) from exc
        try:
            self._drone = sverk_interfaces.init(
                Nodename=f"openclaw_{self.drone_id}", **self._ns_kwargs
            )
            self._drone.gpio.servo_enable()
            self._drone.gpio.servo_set_angle(self._spray_closed_deg)  # клапан закрыт (безопасно)
        except Exception as exc:  # noqa: BLE001 — любая ошибка связи -> наше исключение
            raise DroneLinkError(f"[{self.drone_id}] connect() не удался: {exc}") from exc

    def preflight_check(self, min_battery_pct: float = 40.0) -> DroneStatus:
        self._require_link("preflight_check")
        status = self.get_status()
        if status.battery_pct < min_battery_pct:
            raise DroneLinkError(
                f"[{self.drone_id}] заряд {status.battery_pct:.0f}% < {min_battery_pct:.0f}% "
                "(Регламент п. 2.6.7) — попытка отклонена"
            )
        return status

    def takeoff(self, z: float, timeout: float = 30.0) -> None:
        drone = self._require_link("takeoff")
        try:
            drone.control.navigate_wait(
                x=0.0, y=0.0, z=z, frame_id="body", auto_arm=True,
                timeout=timeout, tolerance=0.2,
            )
        except (TimeoutError, RuntimeError) as exc:
            raise DroneLinkError(f"[{self.drone_id}] взлёт не удался: {exc}") from exc

    def navigate_wait(self, x: float, y: float, z: float, speed: float = 0.4,
                       timeout: float = 60.0) -> None:
        drone = self._require_link("navigate_wait")
        try:
            drone.control.navigate_wait(
                x=x, y=y, z=z, frame_id="map", speed=speed, timeout=timeout, tolerance=0.15,
            )
        except (TimeoutError, RuntimeError) as exc:
            raise DroneLinkError(f"[{self.drone_id}] навигация к ({x:.2f},{y:.2f},{z:.2f}) не удалась: {exc}") from exc

    def paint(self, duration_s: float, passes: int = 1) -> None:
        drone = self._require_link("paint")
        total_passes = max(1, passes)
        try:
            for i in range(total_passes):
                drone.gpio.servo_set_angle(self._spray_open_deg)
                time.sleep(duration_s)
                drone.gpio.servo_set_angle(self._spray_closed_deg)
                if i < total_passes - 1:
                    time.sleep(0.2)  # пауза между проходами
        except Exception as exc:  # noqa: BLE001
            # КЛАПАН ОБЯЗАН быть закрыт даже если сбой случился посреди
            # распыления — иначе форсунка останется открытой в воздухе.
            try:
                drone.gpio.servo_set_angle(self._spray_closed_deg)
            except Exception:  # noqa: BLE001
                logger.critical("[%s] не удалось закрыть клапан после сбоя распыления",
                                self.drone_id)
            raise DroneLinkError(f"[{self.drone_id}] распыление не удалось: {exc}") from exc

    def land(self, timeout: float = 15.0) -> None:
        drone = self._require_link("land")
        # Закрытие клапана — best-effort И ОТДЕЛЬНО от посадки: раньше сбой
        # servo_set_angle перехватывался тем же except и land() ВООБЩЕ не вызывался —
        # дрон оставался в воздухе из-за неработающего сервопривода.
        try:
            drone.gpio.servo_set_angle(self._spray_closed_deg)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] не удалось закрыть клапан перед посадкой: %s", self.drone_id, exc)
        try:
            drone.control.land(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            raise DroneLinkError(f"[{self.drone_id}] посадка не удалась: {exc}") from exc

    def emergency_stop(self, land: bool = True) -> None:
        """Программный KILL SWITCH: 3 канала по очереди (п. 2.6.11/2.8.5).

        1) offboard_control.emergency_stop, 2) fcu.kill_switch(),
        3) если ОБА канала провалились — бросает DroneLinkError, чтобы
        вызывающая сторона (SafetyMonitor.kill_switch / OpenClaw.kill) ЗНАЛА
        об этом и попробовала резервный land(). Раньше метод всегда
        завершался «thinking»-тишиной, и журнал OpenCLaw писал OK для
        несработавшего KILL SWITCH — ложное доказательство для судей.
        """
        drone = self._require_link("emergency_stop")
        try:
            drone.gpio.servo_set_angle(self._spray_closed_deg)
        except Exception as exc:  # noqa: BLE001 — клапан важен, но остановка важнее
            logger.error("[%s] не удалось закрыть клапан при аварийной остановке: %s",
                          self.drone_id, exc)
        try:
            drone.control.emergency_stop(land=land)
            return
        except Exception as exc:  # noqa: BLE001 — при аварии обязаны попытаться и дальше
            logger.error("[%s] emergency_stop через offboard_control не удался: %s", self.drone_id, exc)
        try:
            drone.fcu.kill_switch()
        except Exception as exc2:  # noqa: BLE001
            logger.critical("[%s] kill_switch ТАКЖЕ не удался: %s", self.drone_id, exc2)
            raise DroneLinkError(
                f"[{self.drone_id}] НИ ОДИН программный канал аварийной остановки не сработал: {exc2}"
            ) from exc2

    def get_status(self) -> DroneStatus:
        drone = self._require_link("get_status")
        try:
            s = drone.control.get_status(timeout=2.0)
            # КОНТРАКТ API sverk_interfaces: get_telemetry(frame_id="map") — параметра
            # timeout у него НЕТ (было get_telemetry(timeout=2.0) -> TypeError на
            # реальном железе, из-за чего любой get_status()/preflight_check()
            # гарантированно падал, т.е. проверка заряда по п. 2.6.7 была невозможна).
            t = drone.control.get_telemetry(frame_id="map")
            battery = float(s.battery_pct) if s is not None else 0.0
            armed = bool(s.armed) if s is not None else False
            abort = s.last_abort_reason if s is not None else "нет ответа offboard_control"
            return DroneStatus(
                drone_id=self.drone_id, battery_pct=battery, armed=armed,
                x=t.x, y=t.y, z=t.z, last_abort_reason=abort,
            )
        except Exception as exc:  # noqa: BLE001
            raise DroneLinkError(f"[{self.drone_id}] get_status() не удался: {exc}") from exc

    def close(self) -> None:
        if self._drone is not None:
            drone, self._drone = self._drone, None
            try:
                drone.gpio.servo_set_angle(self._spray_closed_deg)
            except Exception:  # noqa: BLE001 — закрытие best-effort
                pass
            try:
                drone.gpio.servo_disable()
            except Exception:  # noqa: BLE001 — закрытие best-effort
                pass
            try:
                # Раньше исключение в drone.close() вылетало наружу и могло прервать
                # цикл закрытия остальных дронов в mission_runner.finally.
                drone.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] close() завершился с ошибкой: %s", self.drone_id, exc)
