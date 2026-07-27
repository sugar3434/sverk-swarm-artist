"""
OpenClaw — сам «программный мост» (middleware), п. 2.1 Регламента:
принимает высокоуровневые команды агентов/координатора и транслирует их в
вызовы `DroneLink` (навигация + сервопривод клапана краски + ESP-NOW/анти-
коллизии обеспечены на уровне `swarm/fleet_coordinator.py`, см. там).

Судейский критерий «качество интеграции (надёжность, latency, обработка
ошибок)» реализован здесь тремя механизмами:
  1. Каждая команда логируется в JSONL (`logs/openclaw_events.jsonl`) с
     латентностью выполнения — доказательная база для судей.
  2. Одна автоматическая попытка повтора при сбое связи (`DroneLinkError`).
  3. Ошибка ОДНОЙ команды не останавливает всю попытку — задача помечается
     failed и координатор продолжает с следующей (мисcия не должна
     «зависать» из-за одного сбойного дрона).
  4. Регламентный «предохранитель высоты» (п. 2.6.2, не более 4 м) проверяется
     ЗДЕСЬ, перед КАЖДОЙ командой, у которой есть целевая z (takeoff, navigate,
     paint_zone), а не только при взлёте в координаторе: middleware — последняя
     точка, через которую проходят ВСЕ команды к дрону, поэтому именно она
     обязана быть «шлюзом безопасности».
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from openclaw.commands import KillCommand, LandCommand, NavigateCommand, PaintCommand, TakeoffCommand
from openclaw.drone_link import DroneLink, DroneLinkError
from openclaw.safety import MAX_ALTITUDE_M, SafetyMonitor, SafetyViolation

logger = logging.getLogger("openclaw.middleware")


class OpenClaw:
    """Мост между высокоуровневыми командами и роем `DroneLink`."""

    def __init__(self, fleet: Dict[str, DroneLink], log_path: str = "logs/openclaw_events.jsonl",
                 max_retries: int = 1, safety: Optional[SafetyMonitor] = None,
                 max_altitude_m: float = MAX_ALTITUDE_M) -> None:
        self.fleet = fleet  # drone_id -> DroneLink
        self.max_retries = max_retries
        # SafetyMonitor опционален (тесты/демо создают OpenClaw без него), но
        # ограничение высоты действует ВСЕГДА: без монитора берётся MAX_ALTITUDE_M.
        self.safety = safety
        self.max_altitude_m = safety.max_altitude_m if safety is not None else max_altitude_m
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # -- внутренняя инфраструктура -----------------------------------

    def _log_event(self, event: str, drone_id: str, ok: bool, latency_ms: float,
                    detail: str = "", **extra) -> None:
        record = {
            "ts": time.time(),
            "event": event,
            "drone_id": drone_id,
            "ok": ok,
            "latency_ms": round(latency_ms, 1),
            "detail": detail,
            **extra,
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # noqa: BLE001 — логирование не должно ронять полёт
            logger.warning("Не удалось записать лог OpenCLaw: %s", exc)
        level = logging.INFO if ok else logging.ERROR
        logger.log(level, "%s[%s] %s (%.1fмс) %s", "OK " if ok else "FAIL ",
                   drone_id, event, latency_ms, detail)

    def log_external_event(self, event: str, drone_id: str, ok: bool, detail: str = "") -> None:
        """Публичная запись события в тот же журнал (нужна координатору, чтобы
        зафиксировать подтверждение KILL SWITCH — доказательство для судей)."""
        self._log_event(event, drone_id, ok, 0.0, detail=detail)

    def _get_link(self, event: str, drone_id: str) -> Optional[DroneLink]:
        """Возвращает канал дрона или None (с логом), НЕ бросая KeyError.

        Раньше `self.fleet[cmd.drone_id]` бросал KeyError для незнакомого
        drone_id (например, если дрон не подключился и рой оказался меньше
        расписания) — исключение вылетало в поток координатора и убивало его.
        """
        link = self.fleet.get(drone_id)
        if link is None:
            self._log_event(event, drone_id, False, 0.0,
                            detail="дрона нет в рое (не подключился?) — команда отклонена")
        return link

    def _check_altitude(self, event: str, drone_id: str, z: float) -> bool:
        """Регламент п. 2.6.2: высота не более 4 м — проверка ПЕРЕД командой."""
        try:
            if self.safety is not None:
                self.safety.check_altitude(z)
            else:
                if z is None or not (0.0 <= float(z) <= self.max_altitude_m):
                    raise SafetyViolation(
                        f"Заданная высота z={z!r} вне допустимого диапазона "
                        f"0..{self.max_altitude_m:.2f}м (Регламент п. 2.6.2)"
                    )
        except SafetyViolation as exc:
            self._log_event(event, drone_id, False, 0.0,
                            detail=f"ОТКЛОНЕНО по безопасности: {exc}", target_z=z)
            logger.critical("[%s] команда %s ОТКЛОНЕНА: %s", drone_id, event, exc)
            return False
        return True

    def _run(self, event: str, drone_id: str, fn, **extra) -> bool:
        """Выполнить одно действие с 1 повтором и логированием латентности."""
        attempts = self.max_retries + 1
        last_error: Optional[str] = None
        for attempt in range(1, attempts + 1):
            t0 = time.monotonic()
            try:
                fn()
                self._log_event(event, drone_id, True, (time.monotonic() - t0) * 1000,
                                 detail=f"попытка {attempt}/{attempts}", **extra)
                return True
            except SafetyViolation:
                raise  # нарушение регламента не «повторяем» — оно должно всплыть наверх
            except Exception as exc:  # noqa: BLE001 — ЛЮБОЙ сбой (включая AttributeError
                # из недоинициализированного ROS-клиента) должен приводить к мягкой
                # деградации, а не ронять поток дрона в координаторе.
                last_error = f"{type(exc).__name__}: {exc}"
                self._log_event(event, drone_id, False, (time.monotonic() - t0) * 1000,
                                 detail=f"попытка {attempt}/{attempts}: {last_error}", **extra)
                if attempt < attempts:
                    # Не тратим остаток лимита попытки (п. 2.2) на повторы, если
                    # времени уже нет — лучше сразу отдать управление на посадку.
                    if self.safety is not None and self.safety.remaining_flight_s() <= 0.0:
                        logger.error("[%s] %s: повтор отменён — лимит времени попытки исчерпан",
                                     drone_id, event)
                        break
                    time.sleep(0.5)
        logger.error("[%s] %s окончательно провалилась после %d попыток: %s",
                     drone_id, event, attempts, last_error)
        return False

    # -- высокоуровневые команды --------------------------------------

    def takeoff(self, cmd: TakeoffCommand) -> bool:
        link = self._get_link("takeoff", cmd.drone_id)
        if link is None or not self._check_altitude("takeoff", cmd.drone_id, cmd.altitude_m):
            return False
        return self._run("takeoff", cmd.drone_id, lambda: link.takeoff(cmd.altitude_m),
                          altitude_m=cmd.altitude_m)

    def navigate(self, cmd: NavigateCommand) -> bool:
        link = self._get_link("navigate", cmd.drone_id)
        if link is None or not self._check_altitude("navigate", cmd.drone_id, cmd.z):
            return False
        return self._run(
            "navigate", cmd.drone_id,
            lambda: link.navigate_wait(cmd.x, cmd.y, cmd.z, speed=cmd.speed_mps),
            target=(cmd.x, cmd.y, cmd.z),
        )

    def paint_zone(self, cmd: PaintCommand) -> bool:
        """«Закрасить зону {cell} цветом, {duration_s}с распыления» — Регламент п. 2.1."""
        link = self._get_link("paint_zone", cmd.drone_id)
        if link is None or not self._check_altitude("paint_zone", cmd.drone_id, cmd.z):
            return False

        def _do() -> None:
            # speed берём из команды (по умолчанию PAINT_TRAVEL_SPEED_MPS) — ровно
            # та же скорость, по которой планировщик считал время перелёта.
            link.navigate_wait(cmd.x, cmd.y, cmd.z, speed=cmd.speed_mps)
            link.paint(cmd.duration_s, passes=cmd.passes)

        return self._run("paint_zone", cmd.drone_id, _do, cell=cmd.cell,
                          duration_s=cmd.duration_s, passes=cmd.passes)

    def land(self, cmd: LandCommand) -> bool:
        link = self._get_link("land", cmd.drone_id)
        if link is None:
            return False
        return self._run("land", cmd.drone_id, lambda: link.land())

    def kill(self, cmd: KillCommand) -> bool:
        link = self._get_link("kill_switch", cmd.drone_id)
        if link is None:
            return False
        # Аварийная остановка не должна ждать 0.5с между повторами и вообще
        # повторяться средствами _run: у LiveDroneLink.emergency_stop уже есть
        # свои каналы (offboard -> fcu.kill_switch), а резервный land() делает
        # SafetyMonitor.kill_switch.
        t0 = time.monotonic()
        try:
            link.emergency_stop(land=cmd.also_land)
        except Exception as exc:  # noqa: BLE001
            self._log_event("kill_switch", cmd.drone_id, False, (time.monotonic() - t0) * 1000,
                             detail=f"{type(exc).__name__}: {exc}")
            return False
        self._log_event("kill_switch", cmd.drone_id, True, (time.monotonic() - t0) * 1000,
                         detail="программный KILL SWITCH выполнен")
        return True

    def status(self, drone_id: str):
        link = self.fleet.get(drone_id)
        if link is None:
            raise DroneLinkError(f"[{drone_id}] нет такого дрона в рое")
        return link.get_status()
