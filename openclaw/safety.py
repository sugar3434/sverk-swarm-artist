"""
Safety — программные проверки безопасности, требуемые Регламентом
(разделы 2.6 «Технический регламент беспилотных аппаратов» и
2.8 «Общие правила безопасности»).

Это ДОПОЛНЕНИЕ к штатным механизмам sverk-ros2 (RC kill switch в
offboard_control, Failsafe при потере сигнала — п. 2.6.4/2.6.11), а не их
замена. См. docs/ARCHITECTURE_CONTRACT.md.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from openclaw.drone_link import DroneLink, DroneLinkError

logger = logging.getLogger("openclaw.safety")

MIN_BATTERY_PCT = 40.0        # Регламент п. 2.6.7: проверка заряда >= 40% перед попыткой
MAX_ALTITUDE_M = 4.0          # Регламент п. 2.6.2: высота полёта не более 4 м
ATTEMPT_BUDGET_S = 15 * 60.0  # Регламент п. 2.2: 15 минут на попытку (включая диалог агентов)

# Регламент п. 2.2 требует, чтобы к концу 15 минут дроны были УЖЕ посажены,
# а не только начали садиться. Поэтому монитор считает бюджет истекшим
# ЗАРАНЕЕ, оставляя резерв на синхронную посадку всего роя.
LANDING_RESERVE_S = 30.0


class SafetyViolation(RuntimeError):
    """Нарушение правил безопасности — попытка должна быть прервана/отклонена."""


@dataclass
class SafetyMonitor:
    """Централизованный монитор безопасности для всего роя.

    Используется координатором (`swarm/fleet_coordinator.py`) на трёх этапах:
      1. `preflight(fleet)` — перед допуском к попытке (заряд, связь).
      2. `check_altitude(z)` — перед каждой навигационной командой.
      3. `check_time_budget()` — и на каждой итерации основного цикла, и из
         сторожевого потока координатора, чтобы лимит гарантированно
         ловился ДО истечения 15 минут, а не после.
    Плюс `kill_switch(fleet, reason)` — немедленная остановка ВСЕХ дронов
    (программный дубль аппаратного KILL SWITCH, Регламент п. 2.6.11/2.8.5).
    """

    started_at: float = field(default_factory=time.monotonic)
    budget_s: float = ATTEMPT_BUDGET_S
    min_battery_pct: float = MIN_BATTERY_PCT
    max_altitude_m: float = MAX_ALTITUDE_M
    landing_reserve_s: float = LANDING_RESERVE_S
    on_violation: Optional[Callable[[str], None]] = None
    _kill_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _report(self, message: str) -> None:
        logger.error(message)
        if self.on_violation is not None:
            try:
                self.on_violation(message)
            except Exception:  # noqa: BLE001 — отчёт не должен ронять safety-путь
                pass

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_s(self) -> float:
        """Сколько времени осталось ДО жёсткого дедлайна попытки (без резерва)."""
        return max(0.0, self.budget_s - self.elapsed_s())

    def remaining_flight_s(self) -> float:
        """Время на ПОЛЕЗНУЮ работу (за вычетом резерва на посадку) —
        именно по этому значению должно планироваться расписание покраски."""
        return max(0.0, self.budget_s - self.landing_reserve_s - self.elapsed_s())

    def check_time_budget(self, extra_s: float = 0.0) -> None:
        """Регламент п. 2.2: 15 минут на всё, включая диалог агентов.

        `extra_s` — сколько секунд займёт действие, которое мы собираемся
        начать: если оно не успеет завершиться до дедлайна с учётом резерва
        на посадку — начинать его нельзя (иначе посадка выйдет за 15 минут).
        """
        if self.remaining_flight_s() <= extra_s:
            raise SafetyViolation(
                f"Истёк лимит попытки {self.budget_s:.0f}с с резервом на посадку "
                f"{self.landing_reserve_s:.0f}с (Регламент п. 2.2; прошло "
                f"{self.elapsed_s():.0f}с, требовалось ещё {extra_s:.0f}с) — "
                "миссия должна быть немедленно свёрнута (посадка)."
            )

    def check_altitude(self, z: float) -> None:
        """Регламент п. 2.6.2: высота полёта дронов не более 4 м.

        Отдельно отбрасываются None/NaN/inf и отрицательная высота (команда
        «лететь под землю» — такая же авария, как и превышение потолка).
        """
        if z is None or not math.isfinite(float(z)):
            raise SafetyViolation(
                f"Некорректная целевая высота z={z!r} — команда отклонена (п. 2.6.2)"
            )
        if z > self.max_altitude_m:
            raise SafetyViolation(
                f"Заданная высота {z:.2f}м превышает лимит {self.max_altitude_m:.2f}м "
                "(Регламент п. 2.6.2)"
            )
        if z < 0.0:
            raise SafetyViolation(
                f"Заданная высота {z:.2f}м отрицательна (полёт ниже грунта) — команда отклонена"
            )

    def preflight(self, fleet: List[DroneLink]) -> None:
        """Проверка перед допуском к попытке — заряд каждого дрона (п. 2.6.7)."""
        if not fleet:
            raise SafetyViolation(
                "Предполётная проверка: рой пуст (ни одного дрона не подключено) — "
                "попытка невозможна"
            )
        problems: List[str] = []
        for link in fleet:
            try:
                link.preflight_check(min_battery_pct=self.min_battery_pct)
            except DroneLinkError as exc:
                problems.append(str(exc))
            except Exception as exc:  # noqa: BLE001 — любой сбой проверки = не допуск к полёту
                problems.append(
                    f"[{getattr(link, 'drone_id', '?')}] предполётная проверка сорвалась: {exc!r}"
                )
        if problems:
            for p in problems:
                self._report(p)
            raise SafetyViolation(
                "Предполётная проверка не пройдена:\n" + "\n".join(problems)
            )

    def kill_switch(self, fleet: List[DroneLink], reason: str) -> List[str]:
        """Аварийная остановка ВСЕГО роя (Регламент п. 2.6.11, 2.8.3/2.8.6).

        Программный дубль аппаратного RC-перехвата offboard_control. Пытается
        остановить КАЖДЫЙ дрон независимо — ошибка на одном НЕ прерывает цикл
        и не мешает остановить остальных. Если emergency_stop не сработал
        (дрон уже отвалился по ошибке связи), делается вторая попытка через
        land() — резервный канал.

        Вызовы сериализованы блокировкой: координатор может дёрнуть abort()
        из нескольких потоков одновременно, а параллельные emergency_stop на
        одном и том же дроне — гонка по ROS-сервисам.

        Возвращает список drone_id, которые НЕ удалось остановить ни одним
        каналом (пустой список = весь рой остановлен).
        """
        failed: List[str] = []
        with self._kill_lock:
            self._report(f"KILL SWITCH активирован: {reason}")
            for link in fleet:
                drone_id = getattr(link, "drone_id", "?")
                try:
                    link.emergency_stop(land=True)
                    continue
                except Exception as exc:  # noqa: BLE001 — обязаны попытаться на каждом
                    logger.critical("[%s] emergency_stop не удался при KILL SWITCH: %s",
                                    drone_id, exc)
                try:  # последний шанс убрать дрон из воздуха
                    link.land()
                    logger.warning("[%s] KILL SWITCH: аварийная посадка выполнена "
                                   "резервным каналом land()", drone_id)
                except Exception as exc2:  # noqa: BLE001
                    logger.critical("[%s] резервный land() ТАКЖЕ не удался: %s", drone_id, exc2)
                    failed.append(drone_id)
        if failed:
            self._report(
                "KILL SWITCH: НЕ удалось программно остановить дроны: "
                + ", ".join(failed)
                + " — требуется аппаратный KILL SWITCH с пульта (п. 2.6.11)"
            )
        return failed
