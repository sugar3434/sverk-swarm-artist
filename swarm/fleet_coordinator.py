"""
FleetCoordinator — «Координатор роя»: исполняет расписание
(`swarm.conflict_scheduler.ScheduledTask`) на реальном времени, используя
`openclaw.middleware.OpenClaw` для каждой команды и `openclaw.safety.SafetyMonitor`
для проверок безопасности.

Отвечает за:
  - синхронный одновременный взлёт всех 4 дронов (Регламент, критерий очного
    этапа: «одновременный взлёт дронов с роевым взаимодействием», 20 баллов);
  - выполнение ScheduledTask по расписанию (каждый дрон — свой поток, старт
    строго по `start_offset_s` от начала выполнения — расписание уже
    бесконфликтно благодаря `conflict_scheduler.schedule()`);
  - непрерывный контроль лимита времени попытки и заряда (`SafetyMonitor`);
  - синхронную посадку по завершении/по сигналу отмены;
  - «мягкую деградацию»: сбой одного дрона (после исчерпания повторов внутри
    OpenClaw) не останавливает остальных — их задачи продолжаются, а
    аварийный дрон переводится в land()/kill().
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

from common.schema import ScheduledTask
from openclaw.commands import LandCommand, PaintCommand, TakeoffCommand
from openclaw.drone_link import DroneLink
from openclaw.middleware import OpenClaw
from openclaw.safety import SafetyMonitor, SafetyViolation

logger = logging.getLogger("swarm.fleet_coordinator")

# Период сторожевого потока: как часто проверяется лимит времени попытки
# (Регламент п. 2.2). 0.5с даёт гарантию, что прерывание произойдёт ДО
# истечения бюджета, а не через минуту после — независимо от того, сколько
# длится текущая задача покраски.
WATCHDOG_PERIOD_S = 0.5


@dataclass
class DroneRunReport:
    drone_id: str
    tasks_done: int = 0
    tasks_failed: int = 0
    aborted: bool = False
    abort_reason: str = ""


@dataclass
class MissionReport:
    per_drone: Dict[str, DroneRunReport] = field(default_factory=dict)
    killed: bool = False
    kill_reason: str = ""
    total_elapsed_s: float = 0.0
    skipped_unknown_drone: int = 0   # задачи расписания для дронов вне роя
    skipped_unsafe: int = 0          # задачи, отклонённые проверками безопасности


class FleetCoordinator:
    """Оркестратор попытки: взлёт -> расписание -> посадка."""

    def __init__(self, fleet: Dict[str, DroneLink], openclaw: OpenClaw,
                 safety: SafetyMonitor, takeoff_altitude_m: float = 1.5) -> None:
        self.fleet = fleet
        self.openclaw = openclaw
        self.safety = safety
        self.takeoff_altitude_m = takeoff_altitude_m
        self._abort_event = threading.Event()
        # Гонки: abort() может быть вызван одновременно из нескольких потоков
        # дронов И из сторожевого потока. Без блокировки KILL SWITCH уходил бы
        # в дроны параллельно несколько раз (гонка по ROS-сервисам), а причина
        # аварии терялась.
        self._abort_lock = threading.Lock()
        self._abort_reason = ""

    # -- публичное аварийное прерывание (пульт/KILL SWITCH) ------------

    def abort(self, reason: str) -> None:
        """Вызывается извне (напр. обработчик пульта/KILL SWITCH) — синхронно
        останавливает и сажает весь рой немедленно.

        Идемпотентна и потокобезопасна: повторные вызовы только логируются,
        реальный KILL SWITCH уходит в рой РОВНО ОДИН раз.
        """
        with self._abort_lock:
            first = not self._abort_event.is_set()
            self._abort_event.set()
            if first:
                self._abort_reason = reason
            else:
                logger.warning("abort(%s) проигнорирован: рой уже остановлен по причине %r",
                               reason, self._abort_reason)
                return
            failed = self.safety.kill_switch(list(self.fleet.values()), reason)
        if failed:
            logger.critical("abort(): дроны %s не отреагировали на программную остановку", failed)

    @property
    def abort_reason(self) -> str:
        return self._abort_reason

    def reset(self) -> None:
        """Сброс состояния перед НОВОЙ попыткой (иначе прошлый abort навсегда
        блокирует координатор — все задачи мгновенно «прерваны»)."""
        with self._abort_lock:
            self._abort_event.clear()
            self._abort_reason = ""

    # -- синхронный взлёт -----------------------------------------------

    def synchronized_takeoff(self) -> List[str]:
        """Взлёт всех дронов ОДНОВРЕМЕННО (параллельные потоки), возвращает
        список drone_id, которые НЕ взлетели (для критерия «по 5 баллов за
        дрон» — статус каждого дрона фиксируется отдельно)."""
        self.safety.check_altitude(self.takeoff_altitude_m)
        failed: List[str] = []
        lock = threading.Lock()

        def _one(drone_id: str) -> None:
            try:
                ok = self.openclaw.takeoff(TakeoffCommand(drone_id, self.takeoff_altitude_m))
            except Exception as exc:  # noqa: BLE001 — один сбойный дрон не должен
                # ронять поток и оставлять рой в неопределённом состоянии.
                logger.error("[%s] взлёт завершился ошибкой: %s", drone_id, exc)
                ok = False
            if not ok:
                with lock:
                    failed.append(drone_id)

        threads = [threading.Thread(target=_one, args=(d,)) for d in self.fleet]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return failed

    # -- выполнение расписания -------------------------------------------

    def run_schedule(self, scheduled: List[ScheduledTask]) -> MissionReport:
        report = MissionReport()
        for drone_id in self.fleet:
            report.per_drone[drone_id] = DroneRunReport(drone_id=drone_id)

        by_drone: Dict[str, List[ScheduledTask]] = {d: [] for d in self.fleet}
        for st in scheduled:
            if st.drone_id not in by_drone:
                # Раньше здесь был by_drone.setdefault(...), из-за чего для
                # незнакомого drone_id создавался поток, а первая же строка
                # `report.per_drone[drone_id]` бросала KeyError ВНУТРИ потока:
                # задачи молча терялись, отчёт выглядел «успешным».
                report.skipped_unknown_drone += 1
                logger.error("Задача %s назначена дрону %r, которого нет в рое — пропущена",
                             st.task.cell, st.drone_id)
                continue
            by_drone[st.drone_id].append(st)
        for lst in by_drone.values():
            lst.sort(key=lambda st: st.start_offset_s)

        mission_start = time.monotonic()
        threads = []
        counters_lock = threading.Lock()

        def _abort_all(reason: str) -> None:
            try:
                self.abort(reason)
            except Exception as exc:  # noqa: BLE001 — путь аварийной остановки не должен падать
                logger.critical("abort() сам завершился ошибкой: %s", exc)

        # --- сторожевой поток лимита времени (Регламент п. 2.2) -----------
        # Ключевое отличие от прежней версии: раньше check_time_budget()
        # вызывался ТОЛЬКО перед началом очередной задачи дрона. Если задача
        # (перелёт + распыление + повторы middleware) длилась минуту, лимит
        # обнаруживался уже ПОСЛЕ его истечения, а дрон с пустым остатком
        # задач не проверял бюджет вообще. Теперь бюджет опрашивается каждые
        # WATCHDOG_PERIOD_S секунд независимо от задач.
        watchdog_off = threading.Event()

        def _watchdog() -> None:
            while not self._abort_event.is_set() and not watchdog_off.is_set():
                try:
                    self.safety.check_time_budget()
                except SafetyViolation as exc:
                    logger.critical("Сторожевой таймер: %s", exc)
                    _abort_all(str(exc))
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.error("Сторожевой таймер: неожиданная ошибка проверки бюджета: %s", exc)
                watchdog_off.wait(WATCHDOG_PERIOD_S)

        def _wait_until(offset_s: float) -> bool:
            """Ждать наступления offset_s от старта миссии. False = прервано."""
            while True:
                if self._abort_event.is_set():
                    return False
                remaining = offset_s - (time.monotonic() - mission_start)
                if remaining <= 0.0:
                    return True
                # Ждём короткими интервалами через Event: реакция на abort
                # мгновенная (раньше был time.sleep до 5с «вслепую»).
                self._abort_event.wait(min(remaining, WATCHDOG_PERIOD_S))

        def _run_drone(drone_id: str, tasks: List[ScheduledTask]) -> None:
            rep = report.per_drone[drone_id]
            try:
                for st in tasks:
                    if self._abort_event.is_set():
                        rep.aborted = True
                        rep.abort_reason = self._abort_reason or "прервано (KILL SWITCH / abort())"
                        return

                    # Проверяем бюджет С УЧЁТОМ длительности самой задачи:
                    # начинать задачу, которая не успеет закончиться до
                    # дедлайна (с резервом на посадку), нельзя (п. 2.2).
                    task_duration = max(0.0, st.end_offset_s - st.start_offset_s)
                    try:
                        self.safety.check_time_budget(extra_s=task_duration)
                    except SafetyViolation as exc:
                        rep.aborted = True
                        rep.abort_reason = str(exc)
                        _abort_all(str(exc))
                        return

                    # Регламент п. 2.6.2: высота проверяется перед КАЖДОЙ
                    # навигационной командой, а не только при взлёте.
                    # Плюс защита от None-координат (задача без cell_to_world):
                    # раньше `st.task.x or 0.0` молча отправлял дрон в (0,0,0).
                    if st.task.x is None or st.task.y is None or st.task.z is None:
                        rep.tasks_failed += 1
                        with counters_lock:
                            report.skipped_unsafe += 1
                        logger.error("[%s] задача %s без мировых координат (x/y/z=None) — "
                                     "пропущена (иначе дрон улетел бы в (0,0,0))",
                                     drone_id, st.task.cell)
                        continue
                    try:
                        self.safety.check_altitude(st.task.z)
                    except SafetyViolation as exc:
                        rep.tasks_failed += 1
                        with counters_lock:
                            report.skipped_unsafe += 1
                        logger.critical("[%s] задача %s отклонена: %s", drone_id, st.task.cell, exc)
                        continue

                    # Ждём наступления запланированного времени старта задачи.
                    # start_offset_s = момент НАЧАЛА перелёта (см. docstring
                    # swarm/conflict_scheduler.py), расписание бесконфликтно.
                    if not _wait_until(st.start_offset_s):
                        rep.aborted = True
                        rep.abort_reason = self._abort_reason or "прервано во время ожидания"
                        return

                    cmd = PaintCommand(
                        drone_id=drone_id, cell=st.task.cell,
                        x=st.task.x, y=st.task.y, z=st.task.z,
                        duration_s=st.task.duration_s, passes=st.task.passes,
                    )
                    ok = self.openclaw.paint_zone(cmd)
                    if ok:
                        rep.tasks_done += 1
                    else:
                        rep.tasks_failed += 1
                        logger.warning("[%s] задача %s провалена — продолжаем со следующей "
                                        "(мягкая деградация, не роняем всю попытку)",
                                        drone_id, st.task.cell)
            except BaseException as exc:  # noqa: BLE001
                # Без этого любая неожиданная ошибка убивала поток дрона молча:
                # отчёт показывал 0 провалов, судья видел «зависший» дрон.
                rep.aborted = True
                rep.abort_reason = f"внутренняя ошибка потока дрона: {type(exc).__name__}: {exc}"
                logger.critical("[%s] поток дрона аварийно завершён: %s", drone_id, exc,
                                exc_info=True)

        watchdog = threading.Thread(target=_watchdog, name="safety-watchdog", daemon=True)
        watchdog.start()
        try:
            for drone_id, tasks in by_drone.items():
                t = threading.Thread(target=_run_drone, args=(drone_id, tasks),
                                     name=f"drone-{drone_id}")
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
        finally:
            # Останавливаем сторожевой поток: без этого он продолжал бы жить
            # после миссии и мог дёрнуть KILL SWITCH во время посадки.
            watchdog_off.set()
            watchdog.join(timeout=WATCHDOG_PERIOD_S * 4)

        report.killed = self._abort_event.is_set()
        report.kill_reason = self._abort_reason
        report.total_elapsed_s = time.monotonic() - mission_start
        return report

    # -- синхронная посадка ------------------------------------------------

    def synchronized_land(self) -> List[str]:
        failed: List[str] = []
        lock = threading.Lock()

        def _one(drone_id: str) -> None:
            try:
                ok = self.openclaw.land(LandCommand(drone_id))
            except Exception as exc:  # noqa: BLE001 — посадка обязательна (п. 2.2),
                # поэтому ошибку одного дрона только фиксируем.
                logger.error("[%s] посадка завершилась ошибкой: %s", drone_id, exc)
                ok = False
            if not ok:
                with lock:
                    failed.append(drone_id)

        threads = [threading.Thread(target=_one, args=(d,)) for d in self.fleet]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return failed

    def kill_all(self, reason: str) -> List[str]:
        """KILL SWITCH для всего роя (Регламент п. 2.6.11 / 2.8.5).

        Отличия от прежней версии:
          * СНАЧАЛА выставляется _abort_event — потоки дронов перестают
            отправлять новые команды (раньше они продолжали летать, пока
            kill_all последовательно опрашивал дронов с retry+sleep);
          * остановка идёт через SafetyMonitor.kill_switch: цикл гарантированно
            проходит ВСЕ дроны, есть резервный land(), и возвращается список
            тех, кого остановить не удалось (для честного отчёта судьям).
        """
        with self._abort_lock:
            if not self._abort_event.is_set():
                self._abort_reason = reason
            self._abort_event.set()
            failed = self.safety.kill_switch(list(self.fleet.values()), reason)
        # Дополнительно логируем событие в журнал OpenClaw — это и есть
        # доказательство срабатывания KILL SWITCH для демонстрации.
        for drone_id in self.fleet:
            if drone_id not in failed:
                self.openclaw.log_external_event("kill_switch_confirmed", drone_id, True,
                                                 detail=f"остановлен по причине: {reason}")
        if failed:
            logger.critical("Рой остановлен ЧАСТИЧНО: не отреагировали %s (%s)", failed, reason)
        else:
            logger.critical("Рой полностью остановлен: %s", reason)
        return failed
