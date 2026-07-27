#!/usr/bin/env python3
"""
mission_runner.py — точка входа для одной зачётной попытки соревнования
«Рой дронов-художников: ИИ агент в искусстве».

Полный конвейер (Регламент, п. 2.1-2.2):
  1. Получаем художественный промпт (аргумент командной строки).
  2. vision:  промпт -> растр -> квантование по палитре -> черновой список
     задач покраски (PaintTask), включая детекцию контура.
  3. agents:  публичная мультиагентная дискуссия 4 личностей + координатор
     (транслируется в реальном времени, укладывается в общий тайм-бюджет).
  4. swarm:   перевод ячеек в мировые координаты (CanvasGrid) + бесконфликтное
     расписание (conflict_scheduler) с учётом анти-коллизий.
  5. openclaw + swarm.fleet_coordinator: синхронный взлёт, выполнение
     расписания в реальном времени, синхронная посадка. Постоянный контроль
     SafetyMonitor (заряд, высота, лимит 15 минут, KILL SWITCH).

Режимы запуска:
  --sim   (по умолчанию) — сухой прогон на SimDroneLink, без ROS/железа.
          Безопасно запускать где угодно для проверки логики миссии.
  --live  — реальный полёт: требует ROS 2 окружения с sverk-ros2
          (offboard_control/fmu_control/servo_control на борту каждого дрона).

Пример:
    python3 mission_runner.py --prompt "нарисуй сокола над горами" --sim
    python3 mission_runner.py --prompt "нарисуй сокола над горами" --live \
        --namespaces drone_black:/drone_black,drone_red:/drone_red,drone_blue:/drone_blue,drone_yellow:/drone_yellow
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Tuple

from agents.broadcast import Broadcaster
from agents.dialogue_engine import run_dialogue
from agents.llm_client import build_llm_client
from common.schema import PaintTask
from openclaw.commands import PAINT_TRAVEL_SPEED_MPS
from openclaw.drone_link import DroneLink, DroneLinkError, LiveDroneLink, SimDroneLink
from openclaw.middleware import OpenClaw
from openclaw.safety import ATTEMPT_BUDGET_S, MAX_ALTITUDE_M, MIN_BATTERY_PCT, SafetyMonitor, SafetyViolation
from swarm.canvas_grid import CanvasGrid
from swarm.conflict_scheduler import schedule, validate_no_conflicts
from swarm.fleet_coordinator import FleetCoordinator
from vision.bitmap_to_plan import bitmap_to_tasks, merge_adjacent_same_color
from vision.prompt_to_bitmap import PALETTE, generate_bitmap, quantize_to_palette

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mission_runner")

DEFAULT_COLOR_TO_DRONE = {
    "black": "drone_black",
    "red": "drone_red",
    "blue": "drone_blue",
    "yellow": "drone_yellow",
}


def build_fleet(sim: bool, namespaces: Dict[str, str]) -> Tuple[Dict[str, DroneLink], Dict[str, str]]:
    """Создаёт связку DroneLink на каждый из 4 дронов роя.

    В --sim режиме — SimDroneLink (без ROS/железа). В --live режиме —
    LiveDroneLink с namespace каждого дрона (см. docs/ARCHITECTURE_CONTRACT.md:
    один процесс/нода на дрон, у каждого свой offboard_control/fmu_control/
    servo_control в собственном ROS-namespace).

    Возвращает (fleet, failures), где failures = {drone_id: причина}.

    ВАЖНО (было багом): раньше исключение из `link.connect()` третьего дрона
    выбрасывалось наружу, и уже подключённые дроны 1-2 НЕ закрывались —
    их сервоприводы оставались включёнными, ROS-подписки висели, а вся
    попытка падала целиком, хотя по регламенту оценка идёт ПО КАЖДОМУ дрону
    отдельно. Теперь: сбойный дрон аккуратно закрывается и исключается из
    роя, миссия продолжается с оставшимися; исключение бросается только
    если не подключился НИ ОДИН дрон.
    """
    fleet: Dict[str, DroneLink] = {}
    failures: Dict[str, str] = {}
    for color, drone_id in DEFAULT_COLOR_TO_DRONE.items():
        link: DroneLink
        try:
            if sim:
                link = SimDroneLink(drone_id, color, battery_pct=92.0)
            else:
                ns = namespaces.get(drone_id, f"/{drone_id}")
                link = LiveDroneLink(
                    drone_id, color,
                    offboard_namespace=ns,
                    fcu_namespace=f"{ns}/fmu_control",
                    servo_enable=f"{ns}/servo_control/enable",
                    servo_angle_topic=f"{ns}/servo_control/target_angle_deg",
                    servo_center=f"{ns}/servo_control/center",
                )
        except Exception as exc:  # noqa: BLE001 — конструктор LiveDroneLink тоже может упасть
            failures[drone_id] = f"создание канала: {type(exc).__name__}: {exc}"
            logger.error("[%s] не создан канал управления: %s", drone_id, exc)
            continue
        try:
            link.connect()
        except Exception as exc:  # noqa: BLE001
            failures[drone_id] = f"connect(): {type(exc).__name__}: {exc}"
            logger.error("[%s] подключение не удалось: %s — дрон исключён из роя", drone_id, exc)
            try:
                link.close()  # освобождаем ресурсы/серво сбойного канала
            except Exception:  # noqa: BLE001
                pass
            continue
        fleet[drone_id] = link

    if not fleet:
        raise DroneLinkError(
            "Ни один дрон не подключился: " +
            "; ".join(f"{d}: {r}" for d, r in failures.items())
        )
    if failures:
        logger.warning("Рой неполный: подключено %d/%d дронов, не подключены: %s",
                       len(fleet), len(DEFAULT_COLOR_TO_DRONE), list(failures))
    return fleet, failures


def parse_namespaces(raw: str) -> Dict[str, str]:
    """Разбирает '--namespaces drone_black:/drone_black,drone_red:/drone_red,...'."""
    result: Dict[str, str] = {}
    if not raw:
        return result
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        drone_id, ns = pair.split(":", 1)
        result[drone_id.strip()] = ns.strip()
    return result


def run_mission(prompt: str, *, sim: bool, cols: int, rows: int, canvas_size_m: float,
                 namespaces: Dict[str, str], dialogue_rounds: int, log_dir: str) -> int:
    """Запускает полный цикл одной зачётной попытки. Возвращает код выхода
    (0 — успех/безопасное завершение, 1 — критическая ошибка на этапе,
    не совместимом с продолжением, напр. предполётная проверка).
    """
    safety = SafetyMonitor(budget_s=ATTEMPT_BUDGET_S, min_battery_pct=MIN_BATTERY_PCT,
                            max_altitude_m=MAX_ALTITUDE_M)
    broadcaster = Broadcaster(log_path=f"{log_dir}/dialogue_transcript.jsonl")
    fleet: Dict[str, DroneLink] = {}

    try:
        logger.info("=== ЭТАП 1/6: получен художественный промпт: %r ===", prompt)

        # Предполётная проверка ДО того, как диалог и генерация израсходуют
        # лимит попытки (Регламент п. 2.6.7: заряд >= 40% ПЕРЕД каждой
        # попыткой). Раньше рой поднимался только на 5-м этапе, и если заряд
        # был низкий, узнавали об этом, потратив большую часть 15 минут.
        logger.info("=== ЭТАП 2/6: подключение роя и предполётная проверка ===")
        fleet, connect_failures = build_fleet(sim=sim, namespaces=namespaces)
        safety.preflight(list(fleet.values()))
        # Расписание строим ТОЛЬКО для реально подключённых дронов: иначе
        # задачи «отсутствующего» цвета попадут в расписание и будут потеряны
        # внутри координатора.
        color_to_drone = {c: d for c, d in DEFAULT_COLOR_TO_DRONE.items() if d in fleet}
        active_colors = set(color_to_drone)
        if connect_failures:
            logger.warning("Цвета без дрона будут исключены из плана: %s",
                           sorted(set(DEFAULT_COLOR_TO_DRONE) - active_colors))

        logger.info("=== ЭТАП 3/6: генерация растра и чернового плана покраски ===")
        bitmap = generate_bitmap(prompt, cols=cols, rows=rows)
        quantized = quantize_to_palette(bitmap, PALETTE)
        draft_tasks: List[PaintTask] = bitmap_to_tasks(quantized)
        draft_tasks = merge_adjacent_same_color(draft_tasks)
        logger.info("Черновой план: %d задач покраски (после объединения соседних ячеек)",
                    len(draft_tasks))
        if not draft_tasks:
            logger.warning("Черновой план пуст (пустой промпт/весь фон) — рисовать нечего, "
                            "выполняем формальный синхронный взлёт-посадку без покраски.")

        logger.info("=== ЭТАП 4/6: публичный диалог агентов (бюджет %.0fс) ===",
                    min(safety.remaining_s(), 180.0))
        llm = build_llm_client()
        plan = run_dialogue(
            prompt=prompt, draft_tasks=draft_tasks, llm=llm, broadcaster=broadcaster,
            rounds=dialogue_rounds, time_budget_s=min(safety.remaining_s() * 0.5, 180.0),
        )
        logger.info("Диалог завершён: финальный план — %d ячеек, %d реплик в стенограмме",
                    len(plan.cells), len(plan.transcript))
        safety.check_time_budget()

        logger.info("=== ЭТАП 5/6: расчёт координат и бесконфликтного расписания ===")
        grid = CanvasGrid(
            width_m=canvas_size_m, height_m=canvas_size_m, cols=cols, rows=rows,
            origin=(0.0, 1.5, 0.4), wall_normal="y",
        )
        plan_cells = [t for t in plan.cells if t.color in active_colors]
        dropped_by_color = len(plan.cells) - len(plan_cells)
        if dropped_by_color:
            logger.warning("Исключено %d задач: для их цвета нет подключённого дрона",
                           dropped_by_color)

        for task in plan_cells:
            task.x, task.y, task.z = grid.cell_to_world(task.cell)

        # Регламент п. 2.6.2: ни одна точка плана не должна требовать высоты
        # выше 4 м. Проверяем ВЕСЬ план ДО вылета (при больших --canvas-size-m
        # верхние ряды полотна выходят за лимит), иначе нарушение обнаружилось
        # бы уже в воздухе.
        too_high = [(t.cell, t.z) for t in plan_cells
                    if t.z is None or t.z > safety.max_altitude_m or t.z < 0.0]
        if too_high:
            raise SafetyViolation(
                f"{len(too_high)} ячеек плана требуют недопустимой высоты "
                f"(лимит {safety.max_altitude_m:.1f}м, Регламент п. 2.6.2): "
                f"{too_high[:5]}{'...' if len(too_high) > 5 else ''}. "
                "Уменьшите --canvas-size-m или опустите origin полотна."
            )

        min_sep = grid.min_cell_separation_m()
        scheduled = schedule(plan_cells, grid, color_to_drone, min_separation_m=min_sep,
                             drone_speed_mps=PAINT_TRAVEL_SPEED_MPS)
        problems = validate_no_conflicts(scheduled, grid, min_sep)
        if problems:
            # Не должно происходить при корректной реализации schedule(), но
            # если случилось — не летим с потенциальным столкновением, это
            # критическая ошибка безопасности (Регламент п. 2.8.3).
            for p in problems:
                logger.error("Конфликт в расписании: %s", p)
            raise SafetyViolation(
                f"Обнаружено {len(problems)} неразрешённых конфликтов расписания — "
                "попытка отменена до вылета."
            )
        logger.info("Расписание бесконфликтно: %d задач, длительность ~%.1fс",
                    len(scheduled), max((s.end_offset_s for s in scheduled), default=0.0))

        # Регламент п. 2.2: посадка ОБЯЗАТЕЛЬНА до истечения 15 минут, поэтому
        # расписание обрезается по остатку времени с резервом на посадку.
        # Лучше не докрасить часть ячеек, чем быть остановленным судьёй в воздухе.
        flight_budget_s = safety.remaining_flight_s()
        fitting = [st for st in scheduled if st.end_offset_s <= flight_budget_s]
        if len(fitting) != len(scheduled):
            logger.warning("Обрезано %d задач из %d: не укладываются в остаток лимита "
                           "попытки %.0fс (с резервом %.0fс на посадку, п. 2.2)",
                           len(scheduled) - len(fitting), len(scheduled),
                           flight_budget_s, safety.landing_reserve_s)
            scheduled = fitting

        logger.info("=== ЭТАП 6/6: взлёт, выполнение расписания, посадка ===")
        openclaw = OpenClaw(fleet, log_path=f"{log_dir}/openclaw_events.jsonl", safety=safety)
        coord = FleetCoordinator(fleet, openclaw, safety, takeoff_altitude_m=1.5)

        failed_takeoff = coord.synchronized_takeoff()
        if failed_takeoff:
            logger.warning("Не взлетели: %s (получат 0 баллов за взлёт по каждому, "
                            "остальные продолжают попытку)", failed_takeoff)

        report = coord.run_schedule(scheduled)
        for drone_id, rep in report.per_drone.items():
            logger.info("[%s] выполнено=%d провалено=%d прервано=%s",
                        drone_id, rep.tasks_done, rep.tasks_failed, rep.aborted)

        if report.skipped_unknown_drone or report.skipped_unsafe:
            logger.warning("Пропущено задач: %d (нет такого дрона), %d (не прошли проверки)",
                           report.skipped_unknown_drone, report.skipped_unsafe)

        # Посадка выполняется ВСЕГДА, в том числе после аварийного прерывания
        # по времени (п. 2.2): kill_switch уже мог посадить дроны, повторная
        # land() безопасна и идемпотентна.
        failed_land = coord.synchronized_land()
        if failed_land:
            logger.critical("НЕ СЕЛИ штатно: %s — требуется вмешательство оператора "
                            "(KILL SWITCH на пульте, п. 2.6.11)", failed_land)
        logger.info("Попытка завершена: killed=%s (%s), полное время=%.1fс",
                    report.killed, report.kill_reason or "-", report.total_elapsed_s)
        return 0 if not report.killed and not failed_land else 1

    except SafetyViolation as exc:
        logger.critical("НАРУШЕНИЕ БЕЗОПАСНОСТИ: %s — миссия остановлена.", exc)
        if fleet:
            safety.kill_switch(list(fleet.values()), str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 — верхний уровень обязан не рухнуть молча
        logger.critical("Непредвиденная ошибка миссии: %s", exc, exc_info=True)
        if fleet:
            safety.kill_switch(list(fleet.values()), f"непредвиденная ошибка: {exc}")
        return 1
    finally:
        for link in fleet.values():
            try:
                link.close()
            except Exception:  # noqa: BLE001 — закрытие best-effort
                pass


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="Художественный промпт задания")
    parser.add_argument("--sim", action="store_true", default=True,
                         help="Сухой прогон без ROS/железа (по умолчанию)")
    parser.add_argument("--live", action="store_true", help="Реальный полёт (требует ROS 2 + sverk-ros2)")
    parser.add_argument("--cols", type=int, default=8, help="Число столбцов сетки полотна")
    parser.add_argument("--rows", type=int, default=8, help="Число строк сетки полотна")
    parser.add_argument("--canvas-size-m", type=float, default=2.4, help="Размер полотна, м (кв.)")
    parser.add_argument("--dialogue-rounds", type=int, default=2, help="Число раундов диалога агентов")
    parser.add_argument("--namespaces", default="", help="live-режим: drone_id:/ns,drone_id:/ns,...")
    parser.add_argument("--log-dir", default="logs", help="Каталог для логов JSONL")
    args = parser.parse_args(argv)

    sim = not args.live
    namespaces = parse_namespaces(args.namespaces)
    return run_mission(
        args.prompt, sim=sim, cols=args.cols, rows=args.rows, canvas_size_m=args.canvas_size_m,
        namespaces=namespaces, dialogue_rounds=args.dialogue_rounds, log_dir=args.log_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
