"""Комплексные юнит-тесты главного модуля исполнения миссий и зрения (Сверх v2).

Проверяемые направления:
1. Полное отклонение и блокировка любых попыток использования флага --offline.
2. Детерминированная работа модулей зрения: генерация растрового полотна (generate_bitmap),
   квантование цвета (quantize_to_palette) и оптимизирующее слияние задач (bitmap_to_tasks).
3. Полная симуляция сквозного цикла миссии роя (mission_runner) с участием LLM-агентов
   и верификации предполётной безопасности.
4. Безопасное прерывание миссии с кодом завершения 1 при сетевых сбоях (LLMConnectionError)
   без подмены решений фиктивными офлайн-заглушками.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from agents.llm_client import LLMClient, LLMConnectionError
from common.schema import COLORS, FlightCommand, PaintTask, Plan
from mission_runner import check_no_offline_flag, get_cli_parser, main, run_mission
from vision.bitmap_to_plan import bitmap_to_tasks, coords_to_cell_id, merge_adjacent_cells
from vision.prompt_to_bitmap import PALETTE, generate_bitmap, quantize_to_palette


# --- 1. Тесты блокировки офлайн-режима ---

def test_cli_rejects_offline_flag() -> None:
    """Проверка, что парсер и предварительный контроллер категорически отвергают флаг --offline."""
    with pytest.raises(SystemExit) as exc_info:
        check_no_offline_flag(["--offline"])
    assert exc_info.value.code == 1

    with pytest.raises(SystemExit) as exc_info:
        check_no_offline_flag(["--sim", "-o"])
    assert exc_info.value.code == 1

    # Проверка, что argparse не имеет аргумента --offline и падает с ошибкой
    parser = get_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--prompt", "Тестовое полотно", "--offline"])

    # Запуск main() с --offline возвращает код 1
    assert main(["--offline"]) == 1


# --- 2. Тесты модулей компьютерного зрения ---

def test_vision_color_quantize_and_palette() -> None:
    """Проверка цветовой квантификации (приведения к официальной палитре соревнований)."""
    assert quantize_to_palette("black") == "black"
    assert quantize_to_palette("RED") == "red"

    # Hex-код близкий к красному #E02020 -> red
    assert quantize_to_palette("#e02020") == "red"
    # Hex-код близкий к жёлтому #FFF010 -> yellow
    assert quantize_to_palette("#FFF010") == "yellow"
    # RGB кортеж близкий к синему (20, 40, 200) -> blue
    assert quantize_to_palette((20, 40, 200)) == "blue"


def test_vision_bitmap_determinism() -> None:
    """Проверка детерминированности генерации растрового полотна из промпта."""
    prompt = "Вечерний контур города на закате, огни и экспрессия"
    bitmap1 = generate_bitmap(prompt, cols=4, rows=3)
    bitmap2 = generate_bitmap(prompt, cols=4, rows=3)
    
    # Детерминированность: одинаковый промпт должен всегда давать абсолютно идентичный растр
    assert bitmap1 == bitmap2
    assert len(bitmap1) == 3 and len(bitmap1[0]) == 4
    for r in range(3):
        for c in range(4):
            assert bitmap1[r][c] in COLORS


def test_vision_bitmap_to_plan_merge() -> None:
    """Проверка объединения смежных одинаковых цветов без превышения лимитов (duration_s <= 5.0, passes <= 3)."""
    # Матрица, где строка 0 состоит целиком из 'black'
    bitmap = [
        ["black", "black", "black", "red"],
        ["blue", "yellow", "yellow", "yellow"],
    ]
    # При базовой длительности 2.0с и лимите 5.0с:
    # A1(2.0с) + B1(2.0с) -> 4.0с (сливаются)
    # + C1(2.0с) -> было бы 6.0с > 5.0с! Поэтому C1 останется отдельной задачей!
    tasks = bitmap_to_tasks(bitmap, merge_adjacent=True, default_duration_s=2.0, max_duration_s=5.0)
    
    # Найдём первую задачу (A1-B1)
    t0 = tasks[0]
    assert t0.color == "black"
    assert t0.duration_s == 4.0
    assert "Объединённый" in t0.note or "B1" in t0.note

    # Следующий 'black' (C1) не поместился по времени в 5.0с и стал самостоятельной задачей на 2.0с
    t1 = tasks[1]
    assert t1.color == "black" and t1.duration_s == 2.0

    # Проверка, что все задачи соблюдают регламент соревнований
    for t in tasks:
        assert t.duration_s <= 5.0
        assert t.passes <= 3


# --- 3. Тест успешного выполнения полного цикла миссии Сверх v2 ---

class MockOnlineLLMClient(LLMClient):
    """Имитированный сетевой LLM-клиент для верификации конвейера без вызова удалённого сервера."""
    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, system_prompt: str, user_prompt: str, timeout_s: Optional[float] = None) -> str:
        self.call_count += 1
        if "Координатор" in system_prompt:
            # Генерация согласованного расписания
            plan_json = {
                "cells": [
                    {"cell": "A1", "color": "black", "duration_s": 3.0, "passes": 1, "note": "Контур"},
                    {"cell": "B1", "color": "red", "duration_s": 2.5, "passes": 1, "note": "Экспрессия"},
                ],
                "flight_commands": [
                    {"drone_id": "drone_black", "action": "takeoff", "z": 2.5, "speed_mps": 1.0},
                    {"drone_id": "drone_black", "action": "paint_zone", "x": 1.0, "y": 1.0, "z": 2.5, "duration_s": 3.0},
                    {"drone_id": "drone_black", "action": "land"},
                    
                    {"drone_id": "drone_red", "action": "takeoff", "z": 3.0, "speed_mps": 1.0},
                    {"drone_id": "drone_red", "action": "yield_wait", "z": 3.0, "duration_s": 3.5, "note": "Пропускает черный дрон"},
                    {"drone_id": "drone_red", "action": "paint_zone", "x": 2.0, "y": 1.0, "z": 3.0, "duration_s": 2.5},
                    {"drone_id": "drone_red", "action": "land"},
                ]
            }
            return f"Финал дискуссии. План роя:\n```json\n{json.dumps(plan_json, ensure_ascii=False)}\n```"
        else:
            return "Оцениваю параметры: скорость 1.0 м/с, высота Z=2.5м безопасны. Готов к выполнению."


def test_full_simulated_mission_runner_execution(tmp_path: Path) -> None:
    """Тест полного выполнения полётной миссии Сверх v2 на симулированных дронах."""
    log_dir = tmp_path / "logs"
    args_list = [
        "--prompt", "Гармония роя: ночь и огонь",
        "--sim",
        "--cols", "3",
        "--rows", "3",
        "--canvas-size-m", "3.0x3.0",
        "--dialogue-rounds", "1",
        "--log-dir", str(log_dir),
    ]
    parser = get_cli_parser()
    args = parser.parse_args(args_list)

    mock_llm = MockOnlineLLMClient()
    report = run_mission(args, custom_llm=mock_llm)

    assert report["status"] == "success"
    assert report["executed_paint_tasks"] >= 2
    assert report["executed_llm_commands"] == 7
    assert report["is_live_mode"] is False

    # Проверяем сохранение артефактов и протоколов миссии
    assert (log_dir / "mission_report.json").exists()
    assert (log_dir / "mission_dialogue.jsonl").exists()
    
    saved_report = json.loads((log_dir / "mission_report.json").read_text(encoding="utf-8"))
    assert saved_report["prompt"] == "Гармония роя: ночь и огонь"
    assert len(saved_report["execution_log"]) > 0


# --- 4. Тест безопасного прерывания (код 1) при сбое сетевой связи с LLM ---

class FailingOnlineLLMClient(LLMClient):
    """Имитатор отказа сети при обращении к API Сверх AI."""
    def chat(self, system_prompt: str, user_prompt: str, timeout_s: Optional[float] = None) -> str:
        raise LLMConnectionError("Сервер Sverk AI недоступен. Переход в офлайн-режим заблокирован.")


def test_llm_connection_failure_aborts_safely(tmp_path: Path) -> None:
    """Проверка, что при сетевом сбое (LLMConnectionError) миссия безопасно завершается с кодом 1 без офлайн-заглушек."""
    log_dir = tmp_path / "logs_fail"
    args_list = [
        "--prompt", "Тест аварийного завершения",
        "--sim",
        "--log-dir", str(log_dir),
    ]
    parser = get_cli_parser()
    args = parser.parse_args(args_list)

    failing_llm = FailingOnlineLLMClient()

    with pytest.raises(SystemExit) as exc_info:
        run_mission(args, custom_llm=failing_llm)

    # Убеждаемся в коде возврата 1 и отсутствии продолжения выполнения в офлайн
    assert exc_info.value.code == 1

    # В отчётах не должно быть успешно завершённой миссии
    assert not (log_dir / "mission_report.json").exists()
