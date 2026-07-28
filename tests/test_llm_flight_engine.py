"""Комплексные юнит-тесты LLM-клиента, диалогового движка и расчета полётных команд Сверх v2.
Проверяют работу парсера .env, логику повторных попыток, обработку таймаутов, применение JSON-патчей,
парсинг структурированного полётного расписания и полное отсутствие офлайн-режима (Offline mode).
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from common.schema import COLORS, DialogueTurn, FlightCommand, PaintTask, Plan
from agents.broadcast import Broadcaster
from agents.llm_client import (
    LLMClient,
    LLMConnectionError,
    SverkLLMClient,
    load_env_file,
)
from agents.personas import (
    get_agent_prompt,
    get_coordinator_prompt,
    PERSONA_PROMPTS,
)
from agents.dialogue_engine import (
    _chat_before_deadline,
    apply_json_suggestions,
    extract_json,
    run_dialogue,
    synthesize_default_flight_commands,
)


@pytest.fixture
def temp_env_file(tmp_path: Path) -> Path:
    """Создаёт временный файл .env с тестовыми конфигурационными ключами."""
    env_path = tmp_path / ".env"
    content = (
        "# Тестовая конфигурация Sverk AI\n"
        "SVERK_LLM_API_KEY=sk-test-secret-key-v2\n"
        "SVERK_LLM_BASE_URL=https://ai.sverk.io/v1\n"
        "SVERK_LLM_MODEL=gemma4-vlm\n"
    )
    env_path.write_text(content, encoding="utf-8")
    return env_path


def test_load_env_file(temp_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка корректности работы легковесного парсера .env файла."""
    monkeypatch.delenv("SVERK_LLM_API_KEY", raising=False)
    monkeypatch.delenv("SVERK_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SVERK_LLM_MODEL", raising=False)

    loaded = load_env_file(temp_env_file)
    assert loaded.get("SVERK_LLM_API_KEY") == "sk-test-secret-key-v2"
    assert loaded.get("SVERK_LLM_BASE_URL") == "https://ai.sverk.io/v1"
    assert loaded.get("SVERK_LLM_MODEL") == "gemma4-vlm"
    assert os.environ.get("SVERK_LLM_API_KEY") == "sk-test-secret-key-v2"


def test_sverk_llm_client_init(temp_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка инициализации SverkLLMClient данными из .env и аргументами."""
    monkeypatch.delenv("SVERK_LLM_API_KEY", raising=False)
    client = SverkLLMClient(env_file=temp_env_file)
    assert client.api_key == "sk-test-secret-key-v2"
    assert client.base_url == "https://ai.sverk.io/v1"
    assert client.model == "gemma4-vlm"


def test_sverk_llm_client_no_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверка, что отсутствие API-ключа вызывает LLMConnectionError и блокирует офлайн-режим."""
    monkeypatch.delenv("SVERK_LLM_API_KEY", raising=False)
    client = SverkLLMClient(api_key="")
    client.api_key = None  # Принудительно сбрасываем, если в окружении что-то было
    with pytest.raises(LLMConnectionError, match="Отсутствует API-ключ"):
        client.chat("system", "user")


@mock.patch("urllib.request.urlopen")
def test_llm_client_retries_and_success(mock_urlopen: mock.MagicMock) -> None:
    """Проверка повторов (retries) при сетевом сбое и успешного возврата ответа на 2-й попытке."""
    # Первый вызов падает с ошибкой сети, второй успешен
    mock_error_resp = mock.MagicMock()
    mock_error_resp.getcode.return_value = 500
    mock_error_resp.headers = {}
    err = urllib.error.HTTPError("https://ai.sverk.io/v1/chat/completions", 500, "Internal Error", mock_error_resp.headers, None)
    
    mock_success_resp = mock.MagicMock()
    mock_success_resp.getcode.return_value = 200
    mock_success_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "Ответ Sverk AI"}}]
    }).encode("utf-8")
    mock_success_resp.__enter__.return_value = mock_success_resp

    mock_urlopen.side_effect = [err, mock_success_resp]

    client = SverkLLMClient(api_key="test-key", max_retries=2, backoff_factor=0.01)
    response_text = client.chat("System Prompt", "User Prompt")
    
    assert response_text == "Ответ Sverk AI"
    assert mock_urlopen.call_count == 2


@mock.patch("urllib.request.urlopen")
def test_llm_client_all_retries_fail_raises_connection_error(mock_urlopen: mock.MagicMock) -> None:
    """Проверка, что при исчерпании всех повторных попыток выбрасывается LLMConnectionError без перехода в офлайн."""
    mock_urlopen.side_effect = socket.timeout("Timed out connecting to Sverk AI")

    client = SverkLLMClient(api_key="test-key", max_retries=2, backoff_factor=0.01)
    
    with pytest.raises(LLMConnectionError, match="Не удалось получить ответ от Sverk API после 3 попыток"):
        client.chat("Sys", "User")
        
    assert mock_urlopen.call_count == 3


def test_personas_prompts_contain_flight_control() -> None:
    """Проверка, что системные промпты агентов содержат требования к обсуждению параметров полёта."""
    roles = ["Академист", "Экспрессионист", "Минималист", "Детализатор"]
    for role in roles:
        prompt = get_agent_prompt(role)
        assert "speed" in prompt or "скорость" in prompt or "м/с" in prompt, f"Промпт {role} должен содержать скорость полёта."
        assert "Z <= 4.0" in prompt or "высот" in prompt, f"Промпт {role} должен требовать безопасности высоты."
        assert "duration_s" in prompt or "сопл" in prompt, f"Промпт {role} должен учитывать работу сопла."
        assert "yield_wait" in prompt or "столкнов" in prompt, f"Промпт {role} должен обсуждать избегание конфликтов в воздухе."

    coord_prompt = get_coordinator_prompt()
    assert "cells" in coord_prompt and "flight_commands" in coord_prompt
    assert "takeoff" in coord_prompt and "navigate" in coord_prompt and "paint_zone" in coord_prompt and "land" in coord_prompt


def test_broadcaster_logging(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Проверка транслятора диалогов: запись в консоль stdout и создание JSONL-лога."""
    log_file = tmp_path / "logs" / "dialogue_test.jsonl"
    broadcaster = Broadcaster(log_file=log_file, log_to_console=True)

    turn = DialogueTurn(agent="Академист", text="Требую скорость полёта 0.8 м/с и 2 прохода сопла.", ts=time.time())
    broadcaster.emit(turn)

    # Проверяем stdout
    captured = capsys.readouterr()
    assert "[Академист]: Требую скорость полёта 0.8 м/с" in captured.out

    # Проверяем JSONL
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["agent"] == "Академист"
    assert "0.8 м/с" in data["text"]
    assert len(broadcaster.get_transcript()) == 1


def test_chat_before_deadline_timeout() -> None:
    """Проверка генерации TimeoutError при исчерпании бюджета времени в _chat_before_deadline."""
    class SlowLLM(LLMClient):
        def chat(self, sys_p: str, usr_p: str, timeout_s: Optional[float] = None) -> str:
            time.sleep(0.3)
            return "Ответ после таймаута"

    llm = SlowLLM()
    # Устанавливаем дедлайн всего в 0.05 с, чтобы гарантированно сработал TimeoutError
    deadline_ts = time.time() + 0.05
    with pytest.raises(TimeoutError, match="Вызов LLM API превысил допустимый дедлайн"):
        _chat_before_deadline(llm, "sys", "usr", deadline_ts)


def test_apply_json_suggestions_caps_and_modifications() -> None:
    """Проверка применения предложений агентов (add_pass <= 3, extend_duration <= 5.0, recolor, remove)."""
    t1 = PaintTask(cell="A1", color="black", duration_s=4.5, passes=2, priority=0)
    t2 = PaintTask(cell="B2", color="blue", duration_s=2.0, passes=1, priority=1)
    t3 = PaintTask(cell="C3", color="yellow", duration_s=1.0, passes=1, priority=2)

    suggestions_json = {
        "suggestions": [
            {"action": "add_pass", "cell": "A1", "passes": 2, "note": "Добавляем 2 прохода (должно упеться в макс 3)"},
            {"action": "extend_duration", "cell": "A1", "amount_s": 2.0, "note": "Увеличиваем время (должно упереться в макс 5.0с)"},
            {"action": "recolor", "cell": "B2", "color": "red", "note": "Смена цвета на красный"},
            {"action": "remove", "cell": "C3", "note": "Удаление лишней задачи"},
        ]
    }

    updated = apply_json_suggestions([t1, t2, t3], suggestions_json)
    
    assert len(updated) == 2  # C3 была удалена
    
    a1_task = next(t for t in updated if t.cell == "A1")
    assert a1_task.passes == 3  # Ограничение max 3
    assert a1_task.duration_s == 5.0  # Ограничение max 5.0
    assert "Добавляем" in a1_task.note

    b2_task = next(t for t in updated if t.cell == "B2")
    assert b2_task.color == "red"


def test_run_dialogue_full_flow_and_flight_commands_parsing(tmp_path: Path) -> None:
    """Полная проверка работы run_dialogue: раунды обсуждений и финальный парсинг расписания полётов."""
    log_file = tmp_path / "test_run.jsonl"
    broadcaster = Broadcaster(log_file=log_file, log_to_console=False)

    class MockOnlineLLM(LLMClient):
        def chat(self, system_prompt: str, user_prompt: str, timeout_s: Optional[float] = None) -> str:
            if "Координатор" in system_prompt or "flight_commands" in system_prompt:
                # Ответ Координатора с валидной структурой клеток и полётных манёвров
                coord_resp = {
                    "cells": [
                        {"cell": "B3", "color": "black", "duration_s": 2.5, "passes": 2, "priority": 1, "note": "Контур"},
                        {"cell": "C4", "color": "red", "duration_s": 3.0, "passes": 1, "priority": 2, "note": "Акцент"},
                    ],
                    "flight_commands": [
                        {"drone_id": "drone_black", "action": "takeoff", "z": 2.0, "speed_mps": 1.0, "duration_s": 3.0},
                        {"drone_id": "drone_black", "action": "navigate", "x": 10.0, "y": 5.0, "z": 2.0, "speed_mps": 1.0},
                        {"drone_id": "drone_black", "action": "paint_zone", "x": 10.0, "y": 5.0, "z": 2.0, "speed_mps": 0.8, "duration_s": 2.5, "passes": 2},
                        {"drone_id": "drone_black", "action": "land", "x": 0.0, "y": 0.0, "z": 0.0, "speed_mps": 1.0},
                        
                        {"drone_id": "drone_red", "action": "takeoff", "z": 3.0, "speed_mps": 1.5, "duration_s": 3.0},
                        {"drone_id": "drone_red", "action": "yield_wait", "x": 5.0, "y": 5.0, "z": 3.0, "duration_s": 4.0, "note": "Ожидание расхождения"},
                        {"drone_id": "drone_red", "action": "paint_zone", "x": 12.0, "y": 6.0, "z": 3.0, "speed_mps": 1.8, "duration_s": 3.0, "passes": 1},
                        {"drone_id": "drone_red", "action": "land", "x": 0.0, "y": 0.0, "z": 0.0, "speed_mps": 1.5},
                    ],
                }
                return f"Вот финальный согласованный план:\n```json\n{json.dumps(coord_resp, ensure_ascii=False)}\n```"
            else:
                # Дискуссия агента
                if "Экспрессионист" in system_prompt:
                    return (
                        "Предлагаю динамику и скорость 1.8 м/с! "
                        "```json\n{\"suggestions\": [{\"action\": \"recolor\", \"cell\": \"C4\", \"color\": \"red\"}]}\n```"
                    )
                return "Полёт безопасен, высота эшелона Z <= 4.0 м соблюдается."

    draft = [PaintTask(cell="B3", color="black", duration_s=2.0, passes=1)]
    llm = MockOnlineLLM()

    plan: Plan = run_dialogue(
        prompt="Нарисовать вечерний контур города с экспрессией",
        draft_tasks=draft,
        llm=llm,
        broadcaster=broadcaster,
        rounds=1,
        time_budget_s=30.0,
    )

    assert isinstance(plan, Plan)
    assert len(plan.cells) == 2
    assert plan.cells[0].cell == "B3" and plan.cells[0].passes == 2
    assert plan.cells[1].cell == "C4" and plan.cells[1].color == "red"

    assert len(plan.flight_commands) == 8
    drone_black_cmds = [c for c in plan.flight_commands if c.drone_id == "drone_black"]
    assert len(drone_black_cmds) == 4
    assert [c.action for c in drone_black_cmds] == ["takeoff", "navigate", "paint_zone", "land"]

    drone_red_cmds = [c for c in plan.flight_commands if c.drone_id == "drone_red"]
    assert len(drone_red_cmds) == 4
    assert [c.action for c in drone_red_cmds] == ["takeoff", "yield_wait", "paint_zone", "land"]
    assert drone_red_cmds[1].action == "yield_wait" and drone_red_cmds[1].duration_s == 4.0
    assert all(c.z is not None and c.z <= 4.0 for c in plan.flight_commands if c.z is not None)


def test_run_dialogue_llm_connection_error_no_offline_fallback() -> None:
    """Проверка, что при сбое связи (LLMConnectionError) ошибка не подавляется и нет перехода в офлайн-режим."""
    class FailingLLM(LLMClient):
        def chat(self, system_prompt: str, user_prompt: str, timeout_s: Optional[float] = None) -> str:
            raise LLMConnectionError("Связь с Sverk AI прервана. Офлайн-режим отключён.")

    broadcaster = Broadcaster(log_file=None, log_to_console=False)
    llm = FailingLLM()

    with pytest.raises(LLMConnectionError, match="Связь с Sverk AI прервана"):
        run_dialogue("Промпт", [PaintTask("A1", "black")], llm, broadcaster, rounds=1, time_budget_s=10.0)


def test_extract_json() -> None:
    """Проверка надёжности извлечения JSON из различных форматов текста."""
    text1 = "Привет! Вот результат:\n```json\n{\"cells\": []}\n```\nУдачного полёта!"
    assert extract_json(text1) == {"cells": []}

    text2 = "Просто текст без json."
    assert extract_json(text2) is None

    text3 = "Некоторый текст перед JSON {\"action\": \"add_pass\", \"cell\": \"A1\"} и текст после."
    assert extract_json(text3) == {"action": "add_pass", "cell": "A1"}
