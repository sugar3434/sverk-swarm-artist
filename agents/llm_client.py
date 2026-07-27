"""
Абстракция LLM-клиента для диалога агентов роя дронов-художников.

Регламент соревнования требует полной автономности и устойчивости к сбоям
связи во время попытки (сеть на площадке может быть нестабильна). Поэтому
здесь реализована трёхуровневая схема:

    LLMClient (ABC)
      |-- OfflineRuleBasedClient   — детерминированный офлайн-клиент без сети,
      |                              используется в тестах и как безопасный
      |                              fallback при сбое связи.
      |-- OpenAICompatibleClient   — реальный HTTP-клиент (OpenAI-совместимый
      |                              /chat/completions), при любой ошибке не
      |                              падает необработанно, а бросает
      |                              LLMUnavailableError.
      `-- FallbackLLMClient        — обёртка «primary + fallback»: если primary
                                      бросает LLMUnavailableError — прозрачно
                                      уходит на fallback, гарантируя, что
                                      диалог не прервётся.

Фабрика build_llm_client() выбирает нужную комбинацию по переменным окружения.
Модуль НЕ импортирует rclpy/sverk_interfaces — полностью автономен от ROS/дрона.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Optional

import requests

logger = logging.getLogger("agents.llm_client")
if not logger.handlers:
    # Простая настройка на случай, если корневой логгер не сконфигурирован —
    # чтобы предупреждения о сбоях связи были видны на площадке соревнования.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class LLMUnavailableError(RuntimeError):
    """LLM недоступна: нет ключа, таймаут, сетевая ошибка или не-200 ответ.

    Это исключение — контракт между OpenAICompatibleClient и FallbackLLMClient:
    оно НИКОГДА не должно «просачиваться» наружу необработанным из движка
    диалога — оно либо перехватывается FallbackLLMClient, либо (в крайнем
    случае) самим dialogue_engine.py, который деградирует к безопасному плану.
    """


class LLMClient(ABC):
    """Единый интерфейс LLM-клиента для всех личностей и координатора."""

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Отправить системный и пользовательский промпт, получить текст ответа."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Офлайн-заготовки реплик — по одной наполненной палитре фраз на персонажа.
# Используются OfflineRuleBasedClient для генерации правдоподобных, живых
# реплик без обращения к сети. Воспроизводимость обеспечивается seed'ом.
# ---------------------------------------------------------------------------

_ACADEMIST_LINES = [
    "Никакого хаоса — только порядок! Я вижу, что композицию можно выровнять по золотому сечению.",
    "Контур обязан быть чётким. Предлагаю обвести ключевые ячейки и выровнять их по центру полотна.",
    "Симметрия — это не прихоть, а закон хорошей картины. Здесь явно просится зеркальное отражение форм.",
    "Хаотичные мазки коллег меня удручают. Дайте мне навести порядок в геометрии плана.",
    "Пропорции важнее скорости. Один точный контур стоит десяти небрежных пятен.",
]

_EXPRESSIONIST_LINES = [
    "К чёрту линии! Цвет должен кричать! Дайте мне широкий мазок и побольше красного!",
    "Симметрия — это скука! Настоящая картина дышит энергией, а не циркулем!",
    "Хватит считать углы, Академист! Пусть эта ячейка полыхает, а не аккуратно тлеет!",
    "Минималист, ты боишься краски! Я хочу больше прохода здесь — пусть будет ярко и мощно!",
    "Время идёт, а мы всё спорим! Дайте мне действовать — и цвет заговорит сам!",
]

_MINIMALIST_LINES = [
    "Одной капли достаточно. Меньше — значит больше. Предлагаю убрать лишние ячейки из плана.",
    "Тишина в композиции ценнее шума. Здесь можно сократить число проходов вдвое.",
    "Детализатор, не всякая деталь достойна краски. Иногда пустое пространство красноречивее мазка.",
    "Экспрессионист, крик можно заменить одним точным пятном — эффект будет сильнее.",
    "Я бы сжал план до сути: меньше ячеек, меньше времени, больше смысла.",
]

_DETAILER_LINES = [
    "Я пройду каждый сантиметр! Дайте мне ещё 10 минут! Здесь явно не хватает второго прохода!",
    "Подождите, подождите! Если убрать эту ячейку, мы потеряем всю фактуру! Время, время поджимает!",
    "Минималист, нет! Каждая деталь важна! Хотя... ладно, если совсем нет времени, я потерплю.",
    "Ещё чуть-чуть длительности на мелких ячейках — иначе будет недокрашено, а это невыносимо!",
    "Я нервничаю из-за таймера, но не могу бросить работу на полпути — добавьте мне хотя бы один проход!",
]

_COORDINATOR_LINES = [
    "Решение принято: план сбалансирован с учётом порядка, энергии, экономии и деталей.",
    "Спасибо всем за спор. Финальный план учитывает симметрию, акценты и разумный лимит времени.",
    "Координатор фиксирует итог: компромисс между чёткостью, эмоцией, минимализмом и деталями найден.",
]


def _detect_persona(system_prompt: str) -> str:
    """Определить персонажа по вхождению его имени/кредо в системный промпт.

    Это позволяет OfflineRuleBasedClient работать с ЛЮБЫМ system_prompt,
    в том числе если personas.py в будущем изменится — детектор опирается
    на устойчивые маркеры (имя персонажа и характерное кредо), а не на
    точное совпадение текста целиком.

    ВАЖНО: системные промпты персонажей упоминают ДРУГ ДРУГА (например,
    Экспрессионист в своей реплике спорит с «Академистом»), поэтому простой
    поиск по первому совпавшему имени ненадёжен. Вместо этого сначала ищем
    маркер «Ты — @Имя» (или чёткое кредо) в НАЧАЛЕ промпта — так определяется
    сам персонаж, а не те, кого он упоминает в споре.
    """
    text = system_prompt.lower()
    head = text[:120]  # обращение "Ты — @Имя" всегда в начале промпта

    if "координатор" in head and "json" in text:
        return "Координатор"
    if "@академист" in head or "никакого хаоса" in text:
        return "Академист"
    if "@экспрессионист" in head or "цвет должен кричать" in text:
        return "Экспрессионист"
    if "@минималист" in head or "меньше — значит больше" in text or "меньше значит больше" in text:
        return "Минималист"
    if "@детализатор" in head or "каждый сантиметр" in text:
        return "Детализатор"
    return "Координатор"


class OfflineRuleBasedClient(LLMClient):
    """Детерминированный офлайн-клиент без сети.

    Не вызывает никакую настоящую LLM. По системному промпту определяет,
    какому персонажу он принадлежит (по имени/кредо), и подставляет одну из
    заготовленных, но по-разному звучащих реплик в характере, используя
    ``random.Random(seed)`` для лёгкой, но воспроизводимой рандомизации.

    Нужен по двум причинам:
      1. Тесты должны проходить без сети и быстро (см. tests/).
      2. Регламент требует полной автономности робота — если во время
         соревнования пропадёт связь с внешней LLM, диалог обязан
         продолжаться (см. FallbackLLMClient), и офлайн-клиент — тот самый
         безопасный запасной вариант.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)
        self._call_count = 0
        self._last_line_by_persona: dict[str, str] = {}

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        persona = _detect_persona(system_prompt)
        self._call_count += 1
        # Отдельный Random на каждый вызов, но выведенный из общего seed и
        # счётчика вызовов -> результат воспроизводим при одинаковой
        # последовательности вызовов, но реплики не повторяются подряд.
        # random.Random принимает только None/int/float/str/bytes/bytearray,
        # поэтому детерминированно сворачиваем составной ключ в строку.
        seed_key = f"{self._seed}:{self._call_count}:{persona}"
        local_rng = random.Random(seed_key)

        if persona == "Координатор":
            return self._build_coordinator_reply(user_prompt, local_rng)

        lines_map = {
            "Академист": _ACADEMIST_LINES,
            "Экспрессионист": _EXPRESSIONIST_LINES,
            "Минималист": _MINIMALIST_LINES,
            "Детализатор": _DETAILER_LINES,
        }
        lines = lines_map[persona]
        previous = self._last_line_by_persona.get(persona)
        choices = [line for line in lines if line != previous] or lines
        line = local_rng.choice(choices)
        self._last_line_by_persona[persona] = line

        patch = self._build_suggestion(persona, user_prompt, local_rng)
        if patch is None:
            return line
        return f"{line}\n{json.dumps(patch, ensure_ascii=False)}"

    # -- вспомогательные методы -------------------------------------------------

    def _extract_cells(self, user_prompt: str) -> list[str]:
        """Достать id ячеек из текста промпта (ищет вхождения вида "cell": "B3")."""
        cells = re.findall(r'"cell":\s*"([^"]+)"', user_prompt)
        return cells

    def _build_suggestion(self, persona: str, user_prompt: str, rng: random.Random) -> Optional[dict]:
        """Сформировать правдоподобный JSON-патч в характере персонажа."""
        cells = self._extract_cells(user_prompt)
        if not cells:
            return None
        cell = rng.choice(cells)

        if persona == "Академист":
            action, extra = "recolor", {"color": "black"}
        elif persona == "Экспрессионист":
            action, extra = "extend_duration", {"delta_duration_s": 0.5}
        elif persona == "Минималист":
            action, extra = "remove", {}
        else:  # Детализатор
            action, extra = "add_pass", {}

        suggestion = {"cell": cell, "action": action}
        suggestion.update(extra)
        return {"suggest": [suggestion]}

    def _build_coordinator_reply(self, user_prompt: str, rng: random.Random) -> str:
        """Для координатора: если просят JSON — вернуть валидный JSON на основе
        фактически переданных draft-задач (гарантированно валидный json.loads)."""
        wants_json = "json" in user_prompt.lower() or "cells" in user_prompt.lower()
        cells_payload = self._extract_cells_full(user_prompt)

        if not wants_json:
            return rng.choice(_COORDINATOR_LINES)

        # Пустой список — валидный финальный план. Раньше он ошибочно
        # подменялся выдуманной ячейкой A1, которой не было в черновике.
        final_cells = cells_payload if cells_payload is not None else []

        result = {
            "cells": final_cells,
            "notes": rng.choice(_COORDINATOR_LINES),
        }
        return json.dumps(result, ensure_ascii=False)

    def _extract_cells_full(self, user_prompt: str) -> Optional[list[dict]]:
        """Попытаться извлечь из user_prompt полноценный список задач (JSON-массив
        объектов с ключом cell), чтобы честно скопировать/скорректировать их."""
        marker = user_prompt.find("JSON)")
        start = user_prompt.find("[", marker if marker >= 0 else 0)
        if start < 0:
            return None
        try:
            data, _ = json.JSONDecoder().raw_decode(user_prompt[start:])
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, list):
            return None
        cells = []
        for item in data:
            if not isinstance(item, dict) or "cell" not in item:
                continue
            cells.append(
                {
                    "cell": item.get("cell"),
                    "color": item.get("color", "black"),
                    "duration_s": item.get("duration_s", 1.5),
                    "passes": item.get("passes", 1),
                    "priority": item.get("priority", 0),
                    "note": item.get("note", "согласовано координатором"),
                }
            )
        return cells


class OpenAICompatibleClient(LLMClient):
    """Реальный HTTP-клиент к OpenAI-совместимому /chat/completions.

    Конфигурация через переменные окружения:
        SVERK_LLM_BASE_URL — базовый URL API (по умолчанию "https://api.openai.com/v1")
        SVERK_LLM_API_KEY  — ключ API (обязателен)
        SVERK_LLM_MODEL    — имя модели (по умолчанию "gpt-4o-mini")

    При ЛЮБОЙ ошибке (нет ключа, таймаут, не-200 ответ, сетевое исключение)
    логирует предупреждение и бросает LLMUnavailableError — никогда не падает
    необработанным исключением наружу. Это обязательное условие надёжности
    по регламенту соревнования (обработка сбоев связи).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 20.0,
    ):
        self.base_url = (base_url or os.environ.get("SVERK_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("SVERK_LLM_API_KEY")
        self.model = model or os.environ.get("SVERK_LLM_MODEL") or "gpt-4o-mini"
        self.timeout_s = timeout_s

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            logger.warning("SVERK_LLM_API_KEY не задан — LLM недоступна.")
            raise LLMUnavailableError("SVERK_LLM_API_KEY не задан")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        except requests.RequestException as exc:
            logger.warning("Сетевая ошибка при обращении к LLM (%s): %s", url, exc)
            raise LLMUnavailableError(f"Сетевая ошибка при обращении к {url}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — например, UnicodeEncodeError из-за некорректного
            # значения в HTTP-заголовке (ключ/модель с не-ASCII символами) — это тоже
            # сбой связи/конфигурации, а не повод для необработанного падения.
            logger.warning("Неожиданная ошибка при обращении к LLM (%s): %s", url, exc)
            raise LLMUnavailableError(f"Неожиданная ошибка при обращении к {url}: {exc}") from exc

        if response.status_code != 200:
            logger.warning(
                "LLM вернула не-200 статус %s при обращении к %s: %s",
                response.status_code, url, response.text[:500],
            )
            raise LLMUnavailableError(
                f"LLM вернула статус {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("Не удалось разобрать ответ LLM: %s", exc)
            raise LLMUnavailableError(f"Не удалось разобрать ответ LLM: {exc}") from exc


class FallbackLLMClient(LLMClient):
    """Обёртка «primary + fallback»: критичная часть надёжности системы.

    При LLMUnavailableError от primary-клиента прозрачно и без сбоя переходит
    на fallback-клиент (обычно OfflineRuleBasedClient), логируя предупреждение.
    Так публичный диалог агентов не прерывается даже при полном отказе сети
    во время соревнования.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient):
        self.primary = primary
        self.fallback = fallback

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self.primary.chat(system_prompt, user_prompt)
        except LLMUnavailableError as exc:
            logger.warning("Основной LLM-клиент недоступен (%s), переключаюсь на офлайн-заглушку.", exc)
            return self.fallback.chat(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 — любая иная ошибка тоже не должна ронять диалог
            logger.warning("Неожиданная ошибка основного LLM-клиента (%s), переключаюсь на офлайн-заглушку.", exc)
            return self.fallback.chat(system_prompt, user_prompt)


def build_llm_client() -> LLMClient:
    """Фабрика клиента по переменным окружения.

    Если задан SVERK_LLM_API_KEY — пробуем реальный OpenAICompatibleClient,
    но оборачиваем его в FallbackLLMClient с офлайн-заглушкой на случай сбоя
    связи. Если ключ не задан — сразу возвращаем OfflineRuleBasedClient()
    (никаких сетевых попыток вообще).
    """
    api_key = os.environ.get("SVERK_LLM_API_KEY")
    if not api_key:
        logger.info("SVERK_LLM_API_KEY не задан — используется офлайн-клиент.")
        return OfflineRuleBasedClient()

    primary = OpenAICompatibleClient(api_key=api_key)
    return FallbackLLMClient(primary=primary, fallback=OfflineRuleBasedClient())


if __name__ == "__main__":
    # Небольшая демонстрация без сети.
    from agents.personas import PERSONAS, COORDINATOR_PROMPT

    client = OfflineRuleBasedClient(seed=7)
    for name, persona in PERSONAS.items():
        reply = client.chat(persona["system_prompt"], "Промпт: нарисуй сокола. Текущий план (JSON): []")
        print(f"@{name}: {reply}\n")

    coord_reply = client.chat(
        COORDINATOR_PROMPT,
        'Финальные задачи (JSON): [{"cell": "A1", "color": "black", "duration_s": 1.5, '
        '"passes": 1, "priority": 0, "note": ""}]',
    )
    print("Координатор:", coord_reply)
