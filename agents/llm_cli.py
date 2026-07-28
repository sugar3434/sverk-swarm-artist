#!/usr/bin/env python3
"""
agents/llm_cli.py — автономный интерфейс командной строки (CLI) для взаимодействия
с LLM и агентами роя дронов-художников отдельно от ROS-среды и полётного конвейера.

Регламент соревнований и процесс отладки требуют возможности независимой
проверки работы LLM-подключения, оценки качества реплик персонажей и
настройки задержек (повторов) при ожидании ответа API без необходимости
запуска симуляции или реальных дронов.

Возможности CLI:
  1. Режим 'dialogue' — запуск полного публичного обсуждения плана покраски
     (4 личности + Координатор) по заданному художественному промпту.
  2. Режим 'single' — прямой запрос к выбранной личности или Координатору.
  3. Интерактивный режим ('--interactive' / '-i') — живой диалог с агентом
     в терминале в реальном времени.
  4. Прямое управление параметрами сети и задержкой повторов при ожидании
     ответа от API (--retry-delay-s, --max-retries, --base-url, --api-key).

Примеры использования:
    # 1. Быстрая проверка диалога агентов (в офлайн-режиме или через API)
    python3 -m agents.llm_cli --prompt "нарисуй сокола" --mode dialogue

    # 2. Запрос к конкретному персонажу с явной настройкой задержки и ключа API
    python3 -m agents.llm_cli --mode single --persona "Экспрессионист" \
        --prompt "Нужно добавить ярких акцентов" \
        --retry-delay-s 3.0 --max-retries 5

    # 3. Интерактивный чат с Координатором в командной строке
    python3 -m agents.llm_cli --interactive --persona "Координатор"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Добавляем корневую директорию проекта в sys.path при прямом запуске скрипта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.broadcast import Broadcaster
from agents.dialogue_engine import run_dialogue
from agents.llm_client import FallbackLLMClient, OfflineRuleBasedClient, OpenAICompatibleClient, build_llm_client
from agents.personas import COORDINATOR_PROMPT, PERSONAS
from common.schema import PaintTask
from vision.bitmap_to_plan import bitmap_to_tasks, merge_adjacent_same_color
from vision.prompt_to_bitmap import PALETTE, generate_bitmap, quantize_to_palette

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agents.llm_cli")

AVAILABLE_PERSONAS = list(PERSONAS.keys()) + ["Координатор"]


def _get_system_prompt_for_persona(persona_name: str) -> str:
    if persona_name == "Координатор":
        return COORDINATOR_PROMPT
    if persona_name in PERSONAS:
        return PERSONAS[persona_name]["system_prompt"]
    raise ValueError(f"Неизвестный персонаж: {persona_name}. Доступны: {', '.join(AVAILABLE_PERSONAS)}")


def build_custom_client_from_args(args: argparse.Namespace) -> FallbackLLMClient | OfflineRuleBasedClient:
    """Создаёт LLM-клиент с явным учётом CLI-параметров (задержка повтора, ключ, эндпоинт)."""
    if args.api_key:
        os.environ["SVERK_LLM_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["SVERK_LLM_BASE_URL"] = args.base_url
    if args.model:
        os.environ["SVERK_LLM_MODEL"] = args.model
    if args.max_retries is not None:
        os.environ["SVERK_LLM_MAX_RETRIES"] = str(args.max_retries)
    if args.retry_delay_s is not None:
        os.environ["SVERK_LLM_RETRY_DELAY_S"] = str(args.retry_delay_s)

    client = build_llm_client()
    logger.info(
        "Инициализирован LLM-клиент: %s (макс. повторов: %s, задержка ожидания ответа: %s с)",
        type(client).__name__,
        os.environ.get("SVERK_LLM_MAX_RETRIES", 3),
        os.environ.get("SVERK_LLM_RETRY_DELAY_S", 2.0),
    )
    return client


def run_single_request(client: FallbackLLMClient | OfflineRuleBasedClient, persona: str, prompt: str) -> str:
    """Выполнить одиночный запрос к выбранной личности/координатору."""
    sys_prompt = _get_system_prompt_for_persona(persona)
    user_prompt = f"Промпт: {prompt}\nТекущий план (JSON): []"
    print(f"\n[>>> Отправка запроса к @{persona} | ожидание ответа API с повторами при сбое...]")
    reply = client.chat(sys_prompt, user_prompt)
    print(f"\n[@{persona}]: {reply}\n")
    return reply


def run_interactive_loop(client: FallbackLLMClient | OfflineRuleBasedClient, default_persona: str) -> None:
    """Интерактивный сеанс чата с агентами прямо в терминале."""
    current_persona = default_persona
    sys_prompt = _get_system_prompt_for_persona(current_persona)

    print("\n" + "=" * 60)
    print(f"Интерактивный чат роя дронов-художников. Текущий агент: @{current_persona}")
    print("Команды:")
    print(f"  /switch <Имя> — переключить персонажа ({', '.join(AVAILABLE_PERSONAS)})")
    print("  /dialogue <промпт> — запустить полный диалог роя по промпту")
    print("  /exit, /quit или Ctrl+D — завершить чат")
    print("=" * 60 + "\n")

    while True:
        try:
            line = input(f"[{current_persona}] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение интерактивного режима.")
            break

        if not line:
            continue
        if line.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Завершение интерактивного режима.")
            break

        if line.startswith("/switch "):
            target = line.split(" ", 1)[1].strip()
            # Поиск без учёта регистра
            matched = next((p for p in AVAILABLE_PERSONAS if p.lower() == target.lower()), None)
            if matched:
                current_persona = matched
                sys_prompt = _get_system_prompt_for_persona(current_persona)
                print(f"--> Переключено на персонажа: @{current_persona}\n")
            else:
                print(f"--> [Ошибка] Персонаж '{target}' не найден. Доступные: {', '.join(AVAILABLE_PERSONAS)}\n")
            continue

        if line.startswith("/dialogue "):
            prompt_text = line.split(" ", 1)[1].strip()
            _run_dialogue_cli(client, prompt_text, rounds=2, time_budget_s=120.0)
            continue

        print(f"[Ожидание ответа API от @{current_persona} (с задержкой при сбоях/429)...]")
        reply = client.chat(sys_prompt, f"Промпт от оператора: {line}\nТекущий план (JSON): []")
        print(f"\n[@{current_persona}]: {reply}\n")


def _run_dialogue_cli(client: FallbackLLMClient | OfflineRuleBasedClient, prompt: str, rounds: int, time_budget_s: float) -> None:
    """Генерация чернового плана и запуск публичного обсуждения в терминале."""
    print(f"\n[>>> Генерация чернового плана по промпту: «{prompt}»...]")
    bitmap = generate_bitmap(prompt, cols=8, rows=8)
    quantized = quantize_to_palette(bitmap, PALETTE)
    draft_tasks = merge_adjacent_same_color(bitmap_to_tasks(quantized))
    print(f"[>>> Черновой план готов ({len(draft_tasks)} ячеек). Запуск публичной дискуссии агентов...]\n")

    broadcaster = Broadcaster(to_stdout=True)
    plan = run_dialogue(
        prompt=prompt,
        draft_tasks=draft_tasks,
        llm=client,
        broadcaster=broadcaster,
        rounds=rounds,
        time_budget_s=time_budget_s,
    )
    print("\n" + "-" * 50)
    print(f"[ИТОГОВЫЙ ПЛАН КООРДИНАТОРА]: {len(plan.cells)} ячеек утверждены.")
    if plan.notes:
        print(f"[ЗАМЕТКИ КООРДИНАТОРА]: {plan.notes}")
    print("-" * 50 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Автономная работа с LLM и диалогом агентов в командной строке.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompt", default="нарисуй сокола", help="Текст художественного задания (промпт)")
    parser.add_argument(
        "--mode",
        choices=["dialogue", "single"],
        default="dialogue",
        help="Режим: 'dialogue' (дискуссия роя) или 'single' (одна реплика)",
    )
    parser.add_argument(
        "--persona",
        choices=AVAILABLE_PERSONAS,
        default="Академист",
        help="Персонаж для режима 'single' или старта интерактивного чата",
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Запустить интерактивный терминальный чат")
    parser.add_argument("--rounds", type=int, default=2, help="Число раундов в режиме dialogue")
    parser.add_argument("--time-budget", type=float, default=120.0, help="Бюджет времени в секундах на весь диалог")

    # Параметры подключения и задержек при ожидании API ответа
    parser.add_argument("--api-key", default="", help="Ключ API (или OPENAI_API_KEY/SVERK_LLM_API_KEY)")
    parser.add_argument("--base-url", default="", help="Базовый URL (например, http://localhost:11434/v1 для локальных LLM)")
    parser.add_argument("--model", default="", help="Имя модели LLM (например, gpt-4o-mini или llama3)")
    parser.add_argument("--max-retries", type=int, default=None, help="Макс. число повторных попыток при сбоях или 429")
    parser.add_argument(
        "--retry-delay-s",
        type=float,
        default=None,
        help="Задержка в секундах между попытками пока API не даст результат ответа",
    )

    args = parser.parse_args(argv)
    client = build_custom_client_from_args(args)

    if args.interactive:
        run_interactive_loop(client, default_persona=args.persona)
    elif args.mode == "single":
        run_single_request(client, persona=args.persona, prompt=args.prompt)
    elif args.mode == "dialogue":
        _run_dialogue_cli(client, prompt=args.prompt, rounds=args.rounds, time_budget_s=args.time_budget)

    return 0


if __name__ == "__main__":
    sys.exit(main())
