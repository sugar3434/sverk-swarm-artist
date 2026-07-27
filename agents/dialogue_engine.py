"""
Движок публичного диалога 4 личностей + координатора роя дронов-художников.

Алгоритм run_dialogue():
    1. Транслируем системное сообщение о получении промпта.
    2. До `rounds` раундов (но не дольше `time_budget_s`, т.к. время диалога
       входит в общий 15-минутный лимит попытки): по очереди опрашиваем
       Академиста, Экспрессиониста, Минималиста и Детализатора, транслируем
       их реплики, пытаемся извлечь из ответа JSON-патч с предложениями и
       аккуратно применяем его к рабочей копии плана (без падения, если
       патч кривой или ссылается на несуществующую ячейку).
    3. Вызываем координатора с финальным списком задач, просим строгий JSON
       итогового плана. Если разбор не удался — деградируем к последней
       рабочей копии draft_tasks (безопасный fallback, миссия не должна
       останавливаться из-за кривого JSON от реальной LLM).
    4. Возвращаем Plan с итоговыми ячейками и полной стенограммой диалога.

Любая ошибка (сеть, парсинг, неверный формат) перехватывается и логируется
через broadcaster, деградируя к безопасному плану — диалог не должен
прерывать миссию. Модуль полностью автономен от ROS/дрона: НЕ импортирует
rclpy/sverk_interfaces.
"""
from __future__ import annotations

import copy
import json
import math
import queue
import re
import threading
import time
from dataclasses import asdict
from typing import List, Optional

from common.schema import COLORS, DialogueTurn, PaintTask, Plan

from agents.broadcast import Broadcaster
from agents.llm_client import LLMClient
from agents.personas import COORDINATOR_PROMPT, PERSONAS

# Порядок реплик в каждом раунде — фиксирован регламентом ролей:
# Академист -> Экспрессионист -> Минималист -> Детализатор.
PERSONA_ORDER: List[str] = ["Академист", "Экспрессионист", "Минималист", "Детализатор"]

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _tasks_to_json(tasks: List[PaintTask]) -> str:
    """Сериализовать список PaintTask в компактный JSON для промпта LLM."""
    return json.dumps([asdict(t) for t in tasks], ensure_ascii=False)


def _chat_before_deadline(
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    deadline: float,
) -> str:
    """Выполнить синхронный LLM-вызов, не позволив ему превысить общий дедлайн.

    Клиентский HTTP-таймаут может быть больше бюджета всего диалога. Поэтому
    вызов идёт в daemon-потоке, а основной поток ждёт только оставшееся время.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("бюджет времени диалога исчерпан")

    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put((True, llm.chat(system_prompt, user_prompt)))
        except Exception as exc:  # noqa: BLE001
            result_queue.put((False, exc))

    threading.Thread(target=invoke, daemon=True).start()
    try:
        ok, value = result_queue.get(timeout=remaining)
    except queue.Empty as exc:
        raise TimeoutError("LLM-вызов превысил бюджет времени диалога") from exc
    if not ok:
        raise value
    return value


def _ensure_meaningful_priorities(tasks: List[PaintTask]) -> None:
    """Назначить стабильные приоритеты, если координатор оставил все нулевыми.

    Контур идёт первым как регистрационная направляющая композиции, затем
    заливки. Уникальный ранг внутри групп делает сортировку планировщика
    фактически полезной, а не формальной.
    """
    if len(tasks) < 2 or any(task.priority != 0 for task in tasks):
        return
    indexed = list(enumerate(tasks))
    ordered = sorted(
        indexed,
        key=lambda pair: (
            0
            if (
                pair[1].color == "black"
                or "контурн" in pair[1].note.lower()
                or "академист" in pair[1].note.lower()
            )
            else 1,
            pair[0],
        ),
    )
    for priority, (_original_index, task) in enumerate(ordered):
        task.priority = priority


def _extract_first_json_object(text: str) -> Optional[dict]:
    """Найти и разобрать первый JSON-объект {...} в произвольном тексте.

    Возвращает None, если объект не найден или не парсится — вызывающий код
    в этом случае просто использует текст реплики без патча, не падая.
    Если в тексте несколько кандидатов (жадный поиск может захватить лишнее),
    последовательно сужаем совпадение справа, чтобы найти валидный JSON.
    """
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    candidate = match.group(0)
    # Жадный regex мог захватить слишком много (например, два JSON-объекта
    # подряд или мусор после) — пробуем последовательно обрезать с конца,
    # пока не найдём валидный JSON или не иссякнут варианты.
    for end in range(len(candidate), 0, -1):
        chunk = candidate[:end]
        if not chunk.endswith("}"):
            continue
        try:
            return json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _apply_patch(tasks: List[PaintTask], patch: dict, agent: str, broadcaster: Broadcaster) -> None:
    """Применить JSON-патч предложений агента к рабочей копии списка задач.

    Патч ожидается в виде {"suggest": [{"cell": ..., "action": ..., ...}]}.
    Некорректные или ссылающиеся на несуществующую ячейку записи молча
    (но с логом в трансляцию) пропускаются — они не должны прерывать диалог.
    """
    suggestions = patch.get("suggest")
    if not isinstance(suggestions, list):
        return

    by_cell = {t.cell: t for t in tasks}

    for item in suggestions:
        if not isinstance(item, dict):
            continue
        cell_id = item.get("cell")
        action = item.get("action")

        if not isinstance(cell_id, str) or cell_id not in by_cell:
            broadcaster.emit(
                DialogueTurn(
                    agent="Система",
                    text=(
                        f"Предложение @{agent} по ячейке '{cell_id}' проигнорировано: "
                        f"такой ячейки нет в текущем плане."
                    ),
                )
            )
            continue

        task = by_cell[cell_id]

        try:
            if action == "remove":
                if task in tasks:
                    tasks.remove(task)
                    del by_cell[cell_id]
            elif action == "recolor":
                new_color = item.get("color")
                if new_color in COLORS:
                    task.color = new_color
                elif new_color is not None:
                    raise ValueError(f"неизвестный цвет {new_color!r}")
            elif action == "extend_duration":
                delta = float(item.get("delta_duration_s", 0.0))
                new_duration = task.duration_s + delta
                if math.isfinite(new_duration) and new_duration > 0:
                    task.duration_s = new_duration
            elif action == "add_pass":
                task.passes += 1
            elif action == "keep":
                pass
            else:
                broadcaster.emit(
                    DialogueTurn(
                        agent="Система",
                        text=f"Неизвестное действие '{action}' от @{agent} по ячейке '{cell_id}' проигнорировано.",
                    )
                )
        except (ValueError, TypeError) as exc:
            broadcaster.emit(
                DialogueTurn(
                    agent="Система",
                    text=f"Патч от @{agent} по ячейке '{cell_id}' некорректен и проигнорирован ({exc}).",
                )
            )


def _safe_plan_from_draft(prompt: str, tasks: List[PaintTask], transcript: List[DialogueTurn], notes: str) -> Plan:
    """Собрать безопасный Plan на основе рабочей копии draft-задач (fallback)."""
    return Plan(prompt=prompt, cells=list(tasks), transcript=list(transcript), notes=notes)


def _coordinator_json_to_tasks(data: dict, allowed_cells: Optional[set[str]] = None) -> List[PaintTask]:
    """Преобразовать разобранный JSON координатора в список валидных PaintTask.

    Может бросить исключение (ValueError и т.п.) при некорректных данных —
    вызывающий код обязан перехватить его и деградировать к draft-плану.
    """
    cells_raw = data["cells"]
    if not isinstance(cells_raw, list):
        raise ValueError("Поле 'cells' не список")

    result: List[PaintTask] = []
    for item in cells_raw:
        if allowed_cells is not None and item.get("cell") not in allowed_cells:
            continue
        result.append(
            PaintTask(
                cell=item["cell"],
                color=item["color"],
                duration_s=float(item.get("duration_s", 1.5)),
                passes=int(item.get("passes", 1)),
                priority=int(item.get("priority", 0)),
                note=str(item.get("note", "")),
            )
        )
    return result


def run_dialogue(
    prompt: str,
    draft_tasks: List[PaintTask],
    llm: LLMClient,
    broadcaster: Broadcaster,
    rounds: int = 2,
    time_budget_s: float = 180.0,
) -> Plan:
    """Провести публичный диалог 4 личностей + координатора и вернуть Plan.

    Параметры:
        prompt        — исходный текстовый промпт задания (например, "нарисуй сокола").
        draft_tasks   — черновой список PaintTask (из vision/bitmap_to_plan.py),
                        который агенты обсуждают и правят своими предложениями.
        llm           — реализация LLMClient (офлайн или с fallback на офлайн).
        broadcaster   — Broadcaster для публичной трансляции реплик на экран/в лог.
        rounds        — максимальное число раундов обсуждения (регламент: диалог
                        укладывается в общий лимит попытки).
        time_budget_s — бюджет времени на диалог в секундах; при превышении
                        обсуждение прерывается досрочно (используется time.monotonic()).

    Возвращает Plan — итоговый план покраски с полной стенограммой диалога.
    Функция спроектирована так, чтобы НИКОГДА не бросать исключение наружу:
    любая ошибка (сеть, парсинг JSON, неверный формат ответа LLM) перехватывается
    и приводит к деградации на последний валидный черновик плана.
    """
    transcript: List[DialogueTurn] = []
    working_tasks: List[PaintTask] = copy.deepcopy(draft_tasks)
    start_time = time.monotonic()
    deadline = start_time + max(0.0, time_budget_s)
    _ensure_meaningful_priorities(working_tasks)

    try:
        intro = DialogueTurn(
            agent="Система",
            text=(
                f"Получен промпт задания: «{prompt}». Начинаем публичное обсуждение плана "
                f"покраски рядом из {len(working_tasks)} ячеек. В диалоге участвуют "
                f"@Академист, @Экспрессионист, @Минималист и @Детализатор."
            ),
        )
        broadcaster.emit(intro)
        transcript.append(intro)

        previous_round_summary = "(это первый раунд, предыдущих реплик ещё нет)"

        for round_index in range(1, rounds + 1):
            if time.monotonic() - start_time >= time_budget_s:
                broadcaster.emit(
                    DialogueTurn(
                        agent="Система",
                        text=f"Бюджет времени диалога ({time_budget_s:.0f} с) исчерпан — переходим к решению координатора.",
                    )
                )
                break

            round_lines: List[str] = []

            for persona_name in PERSONA_ORDER:
                if time.monotonic() - start_time >= time_budget_s:
                    broadcaster.emit(
                        DialogueTurn(
                            agent="Система",
                            text=f"Бюджет времени диалога исчерпан во время раунда {round_index} — завершаем обсуждение досрочно.",
                        )
                    )
                    break

                persona = PERSONAS[persona_name]
                user_prompt = (
                    f"Промпт: {prompt}\n"
                    f"Текущий план (JSON): {_tasks_to_json(working_tasks)}\n"
                    f"Высказывания предыдущего раунда: {previous_round_summary}\n"
                    f"Сейчас раунд {round_index} из {rounds}."
                )

                try:
                    reply_text = _chat_before_deadline(
                        llm,
                        system_prompt=persona["system_prompt"],
                        user_prompt=user_prompt,
                        deadline=deadline,
                    )
                except Exception as exc:  # noqa: BLE001 — сбой одной реплики не должен рушить весь диалог
                    broadcaster.emit(
                        DialogueTurn(
                            agent="Система",
                            text=f"Ошибка получения реплики от @{persona_name}: {exc}. Пропускаем реплику.",
                        )
                    )
                    continue

                turn = DialogueTurn(agent=persona_name, text=reply_text)
                broadcaster.emit(turn)
                transcript.append(turn)
                round_lines.append(f"@{persona_name}: {reply_text}")

                patch = _extract_first_json_object(reply_text)
                if patch is not None:
                    try:
                        _apply_patch(working_tasks, patch, persona_name, broadcaster)
                    except Exception as exc:  # noqa: BLE001 — некорректный патч не должен рушить диалог
                        broadcaster.emit(
                            DialogueTurn(
                                agent="Система",
                                text=f"Не удалось применить патч от @{persona_name}: {exc}. Патч проигнорирован.",
                            )
                        )

            previous_round_summary = " | ".join(round_lines) if round_lines else previous_round_summary

        # --- Координатор принимает финальное решение -------------------------
        if time.monotonic() >= deadline:
            return _safe_plan_from_draft(
                prompt,
                working_tasks,
                transcript,
                notes="Бюджет диалога исчерпан; возвращена последняя рабочая копия плана.",
            )

        coordinator_user_prompt = (
            f"Промпт: {prompt}\n"
            f"Финальные задачи после обсуждения (JSON): {_tasks_to_json(working_tasks)}\n"
            f"Полная стенограмма диалога: {' || '.join(f'{t.agent}: {t.text}' for t in transcript)}\n"
            "Верни СТРОГО JSON финального плана, как описано в системном промпте."
        )

        try:
            coordinator_reply = _chat_before_deadline(
                llm,
                system_prompt=COORDINATOR_PROMPT,
                user_prompt=coordinator_user_prompt,
                deadline=deadline,
            )
        except Exception as exc:  # noqa: BLE001
            broadcaster.emit(
                DialogueTurn(
                    agent="Система",
                    text=f"Координатор недоступен ({exc}). Используем последнюю рабочую копию плана без изменений.",
                )
            )
            return _safe_plan_from_draft(
                prompt, working_tasks, transcript,
                notes="Fallback: координатор недоступен, план взят из рабочей копии после обсуждения.",
            )

        coordinator_turn = DialogueTurn(agent="Координатор", text=coordinator_reply)
        broadcaster.emit(coordinator_turn)
        transcript.append(coordinator_turn)

        parsed = _extract_first_json_object(coordinator_reply)
        if parsed is None:
            broadcaster.emit(
                DialogueTurn(
                    agent="Система",
                    text="Не удалось разобрать JSON координатора — используем безопасный fallback-план.",
                )
            )
            return _safe_plan_from_draft(
                prompt, working_tasks, transcript,
                notes="Fallback: JSON координатора не распарсился, план взят из рабочей копии после обсуждения.",
            )

        try:
            final_cells = _coordinator_json_to_tasks(
                parsed,
                allowed_cells={task.cell for task in working_tasks},
            )
        except (KeyError, ValueError, TypeError) as exc:
            broadcaster.emit(
                DialogueTurn(
                    agent="Система",
                    text=f"JSON координатора невалиден ({exc}) — используем безопасный fallback-план.",
                )
            )
            return _safe_plan_from_draft(
                prompt, working_tasks, transcript,
                notes=f"Fallback: JSON координатора невалиден ({exc}).",
            )

        _ensure_meaningful_priorities(final_cells)
        final_notes = str(parsed.get("notes", "")) if isinstance(parsed, dict) else ""
        return Plan(prompt=prompt, cells=final_cells, transcript=transcript, notes=final_notes)

    except Exception as exc:  # noqa: BLE001 — последний рубеж защиты: диалог не должен падать никогда
        broadcaster.emit(
            DialogueTurn(
                agent="Система",
                text=f"Непредвиденная ошибка диалога ({exc}). Аварийно возвращаем последний известный план.",
            )
        )
        return _safe_plan_from_draft(
            prompt, working_tasks or draft_tasks, transcript,
            notes=f"Fallback после непредвиденной ошибки: {exc}",
        )


if __name__ == "__main__":
    # Небольшая демонстрация без сети — использует офлайн-клиент.
    from agents.llm_client import OfflineRuleBasedClient

    demo_tasks = [
        PaintTask(cell="A1", color="black", duration_s=1.5, passes=1),
        PaintTask(cell="B2", color="red", duration_s=1.0, passes=1),
        PaintTask(cell="C3", color="blue", duration_s=2.0, passes=1),
        PaintTask(cell="D4", color="yellow", duration_s=1.2, passes=1),
    ]
    demo_broadcaster = Broadcaster(log_path="logs/dialogue_transcript_demo_engine.jsonl")
    demo_plan = run_dialogue(
        prompt="нарисуй сокола",
        draft_tasks=demo_tasks,
        llm=OfflineRuleBasedClient(seed=1),
        broadcaster=demo_broadcaster,
        rounds=2,
        time_budget_s=30.0,
    )
    print("\nИтоговый план:", demo_plan)
