"""
Трансляция публичного диалога агентов роя дронов-художников.

По регламенту диалог четырёх личностей + координатора публичный и
транслируется на общий экран в реальном времени, причём время диалога
входит в общий 15-минутный лимит попытки. В данной программной реализации
«экран» — это stdout: в реальной инсталляции на соревновании поток stdout
процесса выводится на проектор через OBS (захват окна терминала) либо просто
через терминал, развёрнутый на весь экран второго монитора/проектора —
никакого дополнительного UI не требуется, достаточно читаемого форматирования
и запуска процесса с проекцией его консоли.

Дополнительно каждая реплика пишется в JSONL-лог (по одному JSON-объекту на
строку), чтобы после попытки можно было предъявить судьям полную стенограмму
диалога как доказательство его качества и связности.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Optional

from common.schema import DialogueTurn


class Broadcaster:
    """Транслирует реплики диалога на "общий экран" (stdout) и в JSONL-лог.

    Пример реального использования на соревновании: процесс с диалогом
    агентов запускается в терминале, окно которого захватывается OBS и
    выводится проектором на экран для зрителей/судей в реальном времени —
    то есть каждый print() из emit() виден зрителям сразу же, без буферизации
    (см. flush=True ниже).
    """

    def __init__(self, log_path: Optional[str] = "logs/dialogue_transcript.jsonl", to_stdout: bool = True):
        self.log_path = log_path
        self.to_stdout = to_stdout
        if self.log_path:
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

    def emit(self, turn: DialogueTurn) -> None:
        """Вывести одну реплику диалога на экран и дописать её в лог-файл."""
        if self.to_stdout:
            timestamp = time.strftime("%H:%M:%S", time.localtime(turn.ts))
            print(f"[{timestamp}] @{turn.agent}: {turn.text}", flush=True)

        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
            except OSError as exc:
                # Сбой записи лога не должен останавливать диалог — только
                # предупреждаем на экране (сам диалог важнее его архивации).
                if self.to_stdout:
                    print(f"[WARN] Broadcaster: не удалось записать лог ({exc})", flush=True)


if __name__ == "__main__":
    # Небольшая демонстрация без сети и без зависимостей от остальных модулей.
    demo_broadcaster = Broadcaster(log_path="logs/dialogue_transcript_demo.jsonl")
    demo_broadcaster.emit(DialogueTurn(agent="Система", text="Диалог начат (демо)."))
    demo_broadcaster.emit(DialogueTurn(agent="Академист", text="Никакого хаоса — только порядок!"))
