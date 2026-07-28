"""Dialogue broadcast and transcript logging module for Sverk PikoClaw Swarm."""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from common.schema import DialogueTurn, Plan

logger = logging.getLogger("sverk.broadcaster")


class Broadcaster:
    """Broadcaster and transcript recorder for multi-agent LLM dialogue sessions.
    
    Logs formatted transcript turns to stdout in real-time and persists the complete session
    to a JSONL file.
    """

    def __init__(
        self,
        log_file: Optional[Union[str, Path]] = None,
        log_to_console: bool = True,
    ) -> None:
        self.log_file: Optional[Path] = Path(log_file) if log_file else None
        self.log_to_console = log_to_console
        self._transcript: List[DialogueTurn] = []

        if self.log_file is not None:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.error("Failed to create log directory for %s: %s", self.log_file.parent, exc)

    def emit(self, turn: DialogueTurn) -> None:
        """Emits, logs, and persists a single dialogue turn."""
        if turn.ts <= 0.0:
            turn.ts = time.time()

        self._transcript.append(turn)

        # Real-time console output
        if self.log_to_console:
            time_str = time.strftime("%H:%M:%S", time.localtime(turn.ts))
            print(f"[{time_str}] [{turn.agent}]: {turn.text}", file=sys.stdout)

        # Append to JSONL log file
        if self.log_file is not None:
            try:
                data = asdict(turn)
                line = json.dumps(data, ensure_ascii=False)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as exc:
                logger.error("Failed to append dialogue turn to JSONL log %s: %s", self.log_file, exc)

    def get_transcript(self) -> List[DialogueTurn]:
        """Returns a copy of the recorded dialogue transcript."""
        return list(self._transcript)


def print_chat_summary_by_persona(plan: Plan, log_dir: Optional[Union[str, Path]] = None) -> str:
    """Formats multi-agent dialogue transcript by persona role, prints to stdout, and saves to logs/chat_summary_by_persona.txt."""
    persona_descriptions = {
        "Академист": "⬛ @Академист (Color: black | Drone: drone_black | ARUCO Contours & Geometry)",
        "Экспрессионист": "🟥 @Экспрессионист (Color: red | Drone: drone_red | Dynamic Sweeps & Expression)",
        "Минималист": "🟦 @Минималист (Color: blue | Drone: drone_blue | Minimalism & Space)",
        "Детализатор": "🟨 @Детализатор (Color: yellow | Drone: drone_yellow | Filigree Details & Highlights)",
        "Координатор": "👑 @Координатор (Sverk PikoClaw Swarm Leader | Plan Synthesis)",
    }

    turns_by_agent: Dict[str, List[str]] = {}
    for turn in plan.transcript:
        agent = turn.agent
        if agent not in turns_by_agent:
            turns_by_agent[agent] = []
        turns_by_agent[agent].append(turn.text)

    lines = [
        "\n" + "=" * 80,
        "=== 💬 CHAT SUMMARY BY PERSONA ROLE (SVERK PIKOCLAW SWARM) ===",
        "=" * 80,
    ]

    ordered_roles = ["Академист", "Экспрессионист", "Минималист", "Детализатор", "Координатор"]
    for role in ordered_roles:
        header = persona_descriptions.get(role, f"🤖 @{role}")
        lines.append(f"\n{header}:")
        turns = turns_by_agent.get(role, [])
        if not turns:
            lines.append("  • (No dialogue turns recorded for this role)")
        else:
            for idx, text in enumerate(turns, 1):
                clean_text = text.strip()
                lines.append(f"  • Turn {idx}: «{clean_text}»")

    lines.append("\n" + "-" * 80)
    lines.append(f"📌 FINAL COORDINATOR VERDICT: {plan.notes or 'Plan approved for swarm execution.'}")
    lines.append(f"🚁 Agreed execution items: {len(plan.cells)} paint tasks | {len(plan.flight_commands)} LLM flight commands.")
    lines.append("=" * 80 + "\n")

    full_summary_text = "\n".join(lines)
    print(full_summary_text)
    logger.info("Formatted chat summary by persona role generated successfully.")

    target_dir = Path(log_dir) if log_dir is not None else Path("logs")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        summary_file = target_dir / "chat_summary_by_persona.txt"
        summary_file.write_text(full_summary_text, encoding="utf-8")
        logger.info("Saved chat summary by persona to %s", summary_file)
    except OSError as exc:
        logger.error("Failed to write chat summary by persona to file: %s", exc)

    return full_summary_text

