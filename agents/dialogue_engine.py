"""Multi-agent dialogue engine for Sverk PikoClaw Swarm platform.

Manages multi-round discussion between painter persona agents («Академист», «Экспрессионист»,
«Минималист», «Детализатор») and final plan synthesis by Coordinator.
Operates strictly online under Zero Offline Mode.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Union

from common.schema import COLORS, DialogueTurn, FlightCommand, PaintTask, Plan, PERSONA_SPEEDS
from agents.broadcast import Broadcaster
from agents.llm_client import LLMClient, LLMConnectionError
from agents.personas import (
    COLOR_TO_AGENT,
    AGENT_TO_DRONE,
    get_agent_prompt,
    get_coordinator_prompt,
)

logger = logging.getLogger("sverk.dialogue_engine")


def _chat_before_deadline(
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    deadline_ts: float,
) -> str:
    """Executes network LLM request in a daemon thread with deadline watchdog.
    
    Raises TimeoutError or LLMConnectionError on failure without offline fallbacks.
    """
    remaining_s = deadline_ts - time.time()
    if remaining_s <= 0.0:
        raise TimeoutError("Time budget for LLM dialogue expired before request start.")

    result_box: List[str] = []
    error_box: List[Exception] = []

    def _worker() -> None:
        try:
            res = llm.chat(system_prompt, user_prompt, timeout_s=remaining_s)
            result_box.append(res)
        except Exception as exc:
            error_box.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=remaining_s)

    if thread.is_alive():
        raise TimeoutError(f"LLM API call exceeded deadline budget ({remaining_s:.2f} s).")
    if error_box:
        exc = error_box[0]
        if isinstance(exc, LLMConnectionError):
            logger.error("Critical failure of LLM connection: %s. Zero Offline Mode prevents fallback.", exc)
            raise exc
        raise exc
    if not result_box:
        raise TimeoutError("LLM call completed without text content.")

    return result_box[0]


def extract_json(text: str) -> Optional[Any]:
    """Extracts and parses JSON object from raw LLM text response."""
    if not text:
        return None

    # 1. Search inside markdown code fence
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. Search between first opening and last closing bracket
    start_idx = -1
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            start_idx = i
            break
    end_idx = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ("}", "]"):
            end_idx = i
            break
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx : end_idx + 1])
        except json.JSONDecodeError:
            pass

    # 3. Direct JSON parse attempt
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def apply_json_suggestions(
    tasks: List[PaintTask],
    suggestions_data: Union[Dict[str, Any], List[Dict[str, Any]], str, Any],
) -> List[PaintTask]:
    """Applies LLM agent suggestions to current task list subject to competition constraints."""
    if isinstance(suggestions_data, str):
        suggestions_data = extract_json(suggestions_data)

    if not suggestions_data:
        return tasks

    suggestions_list: List[Dict[str, Any]] = []
    if isinstance(suggestions_data, dict):
        if "suggestions" in suggestions_data and isinstance(suggestions_data["suggestions"], list):
            suggestions_list = [item for item in suggestions_data["suggestions"] if isinstance(item, dict)]
        elif "action" in suggestions_data:
            suggestions_list = [suggestions_data]
    elif isinstance(suggestions_data, list):
        suggestions_list = [item for item in suggestions_data if isinstance(item, dict)]

    if not suggestions_list:
        return tasks

    updated = [copy.deepcopy(t) for t in tasks]

    for item in suggestions_list:
        action = str(item.get("action", "")).strip().lower()
        cell = str(item.get("cell", "")).strip()
        if not action or not cell:
            continue

        if action == "remove":
            updated = [t for t in updated if t.cell != cell]
            continue

        for t in updated:
            if t.cell == cell:
                if action == "add_pass":
                    add_count = int(item.get("amount", item.get("passes", 1)))
                    t.passes = min(3, max(1, t.passes + add_count))
                elif action == "extend_duration":
                    amount_s = float(item.get("amount_s", 1.0))
                    t.duration_s = min(5.0, max(0.1, t.duration_s + amount_s))
                elif action == "recolor":
                    new_color = str(item.get("color", "")).strip().lower()
                    if new_color in COLORS:
                        t.color = new_color
                note_add = item.get("note")
                if note_add and isinstance(note_add, str):
                    t.note = (t.note + f" [{note_add}]").strip() if t.note else note_add.strip()

    return updated


# Alias for compatibility
apply_json_patch = apply_json_suggestions


def synthesize_default_flight_commands(tasks: List[PaintTask]) -> List[FlightCommand]:
    """Synthesizes default sequence of flight commands based on persona velocities."""
    commands: List[FlightCommand] = []
    for t in tasks:
        agent = COLOR_TO_AGENT.get(t.color, "Академист")
        drone_id = AGENT_TO_DRONE.get(agent, "drone_black")
        persona_speed = PERSONA_SPEEDS.get(t.color, 1.0)
        
        commands.append(
            FlightCommand(
                drone_id=drone_id,
                action="takeoff",
                z=2.0,
                speed_mps=persona_speed,
                duration_s=3.0,
                note=f"Takeoff for {drone_id} to service cell {t.cell}",
            )
        )
        commands.append(
            FlightCommand(
                drone_id=drone_id,
                action="navigate",
                x=t.x or 0.0,
                y=t.y or 0.0,
                z=2.0,
                speed_mps=persona_speed,
                note=f"Navigate to ARUCO frame position for cell {t.cell}",
            )
        )
        commands.append(
            FlightCommand(
                drone_id=drone_id,
                action="paint_zone",
                x=t.x or 0.0,
                y=t.y or 0.0,
                z=2.0,
                speed_mps=persona_speed,
                duration_s=t.duration_s,
                passes=t.passes,
                note=f"PikoClaw spray {t.color} ({t.passes} passes, {t.duration_s}s)",
            )
        )
        commands.append(
            FlightCommand(
                drone_id=drone_id,
                action="land",
                x=0.0,
                y=0.0,
                z=0.0,
                speed_mps=persona_speed,
                duration_s=3.0,
                note=f"Land {drone_id} at origin",
            )
        )
    return commands


def run_dialogue(
    prompt: str,
    draft_tasks: List[PaintTask],
    llm: LLMClient,
    broadcaster: Broadcaster,
    rounds: int = 2,
    time_budget_s: float = 60.0,
) -> Plan:
    """Runs multi-round online discussion and plan synthesis.
    
    Zero Offline Mode: Any LLMConnectionError or TimeoutError is propagated immediately.
    """
    deadline_ts = time.time() + float(time_budget_s)
    current_tasks = [copy.deepcopy(t) for t in draft_tasks]
    agent_roles = ["Академист", "Экспрессионист", "Минималист", "Детализатор"]

    logger.info("Starting online multi-agent dialogue. Budget: %.1fs, rounds: %d.", time_budget_s, rounds)

    for r in range(1, max(0, rounds) + 1):
        for role in agent_roles:
            system_p = get_agent_prompt(role)
            tasks_str = json.dumps([t.to_dict() for t in current_tasks], ensure_ascii=False, indent=2)
            history_str = "\n".join([f"[{turn.agent}]: {turn.text}" for turn in broadcaster.get_transcript()])

            user_p = (
                f"User Prompt: {prompt}\n\n"
                f"Current Tasks (JSON):\n{tasks_str}\n\n"
                f"Discussion Protocol:\n{history_str or '(discussion started)'}\n\n"
                "Evaluate flight speed in ARUCO frame, altitude safety (Z <= 4.0m), spray nozzle timing, "
                "and airspace hold instructions (yield_wait). Provide suggestions JSON if modifying tasks."
            )

            try:
                reply_text = _chat_before_deadline(llm, system_p, user_p, deadline_ts)
            except LLMConnectionError as exc:
                logger.error("Connection failure during dialogue turn for %s: %s", role, exc)
                raise
            except Exception as exc:
                logger.error("Error during LLM call for %s: %s", role, exc)
                raise

            turn = DialogueTurn(agent=role, text=reply_text, ts=time.time())
            broadcaster.emit(turn)

            json_patch = extract_json(reply_text)
            if json_patch:
                current_tasks = apply_json_suggestions(current_tasks, json_patch)

    logger.info("Discussion rounds completed. Invoking Coordinator for final plan synthesis.")
    coord_system_p = get_coordinator_prompt()
    final_tasks_str = json.dumps([t.to_dict() for t in current_tasks], ensure_ascii=False, indent=2)
    full_history_str = "\n".join([f"[{turn.agent}]: {turn.text}" for turn in broadcaster.get_transcript()])

    coord_user_p = (
        f"Original Request: {prompt}\n\n"
        f"Agreed Tasks:\n{final_tasks_str}\n\n"
        f"Full Transcript:\n{full_history_str}\n\n"
        "Generate single valid JSON with 'cells' and 'flight_commands' arrays for PikoClaw swarm execution in ARUCO frame."
    )

    try:
        coord_reply = _chat_before_deadline(llm, coord_system_p, coord_user_p, deadline_ts)
    except LLMConnectionError as exc:
        logger.error("Connection failure during Coordinator synthesis: %s", exc)
        raise
    except Exception as exc:
        logger.error("Error during Coordinator LLM call: %s", exc)
        raise

    broadcaster.emit(DialogueTurn(agent="Координатор", text=coord_reply, ts=time.time()))

    coord_json = extract_json(coord_reply)
    final_cells: List[PaintTask] = []
    final_flight_commands: List[FlightCommand] = []

    if isinstance(coord_json, dict):
        cells_data = coord_json.get("cells", [])
        if isinstance(cells_data, list):
            for item in cells_data:
                if isinstance(item, dict) and "cell" in item:
                    try:
                        t = PaintTask(
                            cell=str(item["cell"]).strip(),
                            color=str(item.get("color", "black")).strip().lower(),
                            duration_s=float(item.get("duration_s", 2.0)),
                            passes=int(item.get("passes", 1)),
                            priority=int(item.get("priority", 0)),
                            x=float(item["x"]) if item.get("x") is not None else None,
                            y=float(item["y"]) if item.get("y") is not None else None,
                            z=float(item["z"]) if item.get("z") is not None else None,
                            note=str(item.get("note", "")).strip(),
                        )
                        final_cells.append(t)
                    except (ValueError, TypeError, KeyError) as err:
                        logger.warning("Error parsing Coordinator cell %s: %s", item, err)

        commands_data = coord_json.get("flight_commands", [])
        if isinstance(commands_data, list):
            for item in commands_data:
                if isinstance(item, dict) and "drone_id" in item and "action" in item:
                    try:
                        z_val = float(item["z"]) if item.get("z") is not None else None
                        if z_val is not None:
                            z_val = min(4.0, max(0.0, z_val))

                        cmd = FlightCommand(
                            drone_id=str(item["drone_id"]).strip(),
                            action=str(item["action"]).strip(),
                            x=float(item["x"]) if item.get("x") is not None else None,
                            y=float(item["y"]) if item.get("y") is not None else None,
                            z=z_val,
                            speed_mps=float(item.get("speed_mps", 1.0)),
                            duration_s=float(item.get("duration_s", 0.0)),
                            passes=int(item.get("passes", 1)),
                            note=str(item.get("note", "")).strip(),
                        )
                        final_flight_commands.append(cmd)
                    except (ValueError, TypeError, KeyError) as err:
                        logger.warning("Error parsing flight command %s: %s", item, err)

    if not final_cells:
        final_cells = current_tasks

    if not final_flight_commands:
        final_flight_commands = synthesize_default_flight_commands(final_cells)

    plan = Plan(
        prompt=prompt,
        cells=final_cells,
        flight_commands=final_flight_commands,
        transcript=broadcaster.get_transcript(),
        outline_color="black",
        notes="Online plan synthesized by Sverk LLM under Zero Offline Mode.",
    )
    return plan
