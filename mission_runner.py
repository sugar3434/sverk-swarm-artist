#!/usr/bin/env python3
"""Main mission execution entry point for Sverk PikoClaw Swarm platform.

Integrates end-to-end flight & painting pipeline:
1. Preflight safety checks (battery >= 40.0%, altitude ceiling <= 4.0m).
2. Vision module (bitmap generation from text prompt & task merging).
3. Online multi-agent LLM dialogue («Академист», «Экспрессионист», «Минималист», «Детализатор» and «Координатор»).
4. Persona speed calculations and yield_wait airspace management in ARUCO map frame.
5. Print chat summary by persona to stdout and save to logs/chat_summary_by_persona.txt.
6. Execution of flight commands via PikoClaw HTTP bridge / ROS 2 facade.

STRICT ZERO OFFLINE MODE:
--offline CLI flags and dummy rule-based fallback generators (OfflineRuleBasedClient) ARE STRICTLY FORBIDDEN.
On network/connection failure or API error, outputs explicit error message and terminates with exit code 1.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from common.schema import Plan
from agents.broadcast import Broadcaster, print_chat_summary_by_persona
from agents.dialogue_engine import run_dialogue
from agents.llm_client import LLMClient, LLMConnectionError, SverkLLMClient, load_env_file
from openclaw.drone_link import DroneLink, LocalDroneLink, PikaClawDroneLink, SimulatedDroneLink
from openclaw.safety import SafetyMonitor, SafetyViolationError
from swarm.canvas_grid import CanvasGrid
from swarm.fleet_coordinator import FleetCoordinator
from vision.bitmap_to_plan import bitmap_to_tasks
from vision.prompt_to_bitmap import generate_bitmap

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sverk.mission_runner")


def parse_canvas_size(size_val: Union[str, float, int]) -> Tuple[float, float]:
    """Parses canvas size string (e.g. '4.0x4.0' or '4.0')."""
    val_str = str(size_val).strip()
    if "x" in val_str.lower():
        parts = val_str.lower().split("x")
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    val_float = float(val_str)
    return val_float, val_float


def get_cli_parser() -> argparse.ArgumentParser:
    """Configures CLI argument parser. Note: --offline argument is strictly forbidden."""
    parser = argparse.ArgumentParser(
        prog="sverk-piko-claw-swarm",
        description="Sverk PikoClaw Swarm Mission Runner (Strict Zero Offline Mode).",
        epilog="Zero Offline Mode is strictly enforced. Offline fallbacks or --offline flags are forbidden.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Sverk Swarm: Harmony of four elements on PikoClaw",
        help="Textual art task prompt for the swarm.",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run mission on simulated drones without physical PikoClaw ROS2 controller.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run mission on live hardware via Arhepelag PikoClaw bridge / ROS2.",
    )
    parser.add_argument(
        "--piko-claw",
        "--pikaclaw",
        dest="piko_claw",
        action="store_true",
        help="Activate Arhepelag PikoClaw HTTP bridge / ROS 2 integration mode.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of canvas grid columns (default 4).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of canvas grid rows (default 4).",
    )
    parser.add_argument(
        "--canvas-size-m",
        type=str,
        default="4.0x4.0",
        help="Canvas size in meters (e.g. '4.0x4.0').",
    )
    parser.add_argument(
        "--dialogue-rounds",
        type=int,
        default=2,
        help="Number of multi-agent discussion rounds.",
    )
    parser.add_argument(
        "--namespaces",
        type=str,
        default="",
        help="Comma-separated drone IDs or ROS 2 namespaces.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save log files and reports.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="Sverk LLM API key.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="https://ai.sverk.io/v1",
        help="Sverk LLM API base URL.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemma4-vlm",
        help="Target LLM model name.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env configuration file.",
    )

    return parser


def check_no_offline_flag(args_list: Optional[List[str]] = None) -> None:
    """Strict verification scanner for forbidden offline flags in CLI arguments.

    Zero Offline Mode Mandate: If --offline, -offline, -o, or offline=true is passed,
    logs a critical error, prints to stderr, and terminates with sys.exit(1).
    """
    target_args = args_list if args_list is not None else sys.argv[1:]
    forbidden = ("--offline", "-offline", "-o", "offline=true", "--offline-mode")
    if any(arg.lower() in forbidden for arg in target_args):
        logger.critical(
            "CRITICAL SECURITY VIOLATION: Offline flag detected (%s). "
            "Zero Offline Mode is strictly enforced. Mission aborted.",
            [arg for arg in target_args if arg.lower() in forbidden]
        )
        print(
            "[ERROR] Security Violation: Offline mode flag '--offline' is strictly forbidden. "
            "Sverk PikoClaw Swarm enforces Zero Offline Mode. Exiting with code 1.",
            file=sys.stderr
        )
        sys.exit(1)


def init_fleet(
    is_live: bool,
    namespaces_str: str,
    use_piko_claw: bool = False,
) -> Dict[str, DroneLink]:
    """Initializes swarm drone links based on execution mode."""
    fleet: Dict[str, DroneLink] = {}
    default_mapping = [
        ("black", "drone_black", "/piko/drone_black"),
        ("red", "drone_red", "/piko/drone_red"),
        ("blue", "drone_blue", "/piko/drone_blue"),
        ("yellow", "drone_yellow", "/piko/drone_yellow"),
    ]

    custom_names: List[str] = []
    if namespaces_str.strip():
        custom_names = [n.strip() for n in namespaces_str.split(",") if n.strip()]

    for i, (color, default_id, piko_prefix) in enumerate(default_mapping):
        node_id = custom_names[i] if i < len(custom_names) else default_id
        if is_live or use_piko_claw:
            logger.info("Initializing PikaClawDroneLink adapter for persona %r (%s)...", color, node_id)
            link: DroneLink = PikaClawDroneLink(
                node_name=node_id,
                offboard_namespace=f"{piko_prefix}/offboard",
                fcu_namespace=f"{piko_prefix}/fcu",
                servo_enable=f"{piko_prefix}/spray/enable",
                servo_angle_topic=f"{piko_prefix}/spray/angle",
            )
        else:
            logger.info("Initializing SimulatedDroneLink for persona %r (%s)...", color, node_id)
            link = SimulatedDroneLink(drone_id=node_id, initial_battery_pct=95.0)

        fleet[color] = link
        fleet[node_id] = link

    return fleet


def run_mission(args: argparse.Namespace, custom_llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    """Executes full swarm mission lifecycle.
    
    Zero Offline Mode: On network error or LLM failure, outputs explicit error message and exits with code 1.
    """
    logger.info("=== STARTING SVERK PIKOCLAW SWARM MISSION (ZERO OFFLINE MODE) ===")

    load_env_file(args.env_file)
    log_dir_path = Path(args.log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    broadcaster = Broadcaster(
        log_file=log_dir_path / "mission_dialogue.jsonl",
        log_to_console=True,
    )

    llm = custom_llm if custom_llm is not None else SverkLLMClient(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        env_file=args.env_file,
    )

    w_m, h_m = parse_canvas_size(args.canvas_size_m)
    grid = CanvasGrid(
        cols=args.cols,
        rows=args.rows,
        width_m=w_m,
        height_m=h_m,
        origin_x=0.0,
        origin_y=0.0,
        origin_z=2.0,
        orientation="horizontal",
    )

    logger.info("Generating bitmap raster for prompt: %r", args.prompt)
    bitmap = generate_bitmap(args.prompt, cols=args.cols, rows=args.rows)
    draft_tasks = bitmap_to_tasks(bitmap, merge_adjacent=True)
    for t in draft_tasks:
        grid.populate_task(t)

    logger.info("Generated %d draft paint tasks from bitmap.", len(draft_tasks))

    is_live_mode = getattr(args, "live", False) and not getattr(args, "sim", False)
    use_piko = getattr(args, "piko_claw", False)
    fleet_links = init_fleet(is_live=is_live_mode, namespaces_str=getattr(args, "namespaces", ""), use_piko_claw=use_piko)
    safety_monitor = SafetyMonitor()

    coordinator = FleetCoordinator(fleet=fleet_links, grid=grid, safety_monitor=safety_monitor)

    logger.info("Verifying preflight battery status (minimum >= 40.0%%)...")
    active_colors = {t.color for t in draft_tasks}
    for color in active_colors:
        drone_middleware = coordinator.get_drone(color)
        safety_monitor.validate_preflight(drone_middleware, drone_id=color)

    logger.info("Launching multi-agent online LLM dialogue (%d rounds)...", args.dialogue_rounds)
    try:
        plan: Plan = run_dialogue(
            prompt=args.prompt,
            draft_tasks=draft_tasks,
            llm=llm,
            broadcaster=broadcaster,
            rounds=args.dialogue_rounds,
            time_budget_s=120.0,
        )
    except (LLMConnectionError, TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
        logger.critical(
            "CRITICAL LLM NETWORK CONNECTION FAILURE (%s): %s. "
            "Zero Offline Mode prohibits offline fallbacks or dummy rules. Exiting with code 1.",
            type(exc).__name__,
            exc,
        )
        print(
            f"[ABORT] Mission failed due to network / LLM connection failure ({exc}). "
            "Offline fallback generators are disabled under Zero Offline Mode. Terminating with exit code 1.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        logger.critical(
            "Unexpected error during online LLM plan generation: %s. Exiting with code 1.", exc
        )
        print(
            f"[ABORT] Unexpected error during online LLM plan generation: {exc}. Exiting with code 1.",
            file=sys.stderr,
        )
        sys.exit(1)

    summary_path = log_dir_path / "chat_summary_by_persona.txt"
    chat_summary_text = print_chat_summary_by_persona(plan, log_dir=log_dir_path)

    logger.info("Agreed LLM plan received: %d paint tasks, %d flight commands.", len(plan.cells), len(plan.flight_commands))
    try:
        execution_report = coordinator.execute_plan(plan)
    except SafetyViolationError as exc:
        logger.critical("Safety violation during mission execution: %s", exc)
        coordinator.emergency_kill()
        raise

    summary = {
        "prompt": args.prompt,
        "status": execution_report.get("status", "unknown"),
        "executed_paint_tasks": execution_report.get("executed_paint_tasks", 0),
        "executed_llm_commands": execution_report.get("executed_llm_commands", 0),
        "is_live_mode": is_live_mode,
        "is_piko_claw": use_piko,
        "transcript_turns_count": len(plan.transcript),
        "execution_log": execution_report.get("log", []),
        "chat_summary_file": str(summary_path),
    }

    report_path = log_dir_path / "mission_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("=== SVERK PIKOCLAW SWARM MISSION COMPLETED SUCCESSFULLY! Report: %s ===", report_path)

    return summary


def main(args_list: Optional[List[str]] = None) -> int:
    """Main CLI entry point for Sverk PikoClaw Swarm."""
    try:
        check_no_offline_flag(args_list)
        parser = get_cli_parser()
        args = parser.parse_args(args_list)
        run_mission(args)
        return 0
    except SystemExit as sys_e:
        return int(sys_e.code) if sys_e.code is not None else 1
    except Exception as exc:
        logger.exception("Mission execution aborted with error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
