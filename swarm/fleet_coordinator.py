"""Fleet Coordinator module (FleetCoordinator) for Sverk PikoClaw Swarm platform.

Responsibilities:
- Mapping paint colors to drone IDs (black -> drone_black, red -> drone_red, etc.).
- Speed calculations based on persona velocities and travel distances in ARUCO map frame.
- yield_wait airspace management with spatial/temporal hold in ARUCO frame.
- Execution of structured LLM flight commands and paint tasks.
- Strict Zero Offline Mode: rejects empty plans or offline rule-based fallbacks.
"""
from __future__ import annotations

import math
import logging
import time
from typing import Any, Dict, List, Optional, Set, Union

from common.schema import FlightCommand, PaintTask, Plan, ScheduledTask, PERSONA_SPEEDS
from openclaw.drone_link import DroneLink, DroneStatus
from openclaw.middleware import DroneMiddleware
from openclaw.safety import SafetyMonitor, SafetyViolationError
from swarm.canvas_grid import CanvasGrid

logger = logging.getLogger("swarm.fleet_coordinator")


class FleetCoordinator:
    """Coordinator for drone swarm flights and PikoClaw paint execution."""

    COLOR_DRONE_MAPPING: Dict[str, str] = {
        "black": "drone_black",
        "red": "drone_red",
        "blue": "drone_blue",
        "yellow": "drone_yellow",
    }

    # Priority hierarchy for airspace access: lower number = higher priority
    PRIORITY_HIERARCHY: Dict[str, int] = {
        "drone_black": 1,
        "drone_red": 2,
        "drone_blue": 3,
        "drone_yellow": 4,
    }

    def __init__(
        self,
        fleet: Optional[Dict[str, Union[DroneLink, DroneMiddleware]]] = None,
        grid: Optional[CanvasGrid] = None,
        safety_monitor: Optional[SafetyMonitor] = None,
    ) -> None:
        self.safety_monitor = safety_monitor or SafetyMonitor()
        self.grid = grid
        self.fleet: Dict[str, DroneMiddleware] = {}
        self.execution_log: List[Dict[str, Any]] = []

        if fleet:
            for identifier, drone in fleet.items():
                self.register_drone(identifier, drone)

    def register_drone(self, identifier: str, drone: Union[DroneLink, DroneMiddleware]) -> None:
        """Registers a drone in fleet registry mapped by color or ID."""
        target_id = self.COLOR_DRONE_MAPPING.get(identifier.lower(), identifier)
        if isinstance(drone, DroneMiddleware):
            middleware = drone
            middleware.drone_id = target_id
        else:
            middleware = DroneMiddleware(drone=drone, safety_monitor=self.safety_monitor, drone_id=target_id)
        self.fleet[target_id] = middleware
        logger.info(f"[FleetCoordinator] Drone {target_id!r} registered in fleet.")

    def get_drone(self, identifier: str) -> DroneMiddleware:
        """Retrieves drone middleware instance by ID or color."""
        target_id = self.COLOR_DRONE_MAPPING.get(identifier.lower(), identifier)
        if target_id not in self.fleet:
            raise KeyError(f"Drone with identifier or color {identifier!r} (ID={target_id!r}) not found in fleet.")
        return self.fleet[target_id]

    def calculate_travel_distance(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
    ) -> float:
        """Calculates 3D Euclidean spatial travel distance in ARUCO map frame meters:
        
        $$D = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}$$
        """
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        dz = float(p2[2]) - float(p1[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def calculate_transit_time(
        self,
        distance_m: float,
        color_or_persona: str,
        settling_time_s: float = 0.5,
    ) -> float:
        """Calculates transit time based on persona velocity and travel distance in ARUCO frame:
        
        $$T = \\frac{D}{V_{persona}} + T_{settling}$$
        """
        color_key = color_or_persona.lower().strip()
        if color_key not in PERSONA_SPEEDS:
            for c in ("black", "red", "blue", "yellow"):
                if c in color_key:
                    color_key = c
                    break
            else:
                for c, role in (("black", "академист"), ("red", "экспрессионист"), ("blue", "минималист"), ("yellow", "детализатор")):
                    if role in color_key:
                        color_key = c
                        break
        v_persona = PERSONA_SPEEDS.get(color_key, 1.0)
        return (distance_m / v_persona) + settling_time_s

    def _verify_llm_plan(self, plan: Plan) -> None:
        """Verifies LLM plan validity. Zero Offline Mode forbids empty plans or offline rule fallbacks."""
        if not plan.cells and not plan.flight_commands:
            msg = (
                "Mission aborted: received plan from LLM Coordinator contains no cells and no flight commands. "
                "Offline rule-based generators or fallback plans are strictly prohibited under Zero Offline Mode."
            )
            logger.error(f"[FleetCoordinator] {msg}")
            raise RuntimeError(msg)

    def schedule_plan(self, plan: Plan) -> List[ScheduledTask]:
        """Schedules conflict-free timeline and flight command sequences based on persona velocities in ARUCO frame."""
        self._verify_llm_plan(plan)

        sorted_cells = sorted(
            plan.cells,
            key=lambda t: (
                t.priority,
                self.PRIORITY_HIERARCHY.get(self.COLOR_DRONE_MAPPING.get(t.color.lower(), t.color.lower()), 99),
            ),
        )
        scheduled_tasks: List[ScheduledTask] = []
        drone_time_cursors: Dict[str, float] = {d_id: 0.0 for d_id in self.COLOR_DRONE_MAPPING.values()}
        drone_last_poses: Dict[str, Tuple[float, float, float]] = {d_id: (0.0, 0.0, 0.0) for d_id in self.COLOR_DRONE_MAPPING.values()}

        for task in sorted_cells:
            if task.color not in self.COLOR_DRONE_MAPPING:
                raise ValueError(f"Task specifies unknown color: {task.color!r}")
            drone_id = self.COLOR_DRONE_MAPPING[task.color]
            persona_speed = PERSONA_SPEEDS.get(task.color, 1.0)

            if (task.x is None or task.y is None or task.z is None) and self.grid is not None:
                x, y, z = self.grid.cell_to_world(task.cell)
                task.x, task.y, task.z = x, y, z
            elif task.z is not None and self.grid is not None:
                self.safety_monitor.validate_altitude(task.z, drone_id=drone_id)

            target_x = task.x if task.x is not None else 0.0
            target_y = task.y if task.y is not None else 0.0
            target_z = task.z if task.z is not None else 2.0
            target_pose = (target_x, target_y, target_z)

            start_pose = drone_last_poses[drone_id]
            travel_dist = self.calculate_travel_distance(start_pose, target_pose)
            transit_time = self.calculate_transit_time(travel_dist, task.color)

            start_offset = drone_time_cursors.get(drone_id, 0.0)
            flight_duration_est = transit_time + float(task.duration_s)
            end_offset = start_offset + flight_duration_est
            drone_time_cursors[drone_id] = end_offset + 1.0  # 1.0s safety buffer
            drone_last_poses[drone_id] = target_pose

            sequence: List[FlightCommand] = [
                FlightCommand(
                    drone_id=drone_id,
                    action="takeoff",
                    z=target_z,
                    speed_mps=persona_speed,
                    note=f"Takeoff to altitude {target_z:.1f}m for cell {task.cell} ({task.color})",
                ),
                FlightCommand(
                    drone_id=drone_id,
                    action="navigate",
                    x=target_x,
                    y=target_y,
                    z=target_z,
                    speed_mps=persona_speed,
                    note=f"Navigate in ARUCO frame to cell {task.cell} (dist={travel_dist:.2f}m, speed={persona_speed}m/s)",
                ),
                FlightCommand(
                    drone_id=drone_id,
                    action="paint_zone",
                    x=target_x,
                    y=target_y,
                    z=target_z,
                    speed_mps=persona_speed,
                    duration_s=float(task.duration_s),
                    passes=int(task.passes),
                    note=f"PikoClaw spray in cell {task.cell}",
                ),
            ]

            scheduled_tasks.append(
                ScheduledTask(
                    task=task,
                    drone_id=drone_id,
                    start_offset_s=start_offset,
                    end_offset_s=end_offset,
                    flight_sequence=sequence,
                )
            )

        logger.info(f"[FleetCoordinator] Scheduled {len(scheduled_tasks)} tasks based on persona speeds in ARUCO frame.")
        return scheduled_tasks

    def execute_command(self, cmd: FlightCommand) -> bool:
        """Executes a single FlightCommand on target drone in ARUCO map frame."""
        drone = self.get_drone(cmd.drone_id)
        action_clean = cmd.action.lower().strip()
        logger.info(f"[FleetCoordinator] -> [{cmd.drone_id}] Action: {action_clean!r} ({cmd.note})")

        if action_clean in ("takeoff", "взлет", "взлёт"):
            target_z = float(cmd.z) if cmd.z is not None else 2.0
            res = drone.takeoff(z=target_z, speed=cmd.speed_mps)
            self.execution_log.append({"drone_id": cmd.drone_id, "action": "takeoff", "z": target_z, "speed_mps": cmd.speed_mps, "note": cmd.note, "ts": time.time()})
            return res

        elif action_clean in ("navigate", "navigate_wait", "навигация"):
            telemetry = drone.get_telemetry()
            cur_x = getattr(telemetry, "x", 0.0) if not isinstance(telemetry, dict) else telemetry.get("x", 0.0)
            cur_y = getattr(telemetry, "y", 0.0) if not isinstance(telemetry, dict) else telemetry.get("y", 0.0)
            cur_z = getattr(telemetry, "z", 2.0) if not isinstance(telemetry, dict) else telemetry.get("z", 2.0)

            target_x = float(cmd.x) if cmd.x is not None else float(cur_x)
            target_y = float(cmd.y) if cmd.y is not None else float(cur_y)
            target_z = float(cmd.z) if cmd.z is not None else float(cur_z)

            res = drone.navigate_wait(x=target_x, y=target_y, z=target_z, yaw=0.0, speed=cmd.speed_mps)
            self.execution_log.append(
                {"drone_id": cmd.drone_id, "action": "navigate", "x": target_x, "y": target_y, "z": target_z, "speed_mps": cmd.speed_mps, "note": cmd.note, "ts": time.time()}
            )
            return res

        elif action_clean in ("paint_zone", "paint", "покраска"):
            if cmd.x is not None and cmd.y is not None and cmd.z is not None:
                drone.navigate_wait(x=float(cmd.x), y=float(cmd.y), z=float(cmd.z), yaw=0.0, speed=cmd.speed_mps)

            duration = float(cmd.duration_s) if cmd.duration_s > 0 else 2.0
            passes = max(1, int(cmd.passes))
            res = drone.paint_zone(duration_s=duration, passes=passes)
            self.execution_log.append(
                {"drone_id": cmd.drone_id, "action": "paint_zone", "duration_s": duration, "passes": passes, "note": cmd.note, "ts": time.time()}
            )
            return res

        elif action_clean in ("yield_wait", "yield", "wait", "уступка", "ожидание"):
            duration = float(cmd.duration_s) if cmd.duration_s > 0 else 2.0
            telemetry = drone.get_telemetry()
            cur_x = getattr(telemetry, "x", 0.0) if not isinstance(telemetry, dict) else telemetry.get("x", 0.0)
            cur_y = getattr(telemetry, "y", 0.0) if not isinstance(telemetry, dict) else telemetry.get("y", 0.0)
            cur_z = getattr(telemetry, "z", 2.0) if not isinstance(telemetry, dict) else telemetry.get("z", 2.0)

            target_x = float(cmd.x) if cmd.x is not None else float(cur_x)
            target_y = float(cmd.y) if cmd.y is not None else float(cur_y)
            target_z = float(cmd.z) if cmd.z is not None else float(cur_z)

            logger.info(
                f"[FleetCoordinator] [{cmd.drone_id}] YIELD_WAIT AIRSPACE HOLD in ARUCO frame at "
                f"({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) for {duration:.1f}s! ({cmd.note})"
            )
            # Spatial and temporal hold: navigate to target waypoint or maintain current ARUCO position hold
            drone.navigate_wait(x=float(target_x), y=float(target_y), z=float(target_z), yaw=0.0, speed=0.5)
            time.sleep(duration)

            self.execution_log.append(
                {"drone_id": cmd.drone_id, "action": "yield_wait", "x": target_x, "y": target_y, "z": target_z, "duration_s": duration, "note": cmd.note, "ts": time.time()}
            )
            return True

        elif action_clean in ("land", "landing", "посадка"):
            res = drone.land()
            self.execution_log.append({"drone_id": cmd.drone_id, "action": "land", "note": cmd.note, "ts": time.time()})
            return res

        else:
            msg = f"Unknown flight instruction from LLM: action={cmd.action!r} for drone {cmd.drone_id}"
            logger.error(f"[FleetCoordinator] {msg}")
            raise ValueError(msg)

    def _get_active_drone_ids(self, plan: Plan) -> Set[str]:
        active_ids: Set[str] = set()
        for cell in plan.cells:
            d_id = self.COLOR_DRONE_MAPPING.get(cell.color)
            if d_id:
                active_ids.add(d_id)
        for cmd in plan.flight_commands:
            d_id = self.COLOR_DRONE_MAPPING.get(cmd.drone_id.lower(), cmd.drone_id)
            if d_id:
                active_ids.add(d_id)
        return active_ids

    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """Full mission plan execution under Zero Offline Mode."""
        self._verify_llm_plan(plan)
        self.safety_monitor.start_mission()

        active_drone_ids = self._get_active_drone_ids(plan)
        if not active_drone_ids and not self.fleet:
            raise RuntimeError("No active drones registered in fleet to execute plan.")

        logger.info("[FleetCoordinator] Starting preflight verification of active drones...")
        for drone_id in active_drone_ids:
            if drone_id not in self.fleet:
                raise KeyError(f"Drone {drone_id!r} specified in plan is not registered in fleet!")
            drone = self.fleet[drone_id]
            status = drone.get_status()
            if not getattr(status, "connected", True):
                drone.connect()
            self.safety_monitor.validate_preflight(drone, drone_id=drone_id)

        logger.info("[FleetCoordinator] Preflight verification passed (all battery levels >= 40.0%).")
        executed_tasks_count = 0

        if plan.cells:
            scheduled_tasks = self.schedule_plan(plan)
            for st in scheduled_tasks:
                logger.info(f"[FleetCoordinator] Executing scheduled cell {st.task.cell} ({st.task.color})...")
                for flight_cmd in st.flight_sequence:
                    self.execute_command(flight_cmd)
                executed_tasks_count += 1

        if plan.flight_commands:
            logger.info(f"[FleetCoordinator] Executing {len(plan.flight_commands)} LLM flight commands...")
            for cmd in plan.flight_commands:
                self.execute_command(cmd)

        for drone_id in active_drone_ids:
            drone = self.fleet[drone_id]
            telemetry = drone.get_telemetry()
            z = getattr(telemetry, "z", 0.0) if not isinstance(telemetry, dict) else telemetry.get("z", 0.0)
            if float(z) > 0.05:
                logger.info(f"[FleetCoordinator] Mission end: landing drone {drone_id} (z={z:.2f}m).")
                drone.land()

        logger.info("[FleetCoordinator] Swarm mission execution completed successfully!")
        return {
            "status": "success",
            "executed_paint_tasks": executed_tasks_count,
            "executed_llm_commands": len(plan.flight_commands),
            "log": self.execution_log,
        }

    def emergency_stop_all(self) -> None:
        """Triggers emergency motor kill across all registered drones."""
        logger.critical("[FleetCoordinator] EMERGENCY STOP ALL DRONES ACTIVATED!")
        self.safety_monitor.emergency_kill(self.fleet)

    def emergency_kill(self) -> None:
        """Alias for emergency_stop_all."""
        self.emergency_stop_all()
