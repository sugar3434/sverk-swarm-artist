"""Safety monitoring and regulation compliance module for Sverk PikoClaw Swarm.

Enforces physical and competition constraints:
- Preflight battery percentage >= 40.0%
- Maximum flight ceiling altitude <= 4.0 m
- Watchdog timer for total mission duration: 15 minutes (900 s)
- Emergency kill function to power off all drones in the swarm.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, Union

logger = logging.getLogger("openclaw.safety")


class SafetyViolationError(RuntimeError, ValueError):
    """Exception raised when safety regulations or physical bounds are violated."""
    pass


class SafetyMonitor:
    """Monitors battery levels, flight altitudes, and mission watchdog timer."""

    MIN_BATTERY_PCT: float = 40.0
    MAX_ALTITUDE_M: float = 4.0
    MAX_MISSION_TIME_S: float = 900.0  # 15 minutes

    def __init__(
        self,
        min_battery_pct: float = MIN_BATTERY_PCT,
        max_altitude_m: float = MAX_ALTITUDE_M,
        max_mission_time_s: float = MAX_MISSION_TIME_S,
    ) -> None:
        self.min_battery_pct = float(min_battery_pct)
        self.max_altitude_m = float(max_altitude_m)
        self.max_mission_time_s = float(max_mission_time_s)
        self.mission_start_time: float = time.time()

    def start_mission(self) -> None:
        """Resets mission timer."""
        self.mission_start_time = time.time()
        logger.info("[SafetyMonitor] Mission timer started (limit: %.1fs).", self.max_mission_time_s)

    def check_mission_time(self, current_time_s: float | None = None) -> float:
        """Verifies mission watchdog timer. Raises SafetyViolationError if > 15 minutes."""
        now = current_time_s if current_time_s is not None else time.time()
        elapsed = now - self.mission_start_time
        if elapsed > self.max_mission_time_s:
            logger.critical(
                "[SafetyMonitor] Mission watchdog triggered! Elapsed %.1fs > max %.1fs.",
                elapsed, self.max_mission_time_s
            )
            raise SafetyViolationError(
                f"Exceeded maximum mission time of 15 minutes ({elapsed:.1f}s elapsed out of {self.max_mission_time_s}s)."
            )
        return elapsed

    def validate_mission_time(self, current_time_s: float | None = None) -> None:
        """Alias for check_mission_time."""
        self.check_mission_time(current_time_s=current_time_s)

    def check_preflight_battery(self, battery_pct: float, drone_id: str = "drone") -> bool:
        """Verifies battery percentage >= 40.0% prior to flight."""
        if battery_pct < self.min_battery_pct:
            msg = (
                f"[{drone_id}] Preflight battery check failed: current battery {battery_pct:.1f}% "
                f"is below required minimum ({self.min_battery_pct:.1f}%)."
            )
            logger.error("[SafetyMonitor] %s", msg)
            raise SafetyViolationError(msg)
        logger.info(
            "[SafetyMonitor] [%s] Preflight battery check passed: %.1f%% >= %.1f%%.",
            drone_id, battery_pct, self.min_battery_pct
        )
        return True

    def validate_preflight(self, drone: Any, drone_id: str = "drone") -> bool:
        """Validates preflight state of a drone instance."""
        status = drone.get_status()
        battery = getattr(status, "battery_pct", None)
        if battery is None and isinstance(status, dict):
            battery = status.get("battery_pct", status.get("battery", 100.0))
        if battery is None:
            logger.warning("[%s] Could not determine battery percentage, assuming 100%%.", drone_id)
            battery = 100.0
        return self.check_preflight_battery(float(battery), drone_id=drone_id)

    def check_altitude(self, z: float, drone_id: str = "drone") -> bool:
        """Verifies altitude ceiling Z <= 4.0m."""
        if z > (self.max_altitude_m + 1e-6):
            msg = (
                f"[{drone_id}] Altitude violation! Target z={z:.2f}m exceeds "
                f"maximum allowed ceiling: {self.max_altitude_m:.1f}m."
            )
            logger.error("[SafetyMonitor] %s", msg)
            raise SafetyViolationError(msg)
        return True

    def validate_altitude(self, z: float | None, drone_id: str = "drone") -> None:
        """Validates maneuver altitude before execution."""
        if z is not None:
            self.check_altitude(float(z), drone_id=drone_id)

    def emergency_kill(self, fleet: Union[Dict[str, Any], Iterable[Any]]) -> None:
        """Triggers emergency motor disarm (kill) across all drones."""
        drones = fleet.values() if isinstance(fleet, dict) else fleet
        logger.critical(">>> [SafetyMonitor] EMERGENCY KILL ACTIVATED FOR ALL DRONES <<<")
        for drone in drones:
            try:
                if hasattr(drone, "kill"):
                    drone.kill()
                elif hasattr(drone, "emergency_stop"):
                    drone.emergency_stop(land=True)
                else:
                    logger.warning("[SafetyMonitor] Drone object %s has no kill() method.", drone)
            except Exception as e:
                logger.error("[SafetyMonitor] Error executing emergency kill on %s: %s", drone, e)
