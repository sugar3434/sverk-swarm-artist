"""OpenCLaw DroneMiddleware for Sverk PikoClaw Swarm platform.

Provides fault tolerance, safety validation, latency logging, and automatic single retry
on transient timeouts before initiating safe abort.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from openclaw.drone_link import DroneLink, DroneTimeoutError
from openclaw.safety import SafetyMonitor, SafetyViolationError

logger = logging.getLogger("openclaw.middleware")


class DroneMiddleware(DroneLink):
    """Resilient wrapper around DroneLink flight interfaces."""

    def __init__(
        self,
        drone: DroneLink,
        safety_monitor: Optional[SafetyMonitor] = None,
        drone_id: str = "drone",
    ) -> None:
        self.drone = drone
        self.safety_monitor = safety_monitor or SafetyMonitor()
        self.drone_id = drone_id
        self.latency_log: List[Dict[str, Any]] = []

    def _is_transient_timeout(self, err: Exception) -> bool:
        if isinstance(err, (TimeoutError, DroneTimeoutError)):
            return True
        err_str = str(err).lower()
        return "timeout" in err_str or "timed out" in err_str

    def safe_abort(self, action_name: str) -> None:
        """Executes safe abort sequence (land or emergency kill)."""
        logger.critical("[%s] SAFE ABORT TRIGGERED during action '%s'!", self.drone_id, action_name)
        try:
            logger.info("[%s] Attempting emergency landing...", self.drone_id)
            self.drone.land(timeout=10.0)
            logger.info("[%s] Emergency landing succeeded.", self.drone_id)
        except Exception as land_err:
            logger.critical("[%s] Emergency landing failed: %s. Executing emergency kill!", self.drone_id, land_err)
            self.drone.kill()

    def _execute_resilient(self, action_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.safety_monitor.validate_mission_time()

        max_attempts = 2  # 1 initial + 1 retry on timeout
        for attempt in range(1, max_attempts + 1):
            start_ts = time.time()
            try:
                res = func(*args, **kwargs)
                latency = time.time() - start_ts
                self.latency_log.append(
                    {"action": action_name, "latency_s": latency, "attempt": attempt, "status": "success", "ts": start_ts}
                )
                logger.info("[%s] Maneuver '%s' succeeded in %.3fs (attempt %d).", self.drone_id, action_name, latency, attempt)
                return res
            except (TimeoutError, DroneTimeoutError, Exception) as e:
                latency = time.time() - start_ts
                is_timeout = self._is_transient_timeout(e)

                if attempt < max_attempts and is_timeout:
                    self.latency_log.append(
                        {"action": action_name, "latency_s": latency, "attempt": attempt, "status": "retry_timeout", "ts": start_ts, "error": str(e)}
                    )
                    logger.warning(
                        "[%s] Transient timeout in '%s' after %.3fs: %s. Retrying (%d/%d)...",
                        self.drone_id, action_name, latency, e, attempt + 1, max_attempts
                    )
                    continue
                else:
                    status = "failed_timeout" if is_timeout else "failed_error"
                    self.latency_log.append(
                        {"action": action_name, "latency_s": latency, "attempt": attempt, "status": status, "ts": start_ts, "error": str(e)}
                    )
                    logger.error("[%s] Critical maneuver failure in '%s' after %d attempts: %s", self.drone_id, action_name, attempt, e)
                    self.safe_abort(action_name=action_name)
                    raise RuntimeError(f"[{self.drone_id}] Maneuver '{action_name}' failed after {attempt} attempts: {e}") from e

    def connect(self) -> bool:
        start_ts = time.time()
        res = self.drone.connect()
        latency = time.time() - start_ts
        self.latency_log.append({"action": "connect", "latency_s": latency, "attempt": 1, "status": "success" if res else "failed"})
        logger.info("[%s] Connected in %.3fs.", self.drone_id, latency)
        return res

    def takeoff(self, z: float, speed: float = 1.0) -> bool:
        self.safety_monitor.validate_mission_time()
        self.safety_monitor.validate_preflight(self.drone, drone_id=self.drone_id)
        self.safety_monitor.validate_altitude(z, drone_id=self.drone_id)
        return bool(self._execute_resilient("takeoff", self.drone.takeoff, z, speed=speed))

    def navigate_wait(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        speed: float = 1.0,
        tolerance: float = 0.1,
        timeout: float = 30.0,
    ) -> bool:
        self.safety_monitor.validate_mission_time()
        self.safety_monitor.validate_altitude(z, drone_id=self.drone_id)
        return bool(
            self._execute_resilient(
                "navigate_wait",
                self.drone.navigate_wait,
                x=x,
                y=y,
                z=z,
                yaw=yaw,
                speed=speed,
                tolerance=tolerance,
                timeout=timeout,
            )
        )

    def paint_zone(self, duration_s: float, passes: int = 1, angle_deg: float = 60.0) -> bool:
        self.safety_monitor.validate_mission_time()
        return bool(
            self._execute_resilient("paint_zone", self.drone.paint_zone, duration_s=duration_s, passes=passes, angle_deg=angle_deg)
        )

    def land(self, timeout: float = 30.0) -> bool:
        self.safety_monitor.validate_mission_time()
        return bool(self._execute_resilient("land", self.drone.land, timeout=timeout))

    def get_telemetry(self) -> Any:
        return self.drone.get_telemetry()

    def get_status(self) -> Any:
        return self.drone.get_status()

    def kill(self) -> None:
        logger.critical("[%s] Middleware triggering emergency kill.", self.drone_id)
        self.drone.kill()

    def close(self) -> None:
        self.drone.close()


# Aliases
ResilientDroneWrapper = DroneMiddleware
ResilientDroneLink = DroneMiddleware
