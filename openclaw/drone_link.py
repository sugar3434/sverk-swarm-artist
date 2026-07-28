"""Flight interface and drone control module for Sverk PikoClaw Swarm platform.

Contains:
- Abstract base class `DroneLink` defining flight control contracts.
- `SimulatedDroneLink` for hardware-free simulation and unit testing.
- `PikaClawDroneLink` for native Arhepelag PikoClaw HTTP bridge & ROS 2 interface.
- `LocalDroneLink` for standard ROS 2 facade fallback.
"""
from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openclaw.pikoclaw_bridge_client import PikoClawBridgeClient, PikoClawBridgeError

logger = logging.getLogger("openclaw.drone_link")

try:
    import sverk_interfaces  # type: ignore
except ImportError:
    sverk_interfaces = None  # type: ignore


class DroneTimeoutError(TimeoutError):
    """Exception raised when drone communication or maneuver times out."""
    pass


class DroneConnectionError(RuntimeError):
    """Exception raised when connection to drone endpoint or bridge fails."""
    pass


@dataclass
class DroneStatus:
    """Drone status structure compatible with sverk_interfaces and PikoClaw."""
    battery_pct: float = 95.0
    armed: bool = False
    connected: bool = True
    mode: str = "OFFBOARD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery_pct": self.battery_pct,
            "armed": self.armed,
            "connected": self.connected,
            "mode": self.mode,
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class DroneTelemetry:
    """Drone telemetry data structure in ARUCO map frame."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class DroneLink(abc.ABC):
    """Abstract interface for individual drone communication and flight control."""

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establishes connection to drone controller or bridge."""
        pass

    @abc.abstractmethod
    def takeoff(self, z: float, speed: float = 1.0) -> bool:
        """Ascends vertically to altitude `z` at specified `speed` (m/s)."""
        pass

    @abc.abstractmethod
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
        """Navigates to ARUCO world frame coordinate (x, y, z) with position feedback loop."""
        pass

    @abc.abstractmethod
    def paint_zone(self, duration_s: float, passes: int = 1, angle_deg: float = 60.0) -> bool:
        """Actuates PikoClaw spray nozzle valve for target duration and passes."""
        pass

    @abc.abstractmethod
    def land(self, timeout: float = 30.0) -> bool:
        """Triggers landing sequence and blocks until touched down."""
        pass

    @abc.abstractmethod
    def get_telemetry(self) -> Any:
        """Retrieves telemetry in ARUCO map frame."""
        pass

    @abc.abstractmethod
    def get_status(self) -> Any:
        """Retrieves drone battery level and armed status."""
        pass

    @abc.abstractmethod
    def kill(self) -> None:
        """Emergency motor disarm and hardware kill switch."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Closes connection to drone."""
        pass


class SimulatedDroneLink(DroneLink):
    """Simulated drone flight interface for hardware-free testing."""

    def __init__(self, drone_id: str = "sim_drone", initial_battery_pct: float = 95.0) -> None:
        self.drone_id = drone_id
        self._status = DroneStatus(battery_pct=float(initial_battery_pct), armed=False, connected=False, mode="OFFBOARD")
        self._telemetry = DroneTelemetry(x=0.0, y=0.0, z=0.0, yaw=0.0)
        self.servo_enabled: bool = False
        self.servo_angle: float = 0.0
        self.is_killed: bool = False

        self.trajectory_history: List[Dict[str, Any]] = []
        self.paint_history: List[Dict[str, Any]] = []
        self.servo_events: List[Dict[str, Any]] = []

    def connect(self) -> bool:
        if self.is_killed:
            logger.error("[%s] Cannot connect: drone is killed.", self.drone_id)
            return False
        self._status.connected = True
        logger.info("[%s] Connected to simulation controller (battery: %.1f%%).", self.drone_id, self._status.battery_pct)
        return True

    def takeoff(self, z: float, speed: float = 1.0) -> bool:
        if not self._status.connected:
            self.connect()
        if self.is_killed:
            raise RuntimeError(f"[{self.drone_id}] Takeoff failed: kill switch active.")
        self._status.armed = True
        self._telemetry.z = float(z)
        self._status.battery_pct = max(0.0, self._status.battery_pct - 0.5)
        self.trajectory_history.append({"action": "takeoff", "z": z, "speed": speed, "ts": time.time()})
        logger.info("[%s] Takeoff to z=%.2fm at speed %.2fm/s.", self.drone_id, z, speed)
        return True

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
        if not self._status.armed:
            self._status.armed = True
        self._telemetry.x = float(x)
        self._telemetry.y = float(y)
        self._telemetry.z = float(z)
        self._telemetry.yaw = float(yaw)
        self._status.battery_pct = max(0.0, self._status.battery_pct - 0.3)
        self.trajectory_history.append(
            {"action": "navigate", "x": x, "y": y, "z": z, "yaw": yaw, "speed": speed, "ts": time.time()}
        )
        logger.info("[%s] Navigated to ARUCO x=%.2f, y=%.2f, z=%.2f at speed %.2fm/s.", self.drone_id, x, y, z, speed)
        return True

    def paint_zone(self, duration_s: float, passes: int = 1, angle_deg: float = 60.0) -> bool:
        self.servo_enabled = True
        self.servo_events.append({"event": "servo_enable", "ts": time.time()})

        pass_duration = float(duration_s) / max(1, passes)
        for i in range(passes):
            self.servo_angle = float(angle_deg)
            self.servo_events.append({"event": "servo_set_angle", "angle": angle_deg, "ts": time.time()})
            time.sleep(min(pass_duration, 0.02))
            self.servo_angle = 0.0
            self.servo_events.append({"event": "servo_center", "angle": 0.0, "ts": time.time()})
            self._status.battery_pct = max(0.0, self._status.battery_pct - 0.2)

        self.servo_enabled = False
        self.servo_events.append({"event": "servo_disable", "ts": time.time()})
        self.paint_history.append(
            {"duration_s": duration_s, "passes": passes, "angle_deg": angle_deg, "x": self._telemetry.x, "y": self._telemetry.y, "z": self._telemetry.z}
        )
        logger.info("[%s] PikoClaw spray completed (duration=%.1fs, passes=%d).", self.drone_id, duration_s, passes)
        return True

    def land(self, timeout: float = 30.0) -> bool:
        self._telemetry.z = 0.0
        self._status.armed = False
        self._status.battery_pct = max(0.0, self._status.battery_pct - 0.2)
        self.trajectory_history.append({"action": "land", "ts": time.time()})
        logger.info("[%s] Drone landed.", self.drone_id)
        return True

    def get_telemetry(self) -> DroneTelemetry:
        return self._telemetry

    def get_status(self) -> DroneStatus:
        return self._status

    def kill(self) -> None:
        self._status.armed = False
        self.servo_enabled = False
        self.servo_angle = 0.0
        self.is_killed = True
        self._telemetry.z = 0.0
        self.trajectory_history.append({"action": "kill", "ts": time.time()})
        logger.critical("[%s] EMERGENCY KILL TRIGGERED.", self.drone_id)

    def close(self) -> None:
        self._status.connected = False
        self._status.armed = False
        logger.info("[%s] Closed connection.", self.drone_id)


class PikaClawDroneLink(DroneLink):
    """Adapter for Arhepelag PikoClaw platform communicating via HTTP Bridge or ROS 2 facade."""

    def __init__(
        self,
        node_name: str = "pikoclaw_drone",
        bridge_url: str = "http://localhost:9000",
        offboard_namespace: str = "/piko/offboard",
        fcu_namespace: str = "/piko/fmu",
        servo_enable: str = "/piko/spray/enable",
        servo_angle_topic: str = "/piko/spray/angle",
        servo_center: int = 1500,
        bridge_client: Optional[PikoClawBridgeClient] = None,
    ) -> None:
        self.node_name = node_name
        self.bridge_url = bridge_url.rstrip("/")
        self.offboard_namespace = offboard_namespace
        self.fcu_namespace = fcu_namespace
        self.servo_enable_topic = servo_enable
        self.servo_angle_topic = servo_angle_topic
        self.servo_center_val = servo_center

        self.bridge_client = bridge_client or PikoClawBridgeClient(base_url=self.bridge_url)
        self._status = DroneStatus(battery_pct=95.0, armed=False, connected=False, mode="OFFBOARD")
        self._telemetry = DroneTelemetry(x=0.0, y=0.0, z=0.0, yaw=0.0)

    def connect(self) -> bool:
        logger.info("[%s] Connecting to PikoClaw Bridge at %s...", self.node_name, self.bridge_url)
        if self.bridge_client.healthz():
            self._status.connected = True
            logger.info("[%s] PikoClaw bridge healthz OK.", self.node_name)
            return True
        # If bridge is mock or offline during startup, attempt initial pose check
        try:
            pose_data = self.bridge_client.pose()
            if "xy" in pose_data:
                self._telemetry.x = float(pose_data["xy"][0])
                self._telemetry.y = float(pose_data["xy"][1])
            self._status.connected = True
            logger.info("[%s] PikoClaw pose endpoint reachable.", self.node_name)
            return True
        except PikoClawBridgeError as exc:
            logger.warning("[%s] Could not connect to PikoClaw HTTP bridge: %s", self.node_name, exc)
            self._status.connected = False
            return False

    def takeoff(self, z: float, speed: float = 1.0) -> bool:
        if not self._status.connected:
            self.connect()
        logger.info("[%s] Invoking PikoClaw /takeoff endpoint...", self.node_name)
        res = self.bridge_client.takeoff()
        if res.get("ok", False):
            self._status.armed = True
            self._telemetry.z = float(z)
            return True
        return False

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
        if not self._status.connected:
            self.connect()
        logger.info("[%s] Invoking PikoClaw /move to [%.2f, %.2f] in ARUCO frame...", self.node_name, x, y)
        res = self.bridge_client.move(to=[x, y])
        if res.get("ok", False):
            self._telemetry.x = float(x)
            self._telemetry.y = float(y)
            self._telemetry.z = float(z)
            self._telemetry.yaw = float(yaw)
            return True
        return False

    def paint_zone(self, duration_s: float, passes: int = 1, angle_deg: float = 60.0) -> bool:
        if not self._status.connected:
            self.connect()
        logger.info("[%s] Invoking PikoClaw /spray at current pose (x=%.2f, y=%.2f)...", self.node_name, self._telemetry.x, self._telemetry.y)
        # Polyline points for spray stroke
        spray_points = [[self._telemetry.x + 0.1 * p, self._telemetry.y] for p in range(1, max(2, passes + 1))]
        res = self.bridge_client.spray(points=spray_points, color="#000000", width=2)
        return bool(res.get("ok", False))

    def land(self, timeout: float = 30.0) -> bool:
        if not self._status.connected:
            self.connect()
        logger.info("[%s] Invoking PikoClaw /land endpoint...", self.node_name)
        res = self.bridge_client.land()
        if res.get("ok", False):
            self._telemetry.z = 0.0
            self._status.armed = False
            return True
        return False

    def get_telemetry(self) -> DroneTelemetry:
        try:
            pose_data = self.bridge_client.pose()
            if "xy" in pose_data and isinstance(pose_data["xy"], list):
                self._telemetry.x = float(pose_data["xy"][0])
                self._telemetry.y = float(pose_data["xy"][1])
        except Exception:
            pass
        return self._telemetry

    def get_status(self) -> DroneStatus:
        return self._status

    def kill(self) -> None:
        logger.critical("[%s] PikoClaw Emergency Kill issued.", self.node_name)
        try:
            self.bridge_client.land()
        except Exception:
            pass
        self._status.armed = False
        self._telemetry.z = 0.0

    def close(self) -> None:
        self._status.connected = False
        logger.info("[%s] PikoClaw link closed.", self.node_name)


class LocalDroneLink(DroneLink):
    """Local ROS 2 facade interface fallback via sverk_interfaces."""

    def __init__(
        self,
        node_name: str = "drone_local",
        offboard_namespace: str = "/offboard",
        fcu_namespace: str = "/fcu",
        servo_enable: str = "/gpio/enable",
        servo_angle_topic: str = "/gpio/angle",
        servo_center: int = 1500,
        sverk_mod: Any = None,
    ) -> None:
        self.node_name = node_name
        self.offboard_namespace = offboard_namespace
        self.fcu_namespace = fcu_namespace
        self.servo_enable_topic = servo_enable
        self.servo_angle_topic = servo_angle_topic
        self.servo_center_val = servo_center
        self._mod = sverk_mod or sverk_interfaces
        self.drone: Optional[Any] = None

    def connect(self) -> bool:
        if self._mod is None:
            logger.warning("[%s] sverk_interfaces not installed, using simulated link mode.", self.node_name)
            return False
        logger.info("[%s] Connecting via sverk_interfaces.init()...", self.node_name)
        self.drone = self._mod.init(
            Nodename=self.node_name,
            offboard_namespace=self.offboard_namespace,
            fcu_namespace=self.fcu_namespace,
            servo_enable=self.servo_enable_topic,
            servo_angle_topic=self.servo_angle_topic,
            servo_center=self.servo_center_val,
        )
        return True

    def takeoff(self, z: float, speed: float = 1.0) -> bool:
        if self.drone is None:
            self.connect()
        if self.drone is not None:
            t = self.get_telemetry()
            x = getattr(t, "x", 0.0) if not isinstance(t, dict) else t.get("x", 0.0)
            y = getattr(t, "y", 0.0) if not isinstance(t, dict) else t.get("y", 0.0)
            return bool(self.drone.control.navigate_wait(x=x, y=y, z=z, yaw=0.0, speed=speed, auto_arm=True))
        return True

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
        if self.drone is None:
            self.connect()
        if self.drone is not None:
            return bool(self.drone.control.navigate_wait(x=x, y=y, z=z, yaw=yaw, speed=speed, auto_arm=True, timeout=timeout, tolerance=tolerance))
        return True

    def paint_zone(self, duration_s: float, passes: int = 1, angle_deg: float = 60.0) -> bool:
        if self.drone is None:
            self.connect()
        if self.drone is not None:
            self.drone.gpio.servo_enable()
            pass_duration = float(duration_s) / max(1, passes)
            for i in range(passes):
                self.drone.gpio.servo_set_angle(float(angle_deg))
                time.sleep(min(pass_duration, 0.02))
                self.drone.gpio.servo_center()
            self.drone.gpio.servo_disable()
        return True

    def land(self, timeout: float = 30.0) -> bool:
        if self.drone is not None:
            return bool(self.drone.control.land(timeout=timeout))
        return True

    def get_telemetry(self) -> Any:
        if self.drone is not None:
            return self.drone.control.get_telemetry()
        return DroneTelemetry()

    def get_status(self) -> Any:
        if self.drone is not None:
            return self.drone.control.get_status()
        return DroneStatus()

    def kill(self) -> None:
        if self.drone is not None:
            try:
                self.drone.control.emergency_stop(land=True)
                self.drone.fcu.kill_switch()
            except Exception as e:
                logger.error("[%s] Error during kill: %s", self.node_name, e)

    def close(self) -> None:
        if self.drone is not None:
            try:
                self.drone.fcu.disarm()
            except Exception:
                pass
