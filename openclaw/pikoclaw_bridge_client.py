"""HTTP Client for Arhepelag PikoClaw Bridge Node (§6 spec).

Communicates with PikoClaw ROS 2 HTTP bridge sidecar (`bridge/ros2/bridge_node.py` or `bridge/mock.py`).
Exposes PikoClaw painter endpoints (/move, /spray, /takeoff, /land, /pose, /healthz, /led).
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openclaw.pikoclaw_bridge_client")


class PikoClawBridgeError(RuntimeError):
    """Exception raised on PikoClaw bridge communication or HTTP error."""
    pass


class PikoClawBridgeClient:
    """HTTP/JSON REST Client for Arhepelag PikoClaw Bridge Node."""

    def __init__(self, base_url: str = "http://localhost:9000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            logger.error("HTTP %d error from PikoClaw bridge at %s: %s", exc.code, path, exc.reason)
            raise PikoClawBridgeError(f"PikoClaw bridge HTTP {exc.code} error at {path}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.error("Failed to connect to PikoClaw bridge at %s: %s", url, exc)
            raise PikoClawBridgeError(f"Failed to connect to PikoClaw bridge at {url}: {exc}") from exc

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        req = urllib.request.Request(
            url,
            headers={"content-type": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            logger.error("HTTP %d error from PikoClaw bridge at %s: %s", exc.code, path, exc.reason)
            raise PikoClawBridgeError(f"PikoClaw bridge HTTP {exc.code} error at {path}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.error("Failed to connect to PikoClaw bridge at %s: %s", url, exc)
            raise PikoClawBridgeError(f"Failed to connect to PikoClaw bridge at {url}: {exc}") from exc

    def healthz(self) -> bool:
        """Checks healthz endpoint of PikoClaw bridge."""
        try:
            res = self._get("/healthz")
            return bool(res.get("ok", False))
        except Exception:
            return False

    def takeoff(self) -> dict:
        """Issues takeoff command to PikoClaw drone."""
        return self._post("/takeoff", {})

    def land(self) -> dict:
        """Issues land command to PikoClaw drone."""
        return self._post("/land", {})

    def move(self, to: List[float]) -> dict:
        """Navigates PikoClaw drone to target [x, y] in ARUCO map frame."""
        if not isinstance(to, (list, tuple)) or len(to) < 2:
            raise ValueError(f"Target 'to' must be [x, y], got {to!r}")
        return self._post("/move", {"to": [float(to[0]), float(to[1])]})

    def spray(self, points: List[List[float]], color: str = "#ffffff", width: int = 2) -> dict:
        """Triggers PikoClaw spray stroke along polyline points with specified color hex."""
        if not isinstance(points, list) or len(points) < 1:
            raise ValueError(f"Spray 'points' must be a non-empty list of [x, y], got {points!r}")
        clean_points = [[float(p[0]), float(p[1])] for p in points]
        return self._post("/spray", {"points": clean_points, "color": color, "width": int(width)})

    def pose(self) -> dict:
        """Queries current telemetry pose [x, y] and heading from PikoClaw bridge."""
        return self._get("/pose")

    def led(
        self,
        effect: str = "fill",
        color: Optional[str] = None,
        r: int = 0,
        g: int = 0,
        b: int = 0,
        seconds: float = 0,
    ) -> dict:
        """Sets RGB LED pattern on PikoClaw drone."""
        payload: Dict[str, Any] = {
            "effect": effect,
            "r": int(r) & 255,
            "g": int(g) & 255,
            "b": int(b) & 255,
            "seconds": float(seconds),
        }
        if color:
            payload["color"] = color
        return self._post("/led", payload)
