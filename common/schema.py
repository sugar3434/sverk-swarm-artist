"""Unified schema and data structures for Sverk PikoClaw Swarm platform.

Provides strict typed contracts between LLM multi-agent dialogue, flight planning,
PikoClaw ROS 2 / HTTP bridge hardware control, and safety monitoring.
Offline mode is strictly excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

COLORS = ("black", "red", "blue", "yellow")  # black=Академист, red=Экспрессионист, blue=Минималист, yellow=Детализатор

# Target persona velocities in ARUCO map frame (m/s)
PERSONA_SPEEDS: Dict[str, float] = {
    "black": 0.9,       # Академист: 0.8 - 1.0 m/s
    "red": 1.8,         # Экспрессионист: 1.5 - 2.0 m/s
    "blue": 1.2,        # Минималист: 1.0 - 1.4 m/s
    "yellow": 0.6,      # Детализатор: 0.5 - 0.7 m/s
    "drone_black": 0.9,
    "drone_red": 1.8,
    "drone_blue": 1.2,
    "drone_yellow": 0.6,
}


@dataclass
class PaintTask:
    """Task for applying paint to a specific canvas grid cell."""
    cell: str                          # Grid cell identifier (e.g., "B3")
    color: str                         # Color (black, red, blue, yellow)
    duration_s: float = 2.0            # Spray duration in seconds (max 5.0s)
    passes: int = 1                    # Number of passes (max 3)
    priority: int = 0                  # Priority order (lower = earlier)
    x: Optional[float] = None          # ARUCO world coordinate X (m)
    y: Optional[float] = None          # ARUCO world coordinate Y (m)
    z: Optional[float] = None          # ARUCO world coordinate Z (m)
    note: str = ""                     # Decision justification by LLM agent

    def __post_init__(self) -> None:
        if self.color not in COLORS:
            raise ValueError(f"Unknown color {self.color!r}. Allowed colors: {COLORS}")
        self.duration_s = max(0.1, min(5.0, float(self.duration_s)))
        self.passes = max(1, min(3, int(self.passes)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell": self.cell,
            "color": self.color,
            "duration_s": self.duration_s,
            "passes": self.passes,
            "priority": self.priority,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "note": self.note,
        }


@dataclass
class FlightCommand:
    """Structured flight management instruction produced or validated by LLM."""
    drone_id: str                      # Target drone identifier ("drone_black", ...)
    action: str                        # Action: "takeoff", "navigate", "paint_zone", "yield_wait", "land"
    x: Optional[float] = None          # ARUCO target X coordinate (m)
    y: Optional[float] = None          # ARUCO target Y coordinate (m)
    z: Optional[float] = None          # ARUCO target Z coordinate (m, max 4.0m)
    speed_mps: float = 1.0             # Flight speed in m/s (based on persona velocity)
    duration_s: float = 0.0            # Spray duration or yield_wait hold duration in seconds
    passes: int = 1                    # Number of spray passes
    note: str = ""                     # LLM description / maneuver rationale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_id": self.drone_id,
            "action": self.action,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "speed_mps": self.speed_mps,
            "duration_s": self.duration_s,
            "passes": self.passes,
            "note": self.note,
        }


@dataclass
class DialogueTurn:
    """Single turn in multi-agent discussion transcript."""
    agent: str
    text: str
    ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "text": self.text,
            "ts": self.ts,
        }


@dataclass
class Plan:
    """Final swarm mission plan agreed upon by LLM Coordinator and agent persona team."""
    prompt: str
    cells: List[PaintTask] = field(default_factory=list)
    flight_commands: List[FlightCommand] = field(default_factory=list)
    transcript: List[DialogueTurn] = field(default_factory=list)
    outline_color: str = "black"
    notes: str = ""


@dataclass
class ScheduledTask:
    """Conflict-free scheduled task with time bounds and flight command sequence."""
    task: PaintTask
    drone_id: str
    start_offset_s: float
    end_offset_s: float
    flight_sequence: List[FlightCommand] = field(default_factory=list)
