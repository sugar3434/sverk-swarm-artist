"""Canvas Grid Coordinate System module for Sverk PikoClaw Swarm platform.

Maps canvas grid cell identifiers (e.g., "A1", "B3") into ARUCO world frame coordinates (x, y, z) in meters.
Enforces strict flight ceiling altitude validation: target z must never exceed 4.0 m.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Tuple

from openclaw.safety import SafetyViolationError

logger = logging.getLogger("swarm.canvas_grid")


class CanvasGrid:
    """Canvas grid mapping to ARUCO world coordinate frame (aruco_map)."""

    MAX_ALTITUDE_M: float = 4.0

    def __init__(
        self,
        cols: int,
        rows: int,
        width_m: float,
        height_m: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        origin_z: float = 0.0,
        orientation: str = "horizontal",
    ) -> None:
        if cols <= 0 or rows <= 0:
            raise ValueError(f"Grid columns and rows must be positive (got {cols}x{rows}).")
        if width_m <= 0.0 or height_m <= 0.0:
            raise ValueError(f"Canvas dimensions must be positive (got {width_m}x{height_m}).")

        self.cols = int(cols)
        self.rows = int(rows)
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.origin_z = float(origin_z)
        self.orientation = orientation.lower()

        self.cell_width = self.width_m / self.cols
        self.cell_height = self.height_m / self.rows

        # Altitude safety pre-check
        if self.origin_z > (self.MAX_ALTITUDE_M + 1e-6):
            msg = f"Canvas origin_z={self.origin_z:.2f}m exceeds maximum allowed altitude of 4.0m!"
            logger.error(msg)
            raise SafetyViolationError(msg)

    def _parse_cell_id(self, cell_id: str) -> Tuple[int, int]:
        """Parses cell string ID (e.g. 'B3' -> col=1, row=2)."""
        clean_id = cell_id.strip().upper()
        match = re.search(r"([A-Z]+)(\d+)", clean_id)
        if not match:
            match = re.search(r"(\d+)([A-Z]+)", clean_id)
            if not match:
                raise ValueError(f"Invalid cell ID format: {cell_id!r}. Expected format: 'B3', 'A1'.")
            num_part, alpha_part = match.group(1), match.group(2)
        else:
            alpha_part, num_part = match.group(1), match.group(2)

        col_idx = 0
        for char in alpha_part:
            col_idx = col_idx * 26 + (ord(char) - ord('A'))

        num_val = int(num_part)
        row_idx = max(0, num_val - 1) if num_val >= 1 else 0

        if col_idx >= self.cols or row_idx >= self.rows:
            raise ValueError(f"Cell ID {cell_id!r} (col={col_idx}, row={row_idx}) out of grid bounds {self.cols}x{self.rows}.")

        return col_idx, row_idx

    def cell_to_world(self, cell_id: str) -> Tuple[float, float, float]:
        """Converts cell ID into ARUCO world frame coordinates (x, y, z) in meters.

        Raises SafetyViolationError if altitude z exceeds 4.0 m limit.
        """
        col_idx, row_idx = self._parse_cell_id(cell_id)

        center_dx = (col_idx + 0.5) * self.cell_width
        center_dy = (row_idx + 0.5) * self.cell_height

        if self.orientation in ("vertical", "xz", "vertical_xz"):
            x = self.origin_x + center_dx
            y = self.origin_y
            z = self.origin_z + center_dy
        elif self.orientation in ("vertical_yz", "yz"):
            x = self.origin_x
            y = self.origin_y + center_dx
            z = self.origin_z + center_dy
        else:
            # Default: horizontal canvas in ARUCO XY plane
            x = self.origin_x + center_dx
            y = self.origin_y + center_dy
            z = self.origin_z

        if z > (self.MAX_ALTITUDE_M + 1e-6):
            msg = (
                f"Safety regulation violation for cell {cell_id!r}! "
                f"Calculated altitude z={z:.3f}m exceeds maximum ceiling limit of 4.0m."
            )
            logger.error(f"[CanvasGrid] {msg}")
            raise SafetyViolationError(msg)

        return x, y, z

    def get_coordinates(self, cell_id: str) -> Tuple[float, float, float]:
        """Alias for cell_to_world(cell_id)."""
        return self.cell_to_world(cell_id)

    def __getitem__(self, cell_id: str) -> Tuple[float, float, float]:
        """Supports indexing grid['B3']."""
        return self.cell_to_world(cell_id)

    def populate_task(self, task: Any) -> Any:
        """Populates ARUCO world coordinates (x, y, z) for PaintTask object."""
        if hasattr(task, "cell") and getattr(task, "cell", None):
            x, y, z = self.cell_to_world(task.cell)
            task.x = x
            task.y = y
            task.z = z
        return task
