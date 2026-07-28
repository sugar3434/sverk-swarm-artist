"""Unit tests for CanvasGrid module and ARUCO map frame coordinate transformations."""
from __future__ import annotations

import pytest

from common.schema import PaintTask
from openclaw.safety import SafetyViolationError
from swarm.canvas_grid import CanvasGrid


def test_grid_cell_parsing_and_aruco_coordinates() -> None:
    """Tests conversion of grid cell IDs to ARUCO world frame coordinates."""
    grid = CanvasGrid(cols=4, rows=4, width_m=4.0, height_m=4.0, origin_x=0.0, origin_y=0.0, origin_z=1.5, orientation="horizontal")
    
    # Cell A1: col 0, row 0 -> x = (0 + 0.5)*1.0 = 0.5, y = (0 + 0.5)*1.0 = 0.5, z = 1.5
    x, y, z = grid.cell_to_world("A1")
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)
    assert z == pytest.approx(1.5)

    # Cell B3: col 1, row 2 -> x = (1 + 0.5)*1.0 = 1.5, y = (2 + 0.5)*1.0 = 2.5, z = 1.5
    xb, yb, zb = grid.cell_to_world("B3")
    assert xb == pytest.approx(1.5)
    assert yb == pytest.approx(2.5)
    assert zb == pytest.approx(1.5)


def test_grid_altitude_safety_enforcement() -> None:
    """Tests altitude ceiling safety limit (Z <= 4.0m)."""
    # Origin Z > 4.0m must raise SafetyViolationError on init
    with pytest.raises(SafetyViolationError):
        CanvasGrid(cols=2, rows=2, width_m=2.0, height_m=2.0, origin_z=4.5)

    # Vertical grid where top row altitude > 4.0m must raise SafetyViolationError
    grid_vert = CanvasGrid(cols=2, rows=4, width_m=2.0, height_m=8.0, origin_x=0.0, origin_y=0.0, origin_z=1.0, orientation="vertical")
    
    # Cell A1: z = 1.0 + 0.5 * 2.0 = 2.0m (safe)
    x1, y1, z1 = grid_vert.cell_to_world("A1")
    assert z1 == pytest.approx(2.0)
    assert z1 <= 4.0

    # Cell A4: z = 1.0 + 3.5 * 2.0 = 8.0m (> 4.0m limit) -> must raise SafetyViolationError
    with pytest.raises(SafetyViolationError) as exc_info:
        grid_vert.cell_to_world("A4")
    assert "4.0" in str(exc_info.value) or "altitude" in str(exc_info.value).lower()


def test_populate_task_coordinates() -> None:
    """Tests populating PaintTask with ARUCO world coordinates."""
    grid = CanvasGrid(cols=4, rows=4, width_m=4.0, height_m=4.0, origin_z=2.0)
    task = PaintTask(cell="C2", color="red")
    grid.populate_task(task)
    
    assert task.x == pytest.approx(2.5)
    assert task.y == pytest.approx(1.5)
    assert task.z == pytest.approx(2.0)


def test_grid_indexing() -> None:
    """Tests grid['B3'] indexing operator."""
    grid = CanvasGrid(cols=4, rows=4, width_m=4.0, height_m=4.0, origin_z=1.0)
    x, y, z = grid["B3"]
    assert x == pytest.approx(1.5)
    assert y == pytest.approx(2.5)
    assert z == pytest.approx(1.0)
