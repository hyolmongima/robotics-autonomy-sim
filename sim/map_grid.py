# sim/map_grid.py
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

Cell = Tuple[int, int]  # (row, col)
Point2D = Tuple[float, float]


@dataclass
class GridWorld:
    grid: np.ndarray          # (H, W) 0 free, 1 obstacle
    start: Cell
    goal: Cell
    resolution: float         # meters per cell
    center_cell: Cell         # which (row, col) is world (0,0)


def cell_to_world(cell: Cell, *, resolution: float, center_cell: Cell) -> Point2D:
    """
    Map grid cell (row, col) to world (x, y) in meters.

    Convention:
      - world x increases with col
      - world y increases with row
      - center_cell maps to (0,0)
    """
    r, c = cell
    r0, c0 = center_cell
    x = (c - c0) * resolution
    y = (r - r0) * resolution
    return (x, y)


def world_to_cell(xy: Point2D, *, resolution: float, center_cell: Cell) -> Cell:
    """
    Inverse of cell_to_world (rounded to nearest cell).
    """
    x, y = xy
    r0, c0 = center_cell
    c = int(round(x / resolution + c0))
    r = int(round(y / resolution + r0))
    return (r, c)


def make_demo_world() -> GridWorld:
    H = W = 101
    resolution = 1.0
    center_cell = (50, 50)  # world (0,0)

    grid = np.zeros((H, W), dtype=np.uint8)

    # Borders
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # A couple simple obstacles
    grid[30:80, 60] = 1
    grid[40, 20:70] = 1

    start = (10, 10)
    goal = (90, 90)

    return GridWorld(
        grid=grid,
        start=start,
        goal=goal,
        resolution=resolution,
        center_cell=center_cell,
    )
