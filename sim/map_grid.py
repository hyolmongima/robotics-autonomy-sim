# sim/map_grid.py
from dataclasses import dataclass
from typing import Tuple
import numpy as np

Cell = Tuple[int, int]  # (row, col)

@dataclass
class GridWorld:
    grid: np.ndarray   # shape (H, W), 0 free, 1 obstacle
    start: Cell
    goal: Cell

def make_demo_world() -> GridWorld:
    H, W = 40, 40
    grid = np.zeros((H, W), dtype=np.uint8)

    # border walls
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # a couple obstacles
    grid[10:20, 15] = 1
    grid[5, 5:25] = 1
    grid[22:26, 22:35] = 1

    start = (2, 2)
    goal = (37, 37)
    return GridWorld(grid=grid, start=start, goal=goal)
