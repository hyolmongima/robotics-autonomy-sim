# sim/viz/draw.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

Cell = Tuple[int, int]

def draw_grid(ax: plt.Axes, grid: np.ndarray, start: Cell | None = None, goal: Cell | None = None) -> None:
    """
    grid: 2D array, 0 free, 1 obstacle
    start/goal: (row, col)
    """
    ax.imshow(grid, origin="upper")  # row 0 at top like a matrix
    ax.set_xticks([])
    ax.set_yticks([])

    if start is not None:
        r, c = start
        ax.plot(c, r, marker="o")  # x=col, y=row

    if goal is not None:
        r, c = goal
        ax.plot(c, r, marker="x")
