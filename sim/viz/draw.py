# sim/viz/draw.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional
from matplotlib.colors import ListedColormap, BoundaryNorm

from sim.types import Path2D

Cell = Tuple[int, int]


def _grid_extent(grid: np.ndarray, resolution: float, center_cell: Cell) -> Tuple[float, float, float, float]:
    H, W = grid.shape
    r0, c0 = center_cell

    xmin = (0 - c0) * resolution - 0.5 * resolution
    xmax = (W - 1 - c0) * resolution + 0.5 * resolution
    ymin = (0 - r0) * resolution - 0.5 * resolution
    ymax = (H - 1 - r0) * resolution + 0.5 * resolution
    return (xmin, xmax, ymin, ymax)


def draw_grid(
    ax: plt.Axes,
    grid: np.ndarray,
    start: Optional[Cell] = None,
    goal: Optional[Cell] = None,
    *,
    resolution: float = 1.0,
    center_cell: Cell = (0, 0),
    show_gridlines: bool = True,
) -> None:
    """
    Draw occupancy grid in world-meter coordinates.
    grid: 0 free, 1 obstacle
    """
    extent = _grid_extent(grid, resolution, center_cell)

    # White free space + black obstacles
    cmap = ListedColormap(["white", "black"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_aspect("equal")
    ax.set_facecolor("white")

    # Start/goal markers (kept simple)
    if start is not None:
        r, c = start
        x = (c - center_cell[1]) * resolution
        y = (r - center_cell[0]) * resolution
        ax.plot(x, y, marker="o")

    if goal is not None:
        r, c = goal
        x = (c - center_cell[1]) * resolution
        y = (r - center_cell[0]) * resolution
        ax.plot(x, y, marker="x")

    # Light gray gridlines with transparency
    if show_gridlines:
        xmin, xmax, ymin, ymax = extent
        # cell boundaries are spaced by `resolution`
        x_lines = np.arange(xmin, xmax + 1e-9, resolution)
        y_lines = np.arange(ymin, ymax + 1e-9, resolution)

        for x in x_lines:
            ax.axvline(x, linewidth=0.25, alpha=0.05)
        for y in y_lines:
            ax.axhline(y, linewidth=0.25, alpha=0.05)


def draw_path(ax: plt.Axes, path: Path2D) -> None:
    xs = [p[0] for p in path.points]
    ys = [p[1] for p in path.points]
    ax.plot(xs, ys, color="blue", linewidth=2.0)
