import math
from typing import List, Tuple
from sim.types import Path2D
from sim.map_grid import GridWorld, world_to_cell, cell_to_world, Cell

Point2D = Tuple[float, float]

def _manhattan_cells(a: Cell, b: Cell) -> List[Cell]:
    """4-connected segment from a to b (row then col)."""
    r0, c0 = a
    r1, c1 = b
    out: List[Cell] = []

    step_r = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step_r, step_r):
        out.append((r, c0))

    step_c = 1 if c1 >= c0 else -1
    for c in range(c0, c1 + step_c, step_c):
        out.append((r1, c))

    # dedup while preserving order
    dedup: List[Cell] = []
    seen = set()
    for cell in out:
        if cell not in seen:
            dedup.append(cell)
            seen.add(cell)
    return dedup


def _polyline_cells_from_world_waypoints(world: GridWorld, wps_world: List[Point2D]) -> List[Cell]:
    """Connect each consecutive waypoint with a Manhattan cell segment."""
    cells: List[Cell] = []
    for i in range(len(wps_world) - 1):
        a = world_to_cell(wps_world[i], resolution=world.resolution, center_cell=world.center_cell)
        b = world_to_cell(wps_world[i + 1], resolution=world.resolution, center_cell=world.center_cell)
        seg = _manhattan_cells(a, b)
        if i > 0 and len(seg) > 0:
            seg = seg[1:]  # avoid repeating joint
        cells.extend(seg)
    return cells


def smooth_curve(points: List[Point2D], *, corner_radius: float = 10.0, samples_per_corner: int = 20) -> List[Point2D]:
    """
    Corner-rounding smoother for a polyline.

    - points: polyline in world meters
    - corner_radius: how far to pull back from each corner along segments
    - samples_per_corner: number of points to insert around each corner

    Works best when waypoints are axis-aligned (like yours).
    """
    if len(points) < 3:
        return points[:]

    def unit(vx, vy):
        n = math.hypot(vx, vy)
        if n < 1e-9:
            return (0.0, 0.0)
        return (vx / n, vy / n)

    out: List[Point2D] = [points[0]]

    for i in range(1, len(points) - 1):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]

        v1 = (p1[0] - p0[0], p1[1] - p0[1])  # incoming
        v2 = (p2[0] - p1[0], p2[1] - p1[1])  # outgoing

        u1 = unit(*v1)
        u2 = unit(*v2)

        # If direction doesn't change (collinear), keep the point
        if abs(u1[0] - u2[0]) < 1e-9 and abs(u1[1] - u2[1]) < 1e-9:
            out.append(p1)
            continue

        # pull back from corner along each segment
        # clamp radius to segment lengths (so we don't overshoot)
        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)
        r = min(corner_radius, 0.9 * len1, 0.9 * len2)

        a = (p1[0] - u1[0] * r, p1[1] - u1[1] * r)  # entry point
        b = (p1[0] + u2[0] * r, p1[1] + u2[1] * r)  # exit point

        # Insert entry point
        out.append(a)

        # Simple interpolation between a -> corner -> b
        # (not perfect arc, but smooth-ish and works well visually)
        for t_idx in range(1, samples_per_corner):
            t = t_idx / samples_per_corner
            # quadratic Bezier: a, p1, b
            x = (1 - t) ** 2 * a[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * b[0]
            y = (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * b[1]
            out.append((x, y))

        # Insert exit point
        out.append(b)

    out.append(points[-1])
    return out

def fakeplan_squiggly(world: GridWorld) -> Path2D:
    # Waypoints in WORLD meters (your requested route)
    wps = [(-40.0, -40.0), (-40.0, 0.0), (0.0, 0.0), (0.0, 40.0), (40.0, 40.0)]

    # Build a cell-by-cell path so later A* integration feels natural
    cells = _polyline_cells_from_world_waypoints(world, wps)

    # Convert to world points at cell centers
    pts = [cell_to_world(c, resolution=world.resolution, center_cell=world.center_cell) for c in cells]

    # Smooth corners (optional)
    pts_smooth = smooth_curve(pts, corner_radius=50.0, samples_per_corner=2)

    return Path2D(points=pts_smooth)
