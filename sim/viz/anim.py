from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Tuple, List, Optional, Dict, Any, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Patch
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D

from sim.types import Pose2D, VelocityCommand2D, Path2D
from sim.viz.draw import draw_grid, draw_path
from sim.logging.csv_logger import CsvLogger


@dataclass
class VizConfig:
    robot_L: float = 2.0
    robot_W: float = 1.0
    heading_scale: float = 2.0
    heading_lw: float = 0.25
    trail_lw: float = 1.0
    trail_alpha: float = 0.8
    title: str = "Sim"
    max_steps: int = 10_000
    show_legend: bool = True
    detailed: bool = False

    # perf: only draw last N points of trail
    max_trail_points: int = 2000


def _world_bounds(world) -> Tuple[float, float, float, float]:
    H, W = world.grid.shape
    r0, c0 = world.center_cell
    res = world.resolution

    xmin = (0 - c0) * res - 0.5 * res
    xmax = (W - 1 - c0) * res + 0.5 * res
    ymin = (0 - r0) * res - 0.5 * res
    ymax = (H - 1 - r0) * res + 0.5 * res
    return xmin, xmax, ymin, ymax


StepOut3 = Tuple[VelocityCommand2D, Pose2D, bool]
StepOut4 = Tuple[VelocityCommand2D, Pose2D, bool, Dict[str, Any]]
StepOut = Union[StepOut3, StepOut4]


def run_loop(
    *,
    world,
    path: Path2D,
    pose0: Pose2D,
    dt: float,
    step_fn: Callable[[Pose2D], StepOut],
    viz: VizConfig = VizConfig(),
    log_csv: Optional[str] = None,
    log_flush_every: int = 200,
) -> None:
    pose = Pose2D(pose0.x, pose0.y, pose0.yaw)
    xmin, xmax, ymin, ymax = _world_bounds(world)

    def in_bounds(p: Pose2D) -> bool:
        return (xmin <= p.x <= xmax) and (ymin <= p.y <= ymax)

    #Logging
    logger: Optional[CsvLogger] = None
    sim_t: float = 0.0
    if log_csv is not None:
        logger = CsvLogger(log_csv, flush_every=log_flush_every)

    fig, ax = plt.subplots()

    draw_grid(
        ax,
        world.grid,
        world.start,
        world.goal,
        resolution=world.resolution,
        center_cell=world.center_cell,
    )
    draw_path(ax, path)

    robot_rect = Rectangle(
        (-viz.robot_L / 2, -viz.robot_W / 2),
        viz.robot_L,
        viz.robot_W,
        linewidth=1.5,
        fill=False,
    )
    ax.add_patch(robot_rect)

    heading_arrow = FancyArrowPatch(
        (pose.x, pose.y),
        (pose.x + 1.0 * math.cos(pose.yaw), pose.y + 1.0 * math.sin(pose.yaw)),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=viz.heading_lw,
        color="red",
    )
    ax.add_patch(heading_arrow)

    (robot_center_ln,) = ax.plot([], [], marker="o", linestyle="", markersize=4, color="black")

    # Force trail to black so it matches legend
    (trail_ln,) = ax.plot([], [], linewidth=viz.trail_lw, alpha=viz.trail_alpha, color="black")

    # Detailed overlays
    lookahead_ln = None
    lookahead_circle = None
    lookahead_link_ln = None
    debug_text = None

    if viz.detailed:
        (lookahead_ln,) = ax.plot(
            [], [],
            marker="x",
            linestyle="",
            markersize=8,
            markeredgewidth=2.0,
            color="red",
        )

        lookahead_circle = Circle(
            (pose.x, pose.y),
            radius=1.0,
            fill=False,
            linestyle=":",
            linewidth=1.0,
            alpha=0.6,
        )
        ax.add_patch(lookahead_circle)

        # dotted link robot -> lookahead
        (lookahead_link_ln,) = ax.plot(
            [], [],
            linestyle=":",
            linewidth=1.0,
            alpha=0.35,
            color="black",
        )

        debug_text = ax.text(
            0.01, 0.99, "",
            transform=ax.transAxes,
            va="top", ha="left",
        )

    # Legend proxies so shapes look right
    if viz.show_legend:
        handles: List[Any] = [
            Patch(facecolor="none", edgecolor="black", label="Robot"),
            Line2D([0], [0], marker=r"$\rightarrow$", linestyle="None", color="red", markersize=12, label="Heading"),
            Line2D([0], [0], marker="o", linestyle="None", color="black", markersize=5, label="Robot center"),
            Line2D([0], [0], linestyle="-", color="black", linewidth=viz.trail_lw, alpha=viz.trail_alpha, label="Trail"),
        ]
        if viz.detailed:
            handles += [
                Line2D([0], [0], marker="x", linestyle="None", color="red", markersize=8, label="Lookahead pL"),
                Line2D([0], [0], linestyle=":", color="black", alpha=0.6, label="Lookahead circle"),
                Line2D([0], [0], linestyle=":", color="black", alpha=0.35, label="Robot → pL"),
            ]
        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.33, 1))

    ax.set_title(viz.title)

    plt.ion()
    plt.show()

    # Pause / stepping state
    paused = {"v": False}
    step_once = {"v": False}
    view_idx = {"v": None}  # history index when paused

    # History of states for browsing (pose + debug for that pose)
    history: List[Tuple[Pose2D, Optional[Dict[str, Any]]]] = [(Pose2D(pose.x, pose.y, pose.yaw), None)]

    # Trail buffers for fast live rendering
    trail_x: List[float] = [pose.x]
    trail_y: List[float] = [pose.y]

    def on_key(event):
        key = event.key

        # Pause toggle
        if key in (" ", "p", "v"):
            paused["v"] = not paused["v"]
            if paused["v"]:
                view_idx["v"] = len(history) - 1
            else:
                view_idx["v"] = None

        # Quit
        elif key in ("q", "escape"):
            plt.close(fig)

        # While paused: navigate / step
        elif paused["v"]:
            if key in ("right",):
                # if not at latest, just advance view; else step sim once
                if view_idx["v"] is not None and view_idx["v"] < len(history) - 1:
                    view_idx["v"] += 1
                else:
                    step_once["v"] = True

            elif key in ("left",):
                if view_idx["v"] is not None:
                    view_idx["v"] = max(0, view_idx["v"] - 1)

            elif key in ("n",):
                step_once["v"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    def _set_trail_for_view(idx: int) -> None:
        # show at most last max_trail_points
        start = max(0, idx + 1 - viz.max_trail_points)
        xs = [history[k][0].x for k in range(start, idx + 1)]
        ys = [history[k][0].y for k in range(start, idx + 1)]
        trail_ln.set_data(xs, ys)

    def _set_trail_live() -> None:
        # show last max_trail_points from live buffers
        if viz.max_trail_points > 0 and len(trail_x) > viz.max_trail_points:
            xs = trail_x[-viz.max_trail_points:]
            ys = trail_y[-viz.max_trail_points:]
        else:
            xs, ys = trail_x, trail_y
        trail_ln.set_data(xs, ys)

    def render(p: Pose2D, debug: Optional[Dict[str, Any]] = None, *, extra_text: str = "") -> None:
        # footprint transform
        T = Affine2D().rotate(p.yaw).translate(p.x, p.y) + ax.transData
        robot_rect.set_transform(T)

        # heading
        Lh = viz.heading_scale * viz.robot_L
        hx = p.x + Lh * math.cos(p.yaw)
        hy = p.y + Lh * math.sin(p.yaw)
        heading_arrow.set_positions((p.x, p.y), (hx, hy))

        # center dot
        robot_center_ln.set_data([p.x], [p.y])

        # detailed overlays
        if viz.detailed:
            pL = debug.get("lookahead_world") if (debug is not None) else None
            rad = debug.get("lookahead_L") if (debug is not None) else None
            _idx = debug.get("progress_idx") if (debug is not None) else None

            if lookahead_ln is not None and pL is not None:
                lookahead_ln.set_data([pL[0]], [pL[1]])

            if lookahead_circle is not None and rad is not None:
                lookahead_circle.center = (p.x, p.y)
                lookahead_circle.radius = float(rad)

            if lookahead_link_ln is not None and pL is not None:
                lookahead_link_ln.set_data([p.x, pL[0]], [p.y, pL[1]])

            if debug_text is not None:
                mode = "PAUSED" if paused["v"] else "RUNNING"
                # debug_text.set_text(f"{mode}  {extra_text}\n(space/p/v pause, ←/→ browse/step, n step, q quit)")

        fig.canvas.draw_idle()

    # initial draw
    _set_trail_live()
    render(pose, debug=None, extra_text="(click plot, then press space)")
    plt.pause(0.001)

    try:
        # main loop
        for _k in range(viz.max_steps):
            if not plt.fignum_exists(fig.number):
                return

            # paused browse (no sim)
            if paused["v"] and not step_once["v"]:
                if history and view_idx["v"] is not None:
                    hp, hdbg = history[view_idx["v"]]
                    _set_trail_for_view(view_idx["v"])
                    render(hp, hdbg or {}, extra_text=f"(frame {view_idx['v']+1}/{len(history)})")
                plt.pause(0.05)
                continue

            # if paused but requested one sim step
            if step_once["v"]:
                step_once["v"] = False
                view_idx["v"] = len(history) - 1

            # measure controller/step function wall time only
            t0 = time.perf_counter()

            # simulate one step from current pose
            out = step_fn(pose)

            wall_step_s = time.perf_counter() - t0

            if len(out) == 3:
                cmd, new_pose, done = out  # type: ignore[misc]
                debug = None
            else:
                cmd, new_pose, done, debug = out  # type: ignore[misc]

            # ptional CSV logging (one row per sim tick) ---
            if logger is not None:
                row: Dict[str, Any] = {
                    "sim_t": sim_t,
                    "dt": dt,
                    "x": pose.x,
                    "y": pose.y,
                    "yaw": pose.yaw,
                    "v_cmd": cmd.v,
                    "omega_cmd": cmd.omega,
                    "done": int(bool(done)),
                    "wall_step_s": wall_step_s,
                    "overrun": int(wall_step_s > dt),
                }

                if debug:
                    if "lookahead_world" in debug:
                        xLw, yLw = debug["lookahead_world"]
                        row["xLw"] = xLw
                        row["yLw"] = yLw
                    if "progress_idx" in debug:
                        row["progress_idx"] = int(debug["progress_idx"])
                    if "kappa" in debug:
                        row["kappa"] = float(debug["kappa"])
                    if "L_used" in debug:
                        row["L_used"] = float(debug["L_used"])

                logger.log(row)

            # attach debug to current history state (pose used for control)
            history[-1] = (Pose2D(pose.x, pose.y, pose.yaw), debug)

            # render at pose used to compute debug
            _set_trail_live()
            render(pose, debug or {}, extra_text=f"(frame {len(history)})")

            # try to maintain real-time: sleep only what's left after drawing
            elapsed_draw = time.perf_counter() - t0  # includes step + some render overhead
            plt.pause(max(0.001, dt - elapsed_draw))

            # advance state (and append new pose to history/trail buffers)
            pose = new_pose
            history.append((Pose2D(pose.x, pose.y, pose.yaw), None))
            trail_x.append(pose.x)
            trail_y.append(pose.y)

            # advance simulated time (exactly dt per step)
            sim_t += dt

            # stop checks
            if not in_bounds(pose):
                ax.set_title("Robot left world bounds — stopped.")
                _set_trail_live()
                render(pose, debug or {}, extra_text="(stopped)")
                plt.pause(0.001)
                break

            if done:
                ax.set_title("Done — stopped.")
                _set_trail_live()
                render(pose, debug or {}, extra_text="(done)")
                plt.pause(0.001)
                break

            if paused["v"]:
                view_idx["v"] = len(history) - 1

    finally:
        if logger is not None:
            logger.close()

    plt.ioff()
    plt.show()
