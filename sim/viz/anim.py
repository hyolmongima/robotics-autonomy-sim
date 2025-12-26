# sim/viz/anim.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple, List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.transforms import Affine2D

from sim.types import Pose2D, VelocityCommand2D, Path2D
from sim.viz.draw import draw_grid, draw_path


@dataclass
class VizConfig:
    robot_L: float = 3.0
    robot_W: float = 2.0
    heading_scale: float = 2.0      # arrow length = heading_scale * robot_L
    heading_lw: float = 0.25
    trail_lw: float = 1.0
    trail_alpha: float = 0.8
    title: str = "Sim"
    max_steps: int = 10_000         # safety stop
    show_legend: bool = True


def _world_bounds(world) -> Tuple[float, float, float, float]:
    H, W = world.grid.shape
    r0, c0 = world.center_cell
    res = world.resolution

    xmin = (0 - c0) * res - 0.5 * res
    xmax = (W - 1 - c0) * res + 0.5 * res
    ymin = (0 - r0) * res - 0.5 * res
    ymax = (H - 1 - r0) * res + 0.5 * res
    return xmin, xmax, ymin, ymax


def run_loop( *, world, path: Path2D, pose0: Pose2D, dt: float, step_fn: Callable[[Pose2D], Tuple[VelocityCommand2D, Pose2D, bool]], viz: VizConfig = VizConfig()) -> None:
    
    pose = Pose2D(pose0.x, pose0.y, pose0.yaw)
    xmin, xmax, ymin, ymax = _world_bounds(world)
    def in_bounds(p: Pose2D) -> bool:
        return (xmin <= p.x <= xmax) and (ymin <= p.y <= ymax)

    # --- Setup figure ---
    fig, ax = plt.subplots()
    # Static layers
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
        label="Robot",
    )
    ax.add_patch(robot_rect)

    heading_arrow = FancyArrowPatch(
        (pose.x, pose.y),
        (pose.x + 1.0 * math.cos(pose.yaw), pose.y + 1.0 * math.sin(pose.yaw)),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=viz.heading_lw,
        color="red",
        label="Heading",
    )
    ax.add_patch(heading_arrow)

    (trail_ln,) = ax.plot([], [], linewidth=viz.trail_lw, alpha=viz.trail_alpha, label="Trail")

    if viz.show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.33, 1))

    ax.set_title(viz.title)

    trail_x: List[float] = []
    trail_y: List[float] = []

    # Use interactive mode so the while-loop can update the window
    plt.ion()
    plt.show()

    def render(p: Pose2D) -> None:
        # Robot rectangle: rotate about center then translate
        T = Affine2D().rotate(p.yaw).translate(p.x, p.y) + ax.transData
        robot_rect.set_transform(T)

        # Heading arrow
        L = viz.heading_scale * viz.robot_L
        hx = p.x + L * math.cos(p.yaw)
        hy = p.y + L * math.sin(p.yaw)
        heading_arrow.set_positions((p.x, p.y), (hx, hy))

        # Trail
        trail_x.append(p.x)
        trail_y.append(p.y)
        trail_ln.set_data(trail_x, trail_y)

        fig.canvas.draw_idle()

    # Render initial pose once
    render(pose)
    plt.pause(0.001)

    # --- Main loop ---
    for k in range(viz.max_steps):
        if not plt.fignum_exists(fig.number):
            # user closed the window
            return

        cmd, new_pose, done = step_fn(pose)
        pose = new_pose

        if not in_bounds(pose):
            ax.set_title("Robot left world bounds — stopped.")
            render(pose)
            plt.pause(0.001)
            break

        if done:
            ax.set_title("Done — stopped.")
            render(pose)
            plt.pause(0.001)
            break

        render(pose)
        plt.pause(dt)

    # Keep window open if script reaches here (optional)
    plt.ioff()
    plt.show()
