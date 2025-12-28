# scripts/run_simulation.py
from __future__ import annotations

import math
import argparse

from sim.map_grid import make_demo_world
from sim.planners.dummy_planner import fakeplan_squiggly
from sim.controllers.pure_pursuit import PurePursuitController
from sim.types import Pose2D
from sim.viz.anim import run_loop, VizConfig
from sim.sim_loop import make_step_fn_pure_pursuit


def parse_args():
    p = argparse.ArgumentParser(
        description="Run simulation loop with Pure Pursuit tracker + ideal kinematic plant (planner is dummy for now).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # user-facing knobs
    p.add_argument(
        "--visualize_detailed",
        action="store_true",
        help="Show lookahead geometry and debug overlays.",
    )
    p.add_argument(
        "--log_csv",
        type=str,
        default=None,
        help="If set, write a CSV log to this path (e.g., logs/pp_run.csv).",
    )

    # intentionally suppressed knobs (advanced)
    p.add_argument("--dt", type=float, default=0.05, help=argparse.SUPPRESS)
    p.add_argument("--max_vel", type=float, default=20.0, help=argparse.SUPPRESS)
    p.add_argument("--max_yawrate", type=float, default=1.5, help=argparse.SUPPRESS)
    p.add_argument("--L", type=float, default=3.0, help=argparse.SUPPRESS)

    # endpoint handling (semi-advanced; keep visible if you want)
    p.add_argument("--goal_pos_tol", type=float, default=1.0, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()

    # initial state
    pose0 = Pose2D(x=+30, y=+30, yaw=math.pi/2)

    # world + path
    world = make_demo_world()
    path = fakeplan_squiggly(world)

    # controller
    pp = PurePursuitController(path_points=path.points, lookahead_L=args.L)
    pp.reset(pose0)

    # step function (closed-loop wiring + endpoint policy)
    step_fn = make_step_fn_pure_pursuit(
        path=path,
        controller=pp,
        dt=args.dt,
        max_vel=args.max_vel,
        max_yawrate=args.max_yawrate,
        lookahead_L=args.L,
        visualize_detailed=args.visualize_detailed,
        goal_pos_tol=args.goal_pos_tol,
    )

    # run loop (physics + rendering coupled)
    run_loop(
        world=world,
        path=path,
        pose0=pose0,
        dt=args.dt,
        step_fn=step_fn,
        viz=VizConfig(title="Pure Pursuit", detailed=args.visualize_detailed),
        log_csv=args.log_csv,
    )


if __name__ == "__main__":
    main()
