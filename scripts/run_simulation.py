from __future__ import annotations

import math
import argparse

from sim.map_grid import make_demo_world
from sim.planners.dummy_planner import fakeplan_squiggly
from sim.controllers.pure_pursuit import PurePursuitController
from sim.types import Pose2D
from sim.viz.anim import run_loop, VizConfig

from sim.sim_loop import make_step_fn_pure_pursuit
from sim.actuation.actuator import ActuatorConfig
from sim.sim_loop import ActuationPolicy


def parse_args():
    p = argparse.ArgumentParser(
        description="Pure Pursuit demo (actuator enforces hard limits).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--visualize_detailed", action="store_true",
                   help="Show lookahead geometry and debug overlays.")
    p.add_argument("--log_csv", type=str, default=None,
                   help="If set, write a CSV log to this path (e.g., logs/pp_run.csv).")
    
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--L", type=float, default=3.0, help="Lookahead distance (m).")
    p.add_argument("--v_nom", type=float, default=10.0, help="Desired forward speed before actuator limits (m/s).")

    # actuator “physics” knobs 
    p.add_argument("--act_max_v", type=float, default=20.0)
    p.add_argument("--act_max_omega", type=float, default=3.0)
    p.add_argument("--act_max_v_accel", type=float, default=5.0)
    p.add_argument("--act_max_omega_accel", type=float, default=1.5)

    return p.parse_args()


def main():
    args = parse_args()

    pose0 = Pose2D(x=-32, y=5, yaw=math.pi / 2)

    world = make_demo_world()
    path = fakeplan_squiggly(world)

    pp = PurePursuitController(path_points=path.points, lookahead_L=args.L)
    pp.reset(pose0)

    actuation = ActuationPolicy(
        enabled=False,
        cfg=ActuatorConfig(
            max_v=args.act_max_v,
            max_omega=args.act_max_omega,
            max_v_accel=args.act_max_v_accel,
            max_omega_accel=args.act_max_omega_accel,
        ),
    )

    step_fn = make_step_fn_pure_pursuit(
        path=path,
        controller=pp,
        dt=args.dt,
        lookahead_L=args.L,
        v_nom=args.v_nom,
        visualize_detailed=args.visualize_detailed,
        actuation=actuation,
    )

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
