from __future__ import annotations
import math
import argparse

from sim.map_grid import make_demo_world
from sim.planners.dummy_planner import fakeplan_squiggly
from sim.types import Pose2D, VelocityCommand2D
from sim.plant import update_kinematics
from sim.viz.anim import run_loop, VizConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--v", type=float, default=10.0)
    p.add_argument("--omega", type=float, default=math.pi / 10)  # rad/s
    return p.parse_args()


def main():
    args = parse_args() #parse arguments
    pose0 = Pose2D(x=-40, y=-40, yaw=0) #get bot's spawn pose
    world = make_demo_world() #get grid world with start and end cordinates
    
    #Plan path once for now
    path = fakeplan_squiggly(world) #plan path (replaced by A* in MVP implementation)

    #each step is simply, pure pursuit update and plant state update (same frequency)
    def step_fn(pose: Pose2D):
        cmd = VelocityCommand2D(v=args.v, omega=args.omega) #Controller (replaced by pure pursuit in MVP implementation)
        new_pose = update_kinematics(pose, cmd, args.dt)
        new_velocity = cmd #current velocity is commanded velocity in ideal sim world
        done = False #We will write logic to end the physics in MVP step
        return new_velocity, new_pose, done

    run_loop(world=world, path=path, pose0=pose0, dt=args.dt, step_fn=step_fn, viz=VizConfig(title="Basic setup"))


if __name__ == "__main__":
    main()
