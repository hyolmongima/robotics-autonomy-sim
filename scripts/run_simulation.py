from __future__ import annotations
import math
import argparse

from sim.map_grid import make_demo_world
from sim.planners.dummy_planner import fakeplan_squiggly
from sim.controllers.pure_pursuit import PurePursuitController
from sim.types import Pose2D, VelocityCommand2D
from sim.plant import update_kinematics
from sim.viz.anim import run_loop, VizConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="Run simulation loop with A* planning, Pure Pursuit Tracker and Ideal plant in loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--visualize_detailed", action="store_true",
                   help="Show lookahead geometry and debug overlays.")
    p.add_argument("--log_csv", type=str, default=None,
                   help="If set, write a CSV log to this path (e.g., logs/pp_run.csv).")
    
    p.add_argument("--dt", type=float, default=0.05, help=argparse.SUPPRESS)
    p.add_argument("--max_vel", type=float, default=10.0, help=argparse.SUPPRESS)
    p.add_argument("--max_yawrate", type=float, default=1.5, help=argparse.SUPPRESS)
    p.add_argument("--L", type=float, default=3.0, help=argparse.SUPPRESS)

    return p.parse_args()



def main():
    args = parse_args() #parse arguments
    pose0 = Pose2D(x=-30, y=-20, yaw=math.pi/2) #get bot's spawn pose
    world = make_demo_world() #get grid world with start and end cordinates
    
    #Plan path once for now
    path = fakeplan_squiggly(world) #plan path (replaced by A* in MVP implementation)

    #Initialize Pure Pursuit Controller
    pp = PurePursuitController(path_points=path.points, lookahead_L=args.L)
    pp.reset(pose0)

    #each step is simply pure pursuit update and plant state update (same frequency)
    def step_fn(pose: Pose2D):
        '''
        Does one full sim. step 
        Current setup:
            Pure Pursuit takes current pose to give control inputs in form of bot velocity command in bot frame
            Plant state updates (simple Euler update for pose and velocity is just target vel.- ideal system)
        '''
        vel_target, pL = pp.step(pose, max_vel = args.max_vel, max_yawrate = args.max_yawrate)    #Controller step
        new_pose = update_kinematics(pose, vel_target, args.dt) # Plant update
        
        #stopping condition: if current pose is close to end goal within threshold
        done = math.hypot(pose.x - path.points[-1][0], pose.y - path.points[-1][1]) < 1.0 
        if done:
            return VelocityCommand2D(v=0.0, omega=0.0), pose, True
        
        if args.visualize_detailed:
            debug = {
                "lookahead_world": pL,
                "lookahead_L": args.L,
                "progress_idx": pp.progress_idx,
            }
            return vel_target, new_pose, done, debug
        
        return vel_target, new_pose, done

    #run loop: physics update and rendering coupled unlike physics engines 
    run_loop(
        world=world, path=path, pose0=pose0, dt=args.dt, step_fn=step_fn,
        viz=VizConfig(title="Pure Pursuit", 
        detailed=args.visualize_detailed), 
        log_csv="logs/pp_run.csv"
    )



if __name__ == "__main__":
    main()
