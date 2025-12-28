from __future__ import annotations
import math
from typing import Callable, Optional

from sim.types import Pose2D, VelocityCommand2D
from sim.plant import update_kinematics
from sim.math.transforms import wrap_pi
from sim.controllers.pure_pursuit import PurePursuitController  # or Protocol later

def make_step_fn_pure_pursuit(
    *, 
    path,
    controller: PurePursuitController,
    dt: float,
    max_vel: float,
    max_yawrate: float,
    lookahead_L: float,
    visualize_detailed: bool,
    goal_pos_tol: float = 1.0,
):
    goal_xy = path.points[-1]
    def step_fn(pose: Pose2D):
        dist_goal = math.hypot(pose.x - goal_xy[0], pose.y - goal_xy[1])

        # Goal capture overrides tracker near end
        if dist_goal <= goal_pos_tol:
            
            cmd = VelocityCommand2D(v=0.0, omega=0.0)
            done = True
            if visualize_detailed:
                debug = {"mode": "done_stop", "dist_goal": dist_goal, "progress_idx": controller.progress_idx}
                return cmd, pose, done, debug
            return cmd, pose, done
        
        # Normal tracking
        cmd, pL = controller.step(pose, max_vel=max_vel, max_yawrate=max_yawrate)
        new_pose = update_kinematics(pose, cmd, dt)
        done = False

        if visualize_detailed:
            debug = {
                "mode": "track",
                "lookahead_world": pL,
                "lookahead_L": lookahead_L,
                "progress_idx": controller.progress_idx,
                "dist_goal": dist_goal,
            }
            return cmd, new_pose, done, debug

        return cmd, new_pose, done

    return step_fn
