# sim/plant.py
import math
from sim.types import Pose2D, VelocityCommand2D


def wrap_angle(theta: float) -> float:
    """
    Wrap angle to [-pi, pi).
    """
    theta_wrapped = theta % (2.0 * math.pi)  # in [0, 2pi)
    if theta_wrapped >= math.pi:
        theta_wrapped -= 2.0 * math.pi       # in [-pi, pi)
    return theta_wrapped


def update_kinematics(pose: Pose2D, cmd: VelocityCommand2D, dt: float) -> Pose2D:
    """
    Euler integrate unicycle/diff-drive kinematics for one step.

    pose: current pose (world)
    cmd:  (v, omega) in body frame
    dt:   timestep [s]
    """
    dx = cmd.v * math.cos(pose.yaw) * dt
    dy = cmd.v * math.sin(pose.yaw) * dt
    dyaw = cmd.omega * dt

    return Pose2D(
        x=pose.x + dx,
        y=pose.y + dy,
        yaw=wrap_angle(pose.yaw + dyaw),
    )
