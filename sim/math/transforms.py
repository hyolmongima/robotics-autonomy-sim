from __future__ import annotations

from typing import Tuple
import math
import numpy as np

from sim.types import Pose2D, Point2


def world_to_body(p_world: Point2, pose_world: Pose2D) -> Point2:
    """
    Transform a 2D point from world frame into robot/body frame.
    Body frame convention: +x forward, +y left.
    pose_world is robot pose in world: (x,y,yaw).
    """
    pw = np.asarray(p_world, dtype=float)
    t = np.asarray([pose_world.x, pose_world.y], dtype=float)

    dx = pw - t
    c = math.cos(pose_world.yaw)
    s = math.sin(pose_world.yaw)


    xb =  c * dx[0] + s * dx[1]
    yb = -s * dx[0] + c * dx[1]
    return (float(xb), float(yb))


def body_to_world(p_body: Point2, pose_world: Pose2D) -> Point2:
    """
    Transform a 2D point from robot/body frame into world frame.
    Inverse of world_to_body.
    """
    pb = np.asarray(p_body, dtype=float)
    t = np.asarray([pose_world.x, pose_world.y], dtype=float)

    c = math.cos(pose_world.yaw)
    s = math.sin(pose_world.yaw)

    xw = c * pb[0] - s * pb[1] + t[0]
    yw = s * pb[0] + c * pb[1] + t[1]
    return (float(xw), float(yw))
