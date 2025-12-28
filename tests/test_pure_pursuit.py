import math

from sim.types import Pose2D
from sim.controllers.pure_pursuit import PurePursuitController


def test_progress_index_global_init_straight_line():
    # straight path along +x
    P = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    pp = PurePursuitController(P, lookahead_L=5.0, window_W=10, eps_back=2)

    # robot near second segment
    pose = Pose2D(x=15.0, y=2.0, yaw=0.0)
    pp.reset(pose)

    # segment indices: 0 is [0->10], 1 is [10->20]
    assert pp.progress_idx == 1


def test_find_lookahead_returns_circle_hit_ahead():
    # path is x-axis; robot at origin facing +x; lookahead r=5 should hit (5,0)
    P = [(-10.0, 0.0), (10.0, 0.0)]
    pp = PurePursuitController(P, lookahead_L=5.0, window_W=5, eps_back=1)
    pp.reset(Pose2D(x=0.0, y=0.0, yaw=0.0))

    pL = pp._find_lookahead_point(Pose2D(x=0.0, y=0.0, yaw=0.0))
    assert abs(pL[0] - 5.0) < 1e-6
    assert abs(pL[1] - 0.0) < 1e-6


def test_step_straight_path_zero_omega():
    # if lookahead is straight ahead (yL=0) omega should be ~0
    P = [(-10.0, 0.0), (10.0, 0.0)]
    pp = PurePursuitController(P, lookahead_L=5.0, window_W=5, eps_back=1)
    pp.reset(Pose2D(x=0.0, y=0.0, yaw=0.0))

    cmd, pL = pp.step(Pose2D(x=0.0, y=0.0, yaw=0.0), max_vel=3.0)
    assert abs(cmd.omega) < 1e-9
    assert abs(cmd.v - 3.0) < 1e-9


def test_step_left_of_path_turns_toward_path():
    # path is x-axis, robot above it at y=+1 facing +x
    # lookahead should be ahead but below in body frame (yL negative), so omega should be negative
    P = [(-10.0, 0.0), (10.0, 0.0)]
    pp = PurePursuitController(P, lookahead_L=5.0, window_W=10, eps_back=2)
    pp.reset(Pose2D(x=0.0, y=1.0, yaw=0.0))

    cmd, _ = pp.step(Pose2D(x=0.0, y=1.0, yaw=0.0), max_vel=2.0)
    assert cmd.omega < 0.0
