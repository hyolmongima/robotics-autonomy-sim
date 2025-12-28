from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

from sim.types import Pose2D, VelocityCommand2D
from sim.plant import update_kinematics
from sim.math.transforms import wrap_pi

from sim.controllers.pure_pursuit import PurePursuitController, PurePursuitOutput

from sim.actuation.actuator import (
    ActuatorConfig,
    ActuatorState,
    actuator_step,
    reset_actuator_state,
)


def goal_yaw_from_last_segment(points) -> float:
    """Terminal heading from final polyline segment direction (world +x right, +y up)."""
    if points is None or len(points) < 2:
        return 0.0
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    return float(math.atan2(y1 - y0, x1 - x0))


@dataclass(frozen=True)
class GoalPolicy:
    goal_pos_tol: float = 1.0     # meters
    final_align: bool = False     # rotate to terminal yaw before done
    goal_yaw_tol: float = 0.15    # rad


@dataclass(frozen=True)
class ActuationPolicy:
    enabled: bool
    cfg: ActuatorConfig


@dataclass(frozen=True)
class TrackerPolicy:
    """
    Tracker behavior policy (NOT hard limits).
    """
    x_ahead_eps: float = 1e-3
    k_turn_in_place: float = 2.0      # omega_des = k * atan2(yL, xL)


def make_step_fn_pure_pursuit(
    *,
    path: Any,
    controller: PurePursuitController,
    dt: float,
    lookahead_L: float,
    v_nom: float,
    visualize_detailed: bool,
    goal: GoalPolicy = GoalPolicy(),
    actuation: Optional[ActuationPolicy] = None,
    tracker: TrackerPolicy = TrackerPolicy(),
    terminal_yaw_fn: Optional[Callable[[Any], float]] = None,
):
    """
    Returns step_fn(pose) -> (cmd_act, new_pose, done[, debug])

    Pipeline:
      - Goal capture (stop or final align)
      - Pure pursuit geometry: kappa, lookahead
      - Tracker behavior: forms cmd_des (NO hard saturation here)
      - Actuator: acceleration limits + saturation (HARD limits live here)
      - Plant: update_kinematics using cmd_act
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if v_nom < 0.0:
        raise ValueError(f"v_nom must be >= 0, got {v_nom}")
    if path is None or not hasattr(path, "points") or len(path.points) < 1:
        raise ValueError("path must have a non-empty .points attribute")

    goal_xy = path.points[-1]
    goal_yaw = float(terminal_yaw_fn(path)) if terminal_yaw_fn else goal_yaw_from_last_segment(path.points)

    act_state: ActuatorState = reset_actuator_state()

    def _apply_actuation(cmd_des: VelocityCommand2D) -> Tuple[VelocityCommand2D, Dict[str, float]]:
        nonlocal act_state
        if actuation is None or not actuation.enabled:
            return cmd_des, {"act_enabled": 0.0, "cmd_act_v": cmd_des.v, "cmd_act_omega": cmd_des.omega}

        cmd_act, act_state = actuator_step(cmd_des, act_state, dt, actuation.cfg)
        return cmd_act, {"act_enabled": 1.0, "cmd_act_v": cmd_act.v, "cmd_act_omega": cmd_act.omega}

    def _tracker_cmd_from_pp(pp_out: PurePursuitOutput) -> Tuple[VelocityCommand2D, Dict[str, float]]:
        xL, yL = pp_out.lookahead_body
        kappa = float(pp_out.kappa)

        alpha = float(math.atan2(yL, xL))

        # Rotate-in-place if lookahead is behind / nearly lateral
        if xL <= tracker.x_ahead_eps:
            omega_des = float(tracker.k_turn_in_place * alpha)  # actuator will clamp
            cmd = VelocityCommand2D(v=0.0, omega=omega_des)
            dbg = {
                "tracker_mode": 1.0,
                "alpha": alpha,
                "kappa": kappa,
                "cmd_des_v": cmd.v,
                "cmd_des_omega": cmd.omega,
            }
            return cmd, dbg

        # Normal tracking: command nominal speed, omega = v*kappa (actuator clamps)
        v_des = float(v_nom)
        omega_des = float(v_des * kappa)
        cmd = VelocityCommand2D(v=v_des, omega=omega_des)
        dbg = {
            "tracker_mode": 0.0,
            "alpha": alpha,
            "kappa": kappa,
            "cmd_des_v": cmd.v,
            "cmd_des_omega": cmd.omega,
        }
        return cmd, dbg

    def step_fn(pose: Pose2D):
        dist_goal = float(math.hypot(pose.x - goal_xy[0], pose.y - goal_xy[1]))

        # Goal capture
        if dist_goal <= goal.goal_pos_tol:
            if not goal.final_align:
                cmd = VelocityCommand2D(v=0.0, omega=0.0)
                done = True
                if visualize_detailed:
                    debug: Dict[str, float] = _fixed_debug_template()
                    debug.update({
                        "mode": 3.0,
                        "dist_goal": dist_goal,
                        "goal_x": float(goal_xy[0]),
                        "goal_y": float(goal_xy[1]),
                    })
                    return cmd, pose, done, debug
                return cmd, pose, done

            yaw_err = float(wrap_pi(goal_yaw - pose.yaw))
            if abs(yaw_err) <= goal.goal_yaw_tol:
                cmd = VelocityCommand2D(v=0.0, omega=0.0)
                done = True
                if visualize_detailed:
                    debug = _fixed_debug_template()
                    debug.update({
                        "mode": 4.0,
                        "dist_goal": dist_goal,
                        "goal_x": float(goal_xy[0]),
                        "goal_y": float(goal_xy[1]),
                        "yaw_err": yaw_err,
                        "goal_yaw": float(goal_yaw),
                    })
                    return cmd, pose, done, debug
                return cmd, pose, done

            # No clamping here; actuator enforces hard omega limit
            cmd_des = VelocityCommand2D(v=0.0, omega=float(2.0 * yaw_err))
            cmd_act, act_dbg = _apply_actuation(cmd_des)

            new_pose = update_kinematics(pose, cmd_act, dt)
            done = False

            if visualize_detailed:
                debug = _fixed_debug_template()
                debug.update({
                    "mode": 2.0,
                    "dist_goal": dist_goal,
                    "goal_x": float(goal_xy[0]),
                    "goal_y": float(goal_xy[1]),
                    "yaw_err": yaw_err,
                    "goal_yaw": float(goal_yaw),
                    "cmd_des_v": cmd_des.v,
                    "cmd_des_omega": cmd_des.omega,
                    **act_dbg,
                })
                return cmd_act, new_pose, done, debug

            return cmd_act, new_pose, done

        # Pure pursuit geometry
        pp_out = controller.step_curvature(pose)
        pLx, pLy = pp_out.lookahead_world
        xL, yL = pp_out.lookahead_body

        # Tracker -> desired command (no hard limits)
        cmd_des, track_dbg = _tracker_cmd_from_pp(pp_out)

        # Actuator -> achievable command (hard limits)
        cmd_act, act_dbg = _apply_actuation(cmd_des)

        # Plant
        new_pose = update_kinematics(pose, cmd_act, dt)
        done = False

        if visualize_detailed:
            debug = _fixed_debug_template()
            debug.update({
                "mode": 0.0,
                "dist_goal": dist_goal,
                "goal_x": float(goal_xy[0]),
                "goal_y": float(goal_xy[1]),
                "progress_idx": float(getattr(controller, "progress_idx", -1)),
                "pLx": float(pLx),
                "pLy": float(pLy),
                "xL": float(xL),
                "yL": float(yL),
                "L_used": float(lookahead_L),
                **track_dbg,
                **act_dbg,
            })
            return cmd_act, new_pose, done, debug

        return cmd_act, new_pose, done

    return step_fn


def _fixed_debug_template() -> Dict[str, float]:
    nan = float("nan")
    return {
        "mode": nan,
        "dist_goal": nan,
        "goal_x": nan,
        "goal_y": nan,
        "goal_yaw": nan,
        "yaw_err": nan,

        "progress_idx": nan,
        "kappa": nan,
        "alpha": nan,

        "L_used": nan,
        "pLx": nan,
        "pLy": nan,
        "xL": nan,
        "yL": nan,

        "cmd_des_v": nan,
        "cmd_des_omega": nan,

        "act_enabled": nan,
        "cmd_act_v": nan,
        "cmd_act_omega": nan,

        "tracker_mode": nan,
    }
