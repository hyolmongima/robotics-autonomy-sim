from __future__ import annotations

import math
from typing import List, Tuple, Optional, Callable

from sim.types import Pose2D, VelocityCommand2D, Point2
from sim.math.geometry import project_point_to_segment, circle_segment_intersections
from sim.math.transforms import world_to_body


class PurePursuitController:
    """
    Pure Pursuit path-tracking controller for a 2D polyline path.

    Canonical geometry:
      - Lookahead point pL is chosen on the path such that ||pL - p_robot|| = L (circle-path intersection).
      - Express pL in body frame (xL, yL).
      - Curvature: kappa = 2*yL / L^2   (valid when pL is in front and at distance L)
      - Differential-drive mapping: omega = v * kappa

    Policy (actuation) is kept separate and centralized in _cmd_from_lookahead_body():
      - Rotate-in-place fallback if lookahead is behind / nearly lateral.
      - Speed policy hook: get_velocity(kappa) -> v (optional)
      - Saturations (v, omega)
    """

    # Small epsilons for numeric stability & jitter reduction
    _X_AHEAD_EPS: float = 1e-3
    _KAPPA_EPS: float = 1e-6

    def __init__(
        self,
        path_points: List[Point2],
        lookahead_L: float,
        window_W: int = 40,
        eps_back: int = 2,
    ):
        """
        Args:
          path_points: List of (x,y) points defining the path polyline in world frame.
          lookahead_L: Lookahead distance (circle radius) in meters. Must be > 0.
          window_W: Number of segments ahead of progress_idx to search for intersections.
          eps_back: Number of segments behind progress_idx to include when updating progress.
        """
        if lookahead_L <= 0.0:
            raise ValueError(f"lookahead_L must be > 0, got {lookahead_L}")

        self.P: List[Point2] = list(path_points)
        self.L: float = float(lookahead_L)
        self.W: int = int(window_W)
        self.eps_back: int = int(eps_back)
        self.progress_idx: int = 0

    def reset(self, pose: Pose2D) -> None:
        """
        Reset internal progress state for a new run or replanned path.
        """
        if len(self.P) < 2:
            self.progress_idx = 0
            return

        bot_xy = (pose.x, pose.y)
        self.progress_idx = self._update_progress_index(bot_xy, i0=0, i1=len(self.P) - 2)

    def step(
        self,
        pose: Pose2D,
        *,
        max_vel: float,
        max_yawrate: float,
        get_velocity: Optional[Callable[[float], float]] = None,
    ) -> Tuple[VelocityCommand2D, Point2]:
        """
        Compute one Pure Pursuit control update.

        Args:
          pose: Pose2D robot pose in world frame at the current timestep.
          max_vel: Upper bound on commanded forward speed (m/s). Must be >= 0.
          max_yawrate: Upper bound on commanded angular speed (rad/s). Must be >= 0.
          get_velocity: Optional speed policy returning v_cmd from curvature:
                        v_cmd = get_velocity(kappa).
                        If None, uses default curvature-limited policy.

        Returns:
          cmd: VelocityCommand2D(v, omega)
          pL_world: lookahead point in world frame (for viz/debug)
        """
        if max_vel < 0.0:
            raise ValueError(f"max_vel must be >= 0, got {max_vel}")
        if max_yawrate < 0.0:
            raise ValueError(f"max_yawrate must be >= 0, got {max_yawrate}")

        if len(self.P) < 2:
            return VelocityCommand2D(v=0.0, omega=0.0), (pose.x, pose.y)

        # 1) Find lookahead point in world frame
        pL_world = self._find_lookahead_point(pose)

        # 2) Express lookahead point in body frame
        xL, yL = world_to_body(pL_world, pose)

        # 3) Centralized policy: handle rotate-in-place, speed selection, clamping
        cmd = self._cmd_from_lookahead_body(
            xL=xL,
            yL=yL,
            max_vel=max_vel,
            max_yawrate=max_yawrate,
            get_velocity=get_velocity,
        )

        return cmd, pL_world

    # ----------------------------
    # Policy / actuation
    # ----------------------------

    def _cmd_from_lookahead_body(
        self,
        *,
        xL: float,
        yL: float,
        max_vel: float,
        max_yawrate: float,
        get_velocity: Optional[Callable[[float], float]],
    ) -> VelocityCommand2D:
        """
        Convert (xL, yL) in body frame into (v, omega), including edge-case handling.

        - Rotate-in-place if lookahead is behind (xL <= eps) to face the lookahead.
        - Otherwise compute canonical curvature and apply speed policy.
        - Clamp (v, omega) to limits.
        """

        # Canonical PP curvature
        kappa = 2.0 * yL / (self.L * self.L)

        # Choose forward speed
        if max_vel <= 0.0:
            v = 0.0
        elif get_velocity is None:
            v = self._default_velocity_policy(kappa, max_vel=max_vel, max_yawrate=max_yawrate)
        else:
            v = float(get_velocity(kappa))

        # Enforce v bounds 
        v = float(max(0.0, min(max_vel, v)))

        # Convert to omega and clamp
        omega = float(v * kappa)
        omega = float(max(-max_yawrate, min(max_yawrate, omega)))

        return VelocityCommand2D(v=v, omega=omega)

    def _default_velocity_policy(self, kappa: float, *, max_vel: float, max_yawrate: float) -> float:
        """
        Default policy: choose v so that |omega| = |v*kappa| <= max_yawrate.
        If max_yawrate == 0, returns 0 (cannot turn).
        """
        if max_yawrate <= 0.0:
            return 0.0
        return float(min(max_vel, max_yawrate / (abs(kappa) + self._KAPPA_EPS)))

    # ----------------------------
    # Core PP geometry / search
    # ----------------------------

    def _update_progress_index(self, bot_xy: Point2, *, i0: int, i1: int) -> int:
        """
        Closest segment index to bot_xy over segments [i0..i1] using projection distance.
        """
        if len(self.P) < 2:
            return 0

        i0 = max(0, int(i0))
        i1 = min(len(self.P) - 2, int(i1))
        if i0 > i1:
            i0, i1 = i1, i0

        best_i = i0
        best_d = float("inf")
        for i in range(i0, i1 + 1):
            proj = project_point_to_segment(bot_xy, self.P[i], self.P[i + 1])
            if proj.dist < best_d:
                best_d = proj.dist
                best_i = i
        return best_i

    def _pick_ahead_hit(self, hits: List[Point2], pose: Pose2D) -> Optional[Point2]:
        """
        Pick an intersection point that is ahead of the robot (x_body > eps).
        If two are ahead, prefer the one with larger x_body (further forward).
        """
        if not hits:
            return None

        best_p: Optional[Point2] = None
        best_x = -float("inf")

        for pW in hits:
            xB, _yB = world_to_body(pW, pose)
            if xB > self._X_AHEAD_EPS and xB > best_x:
                best_x = xB
                best_p = pW

        return best_p

    def _find_lookahead_point(self, pose: Pose2D) -> Point2:
        """
        Find the lookahead point on the polyline path.

        Strategy:
          - Update progress_idx locally within a window around current progress.
          - Scan forward from progress_idx for circle-segment intersections.
          - If none found:
              - near end: return final waypoint (stable goal capture)
              - else: return projection onto current segment
        """
        bot_xy = (pose.x, pose.y)
        N = len(self.P)
        if N < 2:
            return bot_xy

        # Update progress index within a bounded local window
        i0 = max(0, self.progress_idx - self.eps_back)
        i1 = min(N - 2, self.progress_idx + self.W)
        candidate_idx = self._update_progress_index(bot_xy, i0=i0, i1=i1)

        # Monotonic progress guard (prevents snapping backward)
        self.progress_idx = max(self.progress_idx, candidate_idx)

        # Recompute upper bound after possibly advancing progress
        i1 = min(N - 2, self.progress_idx + self.W)

        # Scan forward for first valid intersection ahead
        for i in range(self.progress_idx, i1 + 1):
            hits = circle_segment_intersections(bot_xy, self.L, self.P[i], self.P[i + 1])
            p_star = self._pick_ahead_hit(hits, pose)
            if p_star is not None:
                return p_star

        # No valid circle intersection: stable fallback choices
        if self.progress_idx >= (N - 3):
            # Near end-of-path: chase final waypoint
            return self.P[-1]

        # Otherwise: project to the current segment
        proj = project_point_to_segment(bot_xy, self.P[self.progress_idx], self.P[self.progress_idx + 1])
        return proj.p
