# sim/controllers/pure_pursuit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from sim.types import Pose2D, Point2
from sim.math.geometry import project_point_to_segment, circle_segment_intersections
from sim.math.transforms import world_to_body


@dataclass(frozen=True)
class PurePursuitOutput:
    kappa: float          # curvature (1/m)
    lookahead_world: Point2
    lookahead_body: Tuple[float, float]  # (xL, yL)


class PurePursuitController:
    """
    Pure Pursuit (geometry-only).

    Computes curvature from a path polyline and robot pose:
      - choose lookahead point pL at distance L on the path (circle-path intersection)
      - transform to body frame (xL, yL)
      - curvature kappa = 2*yL / L^2
    """

    _X_AHEAD_EPS: float = 1e-3

    def __init__(
        self,
        path_points: List[Point2],
        lookahead_L: float,
        window_W: int = 40,
        eps_back: int = 2,
    ):
        if lookahead_L <= 0.0:
            raise ValueError(f"lookahead_L must be > 0, got {lookahead_L}")

        self.P: List[Point2] = list(path_points)
        self.L: float = float(lookahead_L)
        self.W: int = int(window_W)
        self.eps_back: int = int(eps_back)
        self.progress_idx: int = 0

    def reset(self, pose: Pose2D) -> None:
        if len(self.P) < 2:
            self.progress_idx = 0
            return
        bot_xy = (pose.x, pose.y)
        self.progress_idx = self._update_progress_index(bot_xy, i0=0, i1=len(self.P) - 2)

    def step_curvature(self, pose: Pose2D) -> PurePursuitOutput:
        """
        Geometry-only PP step.

        Returns:
          PurePursuitOutput(kappa, lookahead_world, (xL, yL))
        """
        if len(self.P) < 2:
            lookahead = (pose.x, pose.y)
            return PurePursuitOutput(kappa=0.0, lookahead_world=lookahead, lookahead_body=(0.0, 0.0))

        pL_world = self._find_lookahead_point(pose)
        xL, yL = world_to_body(pL_world, pose)

        # Canonical curvature
        kappa = 2.0 * yL / (self.L * self.L)

        return PurePursuitOutput(kappa=float(kappa), lookahead_world=pL_world, lookahead_body=(float(xL), float(yL)))

    # ----------------------------
    # Core PP geometry
    # ----------------------------

    def _update_progress_index(self, bot_xy: Point2, *, i0: int, i1: int) -> int:
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
        bot_xy = (pose.x, pose.y)
        N = len(self.P)
        if N < 2:
            return bot_xy

        i0 = max(0, self.progress_idx - self.eps_back)
        i1 = min(N - 2, self.progress_idx + self.W)
        candidate_idx = self._update_progress_index(bot_xy, i0=i0, i1=i1)

        # monotonic guard
        self.progress_idx = max(self.progress_idx, candidate_idx)

        i1 = min(N - 2, self.progress_idx + self.W)

        for i in range(self.progress_idx, i1 + 1):
            hits = circle_segment_intersections(bot_xy, self.L, self.P[i], self.P[i + 1])
            p_star = self._pick_ahead_hit(hits, pose)
            if p_star is not None:
                return p_star

        # Fallbacks: projection is the cleanest geometry fallback.
        # (Goal capture should be handled by sim_loop, not here.)
        proj = project_point_to_segment(bot_xy, self.P[self.progress_idx], self.P[self.progress_idx + 1])
        return proj.p
