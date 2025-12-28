import math
from typing import List, Tuple, Optional

from sim.types import Pose2D, VelocityCommand2D
from sim.math.geometry import project_point_to_segment, circle_segment_intersections
from sim.math.transforms import world_to_body

Point2 = Tuple[float, float]


class PurePursuitController:
    """
    Pure Pursuit path-tracking controller for a 2D polyline path.

    This controller implements the canonical PurePursuit:
      1) Maintain a progress index into the path (segment index i for segment P[i] -> P[i+1]).
      2) At each update, find a lookahead point pL on the path that lies on a circle of radius L
         centered at the robot position (using circle–segment intersections).
      3) Transform pL into the robot/body frame and compute curvature:
           kappa = 2*yL / L^2
         (valid when pL is exactly L meters from the robot and lies in front of the robot).
      4) Convert curvature to angular velocity for a differential-drive model:
           omega = v_cmd * kappa
    The controller can optionally “own” the linear speed via a velocity policy.
    The default policy is to drive at max_vel.

    """
    def __init__(self, path_points: List[Point2], lookahead_L: float, window_W: int = 40, eps_back: int = 2):
        """
        Args:
          path_points: List of (x,y) points defining the path polyline in world frame.
          lookahead_L: Lookahead distance (circle radius) in meters.
          window_W: Number of segments ahead of progress_idx to search for intersections 
          eps_back: Number of segments behind progress_idx to include when updating progress 
        """
        self.P = path_points
        self.L = float(lookahead_L)
        self.W = int(window_W)
        self.eps_back = int(eps_back)
        self.progress_idx = 0

    def reset(self, pose: Pose2D) -> None:
        """
        Reset the controller’s internal progress state for a new run or replanned path.

        Args:
          pose: Pose2D robot pose in world frame.

        Side effects:
          - Updates self.progress_idx.
        """
        if len(self.P) < 2:
            self.progress_idx = 0
            return
        bot_xy = (pose.x, pose.y)
        self.progress_idx = self._update_progress_index(bot_xy, i0=0, i1=len(self.P) - 2) 

    def step(self, pose: Pose2D, *, max_vel: float, max_yawrate: float) -> Tuple[VelocityCommand2D, Point2]:
        """
        Compute one Pure Pursuit control update.

        Args:
          pose: Pose2D robot pose in world frame at the current timestep.
          max_vel: Upper bound on commanded forward speed (m/s).
          max_yawrate: Upper bound on commanded angular speed (rad/s).
    
        Returns:
          cmd: VelocityCommand2D(v, omega)
          pL_world (Tuple[float,float]): Lookahead point in world frame (for viz/debug)
        """
        if len(self.P) < 2 or self.L <= 0.0: #Guard
            return VelocityCommand2D(v=0.0, omega=0.0), (pose.x, pose.y)
    
        pL_world = self._find_lookahead_point(pose)

        xL, yL = world_to_body(pL_world, pose)

        # rotate-in-place fallback if lookahead point ends up behind the bot in bot frame
        if xL <= 0.0:
            omega = 2.0 * math.atan2(yL, xL)
            return VelocityCommand2D(v=0.0, omega=float(omega)), pL_world

        # canonical curvature equation
        kappa = 2.0 * yL / (self.L * self.L)
        omega_max = max_yawrate
        v_cmd = min(max_vel, omega_max / (abs(kappa) + 1e-6))
        omega = max(-omega_max, min(omega_max, v_cmd * kappa))
        return VelocityCommand2D(v=v_cmd, omega=float(omega)), pL_world


    #Core Methods

    def _update_progress_index(self, bot_xy: Point2, *, i0: int, i1: int) -> int:
        """Closest segment index to bot_xy over segments [i0..i1] using projection distance."""
        best_i = i0
        best_d = float("inf")
        for i in range(i0, i1 + 1):  
            proj = project_point_to_segment(bot_xy, self.P[i], self.P[i + 1])
            if proj.dist < best_d:
                best_d = proj.dist
                best_i = i
        return best_i

    def _pick_ahead_hit(self, hits: List[Point2], pose: Pose2D) -> Optional[Point2]:
        """Pick an intersection point that is ahead of robot (x_body > 0)."""
        if not hits:
            return None
        
        best_p: Optional[Point2] = None
        best_x = -float("inf")

        for pW in hits:
            xB, _yB = world_to_body(pW, pose)
            if xB > 0.0 and xB > best_x:
                best_x = xB
                best_p = pW
        return best_p

    def _find_lookahead_point(self, pose: Pose2D) -> Point2:
        bot_xy = (pose.x, pose.y)
        N = len(self.P)
        if N < 2:
            return bot_xy
        
        # local progress update window
        i0 = max(0, self.progress_idx - self.eps_back)
        i1 = min(N - 2, self.progress_idx + self.W) #clamp it to N-2th index
        
        candidate_idx = self._update_progress_index(bot_xy, i0=i0, i1=i1)
        self.progress_idx = max(self.progress_idx, candidate_idx)  # monotonic guard

        # recompute window upper bound based on possibly advanced progress
        i1 = min(N - 2, self.progress_idx + self.W)

        # scan forward for first valid circle intersection ahead
        for i in range(self.progress_idx, i1 + 1):
            hits = circle_segment_intersections(bot_xy, self.L, self.P[i], self.P[i + 1])
            p_star = self._pick_ahead_hit(hits, pose)
            if p_star is not None:
                return p_star
            
        #Fallback plan if no points in path intersect with Lookahead Circle (either path too far or L too large)
        proj = project_point_to_segment(bot_xy, self.P[self.progress_idx], self.P[self.progress_idx + 1])
        return proj.p

