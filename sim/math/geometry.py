from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import math
import numpy as np

from sim.types import Point2

@dataclass(frozen=True)
class SegmentProjection:
    t: float      # clamped in [0,1]
    p: Point2     # closest point on segment
    dist: float   # ||x - p||


def project_point_to_segment(x: Point2, a: Point2, b: Point2, eps: float = 1e-12) -> SegmentProjection:
    """
    Project point x onto segment a->b.
    Returns:
      t in [0,1] such that p = a + t*(b-a) is the closest point on the segment.
      p = closest point on segment
      dist = Euclidean distance ||x - p||
    """
    OX = np.asarray(x, dtype=float)
    OA = np.asarray(a, dtype=float)
    OB = np.asarray(b, dtype=float)

    AB = OB - OA
    denom = float(AB @ AB)

    if denom <= eps:  # degenerate segment (a ~= b)
        OP = OA
        dist = float(np.linalg.norm(OX - OP))
        return SegmentProjection(t=0.0, p=(float(OP[0]), float(OP[1])), dist=dist)

    t_raw = float(((OX - OA) @ AB) / denom)
    t = float(np.clip(t_raw, 0.0, 1.0))
    OP = OA + t * AB
    dist = float(np.linalg.norm(OX - OP))
    return SegmentProjection(t=t, p=(float(OP[0]), float(OP[1])), dist=dist)


def circle_segment_intersections(c: Point2, r: float, a: Point2, b: Point2, eps: float = 1e-12) -> List[Point2]:
    """
    Intersections between circle (center c, radius r) and segment a->b.
    Returns 0, 1 (tangent), or 2 points.
    """
    if r <= 0:
        return []

    OC = np.asarray(c, dtype=float)
    OA = np.asarray(a, dtype=float)
    OB = np.asarray(b, dtype=float)

    d = OB - OA          # segment direction
    f = OA - OC          # from center to segment start

    A = float(d @ d)
    if A <= eps:         # degenerate segment
        return []

    B = float(2.0 * (f @ d))
    C = float((f @ f) - r * r)

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return []

    sqrt_disc = math.sqrt(disc)

    u1 = (-B - sqrt_disc) / (2.0 * A)
    u2 = (-B + sqrt_disc) / (2.0 * A)

    hits: List[Point2] = []
    for u in (u1, u2):
        if 0.0 <= u <= 1.0:
            P = OA + u * d
            hits.append((float(P[0]), float(P[1])))

    # Deduplicate nearly identical hits (numerical)
    uniq: List[Point2] = []
    for p in hits:
        if all((p[0]-q[0])**2 + (p[1]-q[1])**2 > 1e-18 for q in uniq):
            uniq.append(p)

    return uniq
