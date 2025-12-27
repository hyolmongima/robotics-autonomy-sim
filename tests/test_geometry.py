import math

from sim.math.geometry import project_point_to_segment, circle_segment_intersections


def test_project_point_to_segment_middle():
    # segment from (0,0) to (10,0), point above the middle
    proj = project_point_to_segment((5.0, 3.0), (0.0, 0.0), (10.0, 0.0))
    assert abs(proj.t - 0.5) < 1e-9
    assert abs(proj.p[0] - 5.0) < 1e-9
    assert abs(proj.p[1] - 0.0) < 1e-9
    assert abs(proj.dist - 3.0) < 1e-9


def test_project_point_to_segment_clamps_to_a():
    # point projects before segment start
    proj = project_point_to_segment((-2.0, 1.0), (0.0, 0.0), (10.0, 0.0))
    assert abs(proj.t - 0.0) < 1e-9
    assert proj.p == (0.0, 0.0)
    assert abs(proj.dist - math.hypot(2.0, 1.0)) < 1e-9


def test_project_point_to_segment_clamps_to_b():
    # point projects after segment end
    proj = project_point_to_segment((12.0, -1.0), (0.0, 0.0), (10.0, 0.0))
    assert abs(proj.t - 1.0) < 1e-9
    assert proj.p == (10.0, 0.0)
    assert abs(proj.dist - math.hypot(2.0, 1.0)) < 1e-9


def test_circle_segment_intersections_two_hits():
    # circle centered at origin r=5, segment crosses x-axis from -10 to 10
    hits = circle_segment_intersections((0.0, 0.0), 5.0, (-10.0, 0.0), (10.0, 0.0))
    # should be (-5,0) and (5,0) (order not guaranteed)
    assert len(hits) == 2
    xs = sorted([round(p[0], 6) for p in hits])
    ys = sorted([round(p[1], 6) for p in hits])
    assert xs == [-5.0, 5.0]
    assert ys == [0.0, 0.0]


def test_circle_segment_intersections_tangent_one_hit():
    # tangent at (0,5): segment is horizontal y=5
    hits = circle_segment_intersections((0.0, 0.0), 5.0, (-10.0, 5.0), (10.0, 5.0))
    assert len(hits) == 1
    assert abs(hits[0][0] - 0.0) < 1e-6
    assert abs(hits[0][1] - 5.0) < 1e-6


def test_circle_segment_intersections_no_hit():
    hits = circle_segment_intersections((0.0, 0.0), 5.0, (-10.0, 6.0), (10.0, 6.0))
    assert hits == []
