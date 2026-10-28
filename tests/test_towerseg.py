import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from towerseg import (  # noqa: E402
    GROUND, POLE, Cuboid, fit_circle, fit_cuboid, iou_3d, make_tower, point_iou, ransac_plane, segment,
)


def test_ransac_finds_plane_among_outliers():
    rng = np.random.default_rng(0)
    plane = np.c_[rng.uniform(-5, 5, (500, 2)), rng.normal(0, 0.01, 500)]
    noise = rng.uniform(-5, 5, (200, 3))
    n, d, inl = ransac_plane(np.vstack([plane, noise]))
    assert abs(abs(n[2]) - 1) < 1e-2
    assert inl[:500].mean() > 0.95


def test_circle_fit_recovers_pole():
    rng = np.random.default_rng(1)
    phi = rng.uniform(0, 2 * np.pi, 300)
    xy = np.c_[2 + 0.4 * np.cos(phi), -1 + 0.4 * np.sin(phi)] + rng.normal(0, 0.005, (300, 2))
    c, r = fit_circle(xy)
    assert np.allclose(c, [2, -1], atol=0.01) and abs(r - 0.4) < 0.01


def test_ground_and_pole_segmentation():
    t = make_tower(3)
    seg = segment(t.points)
    assert point_iou(seg.ground, t.labels == GROUND) > 0.95
    assert point_iou(seg.pole, t.labels == POLE) > 0.95
    assert abs(seg.pole_radius - t.pole_radius) < 0.03


def test_iou_identical_and_disjoint():
    a = Cuboid(np.array([0.0, 0, 5]), np.array([0.4, 0.2, 2.0]), 0.3)
    b = Cuboid(np.array([3.0, 0, 5]), np.array([0.4, 0.2, 2.0]), 0.3)
    assert abs(iou_3d(a, a) - 1) < 1e-3
    assert iou_3d(a, b) == 0.0


def test_cuboid_fit_on_clean_panel():
    t = make_tower(5, density=800, noise=0.0)
    gt = t.antennas[0]
    pts = t.points[(t.instance == np.unique(t.instance[t.labels == 3])[0])]
    box = fit_cuboid(pts, t.pole_xy, trim=0.0)
    assert np.linalg.norm(box.center - gt.center) < 0.08
    assert abs(box.size[2] - gt.size[2]) < 0.05
