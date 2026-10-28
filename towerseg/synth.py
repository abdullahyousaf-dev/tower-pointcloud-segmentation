"""Synthetic telecom-tower point clouds with per-point labels and ground-truth antenna cuboids.

Real drone/photogrammetry scans of towers are client data, so this generator builds
a monopole tower from simple surfaces: ground, pole, sector mounts (arms, face pipe,
vertical pipes), panel antennas and RRUs. Points are sampled on surfaces, then noise
and outliers are added.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LABELS = ["ground", "pole", "mount", "antenna", "rru"]
GROUND, POLE, MOUNT, ANTENNA, RRU = range(5)


@dataclass
class Cuboid:
    center: np.ndarray  # (3,)
    size: np.ndarray  # (width, depth, height): width is tangential, depth radial
    yaw: float  # rotation about z of the depth axis (radians); antennas face outward

    def corners_xy(self) -> np.ndarray:
        w, d, _ = self.size
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        depth_ax, width_ax = np.array([c, s]), np.array([-s, c])
        pts = [self.center[:2] + sd * d / 2 * depth_ax + sw * w / 2 * width_ax for sd in (-1, 1) for sw in (-1, 1)]
        return np.array([pts[0], pts[1], pts[3], pts[2]])


@dataclass
class Tower:
    points: np.ndarray
    labels: np.ndarray
    instance: np.ndarray
    antennas: list = field(default_factory=list)
    pole_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pole_radius: float = 0.0


def _sample_box(c: Cuboid, density: float, rng) -> np.ndarray:
    w, d, h = c.size
    faces = [  # (area, fixed axis, sign)
        (w * h, 0, 1), (w * h, 0, -1),  # front/back (depth axis fixed)
        (d * h, 1, 1), (d * h, 1, -1),  # sides
        (w * d, 2, 1), (w * d, 2, -1),  # top/bottom
    ]
    out = []
    for area, axis, sign in faces:
        n = rng.poisson(area * density)
        local = rng.uniform(-0.5, 0.5, (n, 3)) * np.array([d, w, h])
        local[:, axis] = sign * np.array([d, w, h])[axis] / 2
        out.append(local)
    local = np.vstack(out)
    cy, sy = np.cos(c.yaw), np.sin(c.yaw)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return local @ R.T + c.center


def _sample_cylinder(p0, p1, radius, density, rng) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    n = rng.poisson(2 * np.pi * radius * length * density)
    a = axis / length
    tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    t = rng.uniform(0, length, n)[:, None]
    phi = rng.uniform(0, 2 * np.pi, n)[:, None]
    return p0 + t * a + radius * (np.cos(phi) * u + np.sin(phi) * v)


def _occlude_back(pts: np.ndarray, c: Cuboid, keep: float, rng) -> np.ndarray:
    """Drones see the front of an antenna far better than the side facing the pole:
    drop most points whose local depth coordinate is on the back face."""
    rel = pts - c.center
    depth = rel[:, 0] * np.cos(c.yaw) + rel[:, 1] * np.sin(c.yaw)
    back = depth < -c.size[1] / 2 + 0.02
    return pts[~back | (rng.random(len(pts)) < keep)]


def make_tower(seed: int = 0, density: float | None = None, noise: float | None = None) -> Tower:
    rng = np.random.default_rng(seed)
    density = rng.uniform(200, 450) if density is None else density
    noise = rng.uniform(0.008, 0.025) if noise is None else noise
    parts, labels, inst = [], [], []
    k = 0

    def add(pts, label):
        nonlocal k
        parts.append(pts)
        labels.append(np.full(len(pts), label))
        inst.append(np.full(len(pts), k))
        k += 1

    H = rng.uniform(24, 36)
    r = rng.uniform(0.3, 0.5)
    pole_xy = rng.uniform(-1, 1, 2)
    base = np.array([*pole_xy, 0.0])

    # Ground: 12 x 12 m around the pole, sparser than the structure (seen from far away).
    g = rng.uniform(-6, 6, (int(144 * density * 0.15), 2)) + pole_xy
    g = g[np.linalg.norm(g - pole_xy, axis=1) > r]
    add(np.c_[g, rng.normal(0, 0.02, len(g))], GROUND)

    add(_sample_cylinder(base, base + [0, 0, H], r, density, rng), POLE)

    antennas = []
    h_a = H - rng.uniform(1.5, 3.0)
    rot = rng.uniform(0, 2 * np.pi)
    for s in range(3):
        az = rot + s * 2 * np.pi / 3 + np.deg2rad(rng.uniform(-15, 15))
        out, tan = np.array([np.cos(az), np.sin(az), 0]), np.array([-np.sin(az), np.cos(az), 0])
        R_m = r + rng.uniform(0.9, 1.4)
        centre = base + [0, 0, h_a]
        mount = [
            _sample_cylinder(centre + r * out, centre + R_m * out, 0.04, density, rng),  # arm
            _sample_cylinder(centre + R_m * out - 1.3 * tan, centre + R_m * out + 1.3 * tan, 0.04, density, rng),
        ]
        n_ant = rng.integers(1, 4)
        offsets = np.linspace(-0.8, 0.8, n_ant) if n_ant > 1 else [0.0]
        for off in offsets:
            pipe_c = centre + R_m * out + off * tan
            mount.append(_sample_cylinder(pipe_c - [0, 0, 1.3], pipe_c + [0, 0, 1.3], 0.03, density, rng))
            size = np.array([rng.uniform(0.25, 0.45), rng.uniform(0.1, 0.2), rng.uniform(1.2, 2.4)])
            yaw = az + np.deg2rad(rng.uniform(-6, 6))
            gap = rng.uniform(0.06, 0.22)  # bracket clearance; small gaps can merge with the pipe
            c = Cuboid(pipe_c + (0.03 + gap + size[1] / 2) * out + [0, 0, rng.uniform(-0.2, 0.2)], size, yaw)
            antennas.append(c)
            add(_occlude_back(_sample_box(c, density, rng), c, rng.uniform(0.2, 0.6), rng), ANTENNA)
        add(np.vstack(mount), MOUNT)

    # RRUs bolted to the pole below the antennas.
    for _ in range(rng.integers(1, 4)):
        az = rng.uniform(0, 2 * np.pi)
        size = np.array([0.3, 0.18, 0.45])
        c = Cuboid(base + (r + 0.12 + size[1] / 2) * np.array([np.cos(az), np.sin(az), 0]) + [0, 0, h_a - rng.uniform(2.5, 4)], size, az)
        add(_sample_box(c, density, rng), RRU)

    pts = np.vstack(parts)
    pts += rng.normal(0, noise, pts.shape)
    lab, ins = np.concatenate(labels), np.concatenate(inst)

    # Floating outliers (photogrammetry noise), labelled -1.
    n_out = int(0.003 * len(pts))
    lo, hi = pts.min(0), pts.max(0)
    pts = np.vstack([pts, rng.uniform(lo, hi, (n_out, 3))])
    lab = np.concatenate([lab, np.full(n_out, -1)])
    ins = np.concatenate([ins, np.full(n_out, -1)])
    return Tower(pts, lab, ins, antennas, pole_xy, r)
