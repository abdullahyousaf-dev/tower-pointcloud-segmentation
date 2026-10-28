"""Tower point cloud -> ground and pole removal -> per-point classification from local
geometry -> antenna instances -> fitted cuboids (position, size, azimuth)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .synth import ANTENNA, Cuboid


# ----------------------------------------------------------------- geometry fits
def ransac_plane(pts: np.ndarray, iters: int = 200, thresh: float = 0.06, seed: int = 0):
    """Best plane n.x + d = 0 by inlier count. Returns (normal, d, inlier_mask)."""
    rng = np.random.default_rng(seed)
    best = (None, None, np.zeros(len(pts), bool))
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -n @ a
        inl = np.abs(pts @ n + d) < thresh
        if inl.sum() > best[2].sum():
            best = (n, d, inl)
    return best


def fit_circle(xy: np.ndarray):
    """Algebraic (Kasa) least-squares circle fit. Returns (centre, radius)."""
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy**2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([cx, cy]), float(np.sqrt(c + cx**2 + cy**2))


def point_spacing(pts: np.ndarray, sample: int = 3000, seed: int = 0) -> float:
    """Median nearest-neighbour distance, used to scale neighbourhood sizes."""
    rng = np.random.default_rng(seed)
    sub = pts[rng.choice(len(pts), min(sample, len(pts)), replace=False)]
    d, _ = NearestNeighbors(n_neighbors=2).fit(pts).kneighbors(sub)
    return float(np.median(d[:, 1]))


# ----------------------------------------------------------- ground and pole
@dataclass
class Segmentation:
    ground: np.ndarray  # bool mask
    pole: np.ndarray  # bool mask
    pole_xy: np.ndarray
    pole_radius: float
    top: float

    @property
    def rest(self) -> np.ndarray:
        return ~self.ground & ~self.pole


def segment(points: np.ndarray) -> Segmentation:
    # Ground: RANSAC plane on the lowest part of the cloud, then everything near it.
    low = points[points[:, 2] < np.percentile(points[:, 2], 30)]
    n, d, _ = ransac_plane(low)
    ground = np.abs(points @ n + d) < 0.08

    # Pole: circle fits on slices from the middle of the tower, where only the pole is.
    rest = points[~ground]
    top = np.percentile(rest[:, 2], 99.5)
    fits = []
    for z in np.linspace(0.25, 0.6, 6) * top:
        sl = rest[np.abs(rest[:, 2] - z) < 0.15, :2]
        if len(sl) > 30:
            centre, rad = fit_circle(sl)
            res = np.abs(np.linalg.norm(sl - centre, axis=1) - rad)
            centre, rad = fit_circle(sl[res < 0.05])
            fits.append((*centre, rad))
    cx, cy, rad = np.median(fits, axis=0)
    radial = np.linalg.norm(points[:, :2] - [cx, cy], axis=1)
    pole = ~ground & (radial < rad + 0.06) & (points[:, 2] < top + 0.3)
    return Segmentation(ground, pole, np.array([cx, cy]), float(rad), float(top))


# ------------------------------------------------------- per-point features
FEATURE_NAMES = [
    "linearity", "planarity", "scattering", "verticality", "normal_z",
    "local_z_extent", "local_xy_extent", "radial_dist", "below_top", "density",
]


def point_features(points: np.ndarray, seg: Segmentation, idx: np.ndarray, k: int = 24) -> np.ndarray:
    """Eigenvalue features of each point's k-neighbourhood (Weinmann et al. style),
    plus where the point sits relative to the pole. Pipes come out linear,
    panels planar, RRUs boxy."""
    pts = points[idx]
    nn = NearestNeighbors(n_neighbors=min(k, len(pts))).fit(pts)
    dist, nbr = nn.kneighbors(pts)
    P = pts[nbr]  # (N, k, 3)
    C = P - P.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", C, C) / P.shape[1]
    evals, evecs = np.linalg.eigh(cov)  # ascending
    l3, l2, l1 = (np.maximum(evals[:, i], 1e-12) for i in range(3))
    normal = evecs[:, :, 0]
    principal = evecs[:, :, 2]
    radial = np.linalg.norm(pts[:, :2] - seg.pole_xy, axis=1) - seg.pole_radius
    return np.c_[
        (l1 - l2) / l1, (l2 - l3) / l1, l3 / l1,
        np.abs(principal[:, 2]), np.abs(normal[:, 2]),
        np.ptp(P[:, :, 2], axis=1), np.linalg.norm(np.ptp(P[:, :, :2], axis=1), axis=1),
        radial, seg.top - pts[:, 2], dist[:, -1],
    ]


# --------------------------------------------------------------- instances
def antenna_instances(points: np.ndarray, idx: np.ndarray, pred: np.ndarray,
                      min_points: int = 60, min_height: float = 0.6):
    """Group points predicted as antenna into panels with DBSCAN; radius scales with
    point spacing. Returns a list of index arrays into `points`."""
    a = idx[pred == ANTENNA]
    if len(a) < min_points:
        return []
    eps = float(np.clip(3.0 * point_spacing(points[a]), 0.06, 0.15))
    lab = DBSCAN(eps=eps, min_samples=8).fit_predict(points[a])
    groups = [a[lab == c] for c in np.unique(lab[lab >= 0]) if (lab == c).sum() >= min_points]
    # Panel antennas are at least ~1.2 m tall; short blobs are misclassified mount parts.
    return [g for g in groups if np.ptp(points[g, 2]) > min_height]


def fit_cuboid(pts: np.ndarray, pole_xy: np.ndarray, trim: float = 2.0) -> Cuboid:
    """Upright cuboid. Yaw from PCA of the points seen from above (the panel's thin
    axis has the least spread); extents from percentiles in that rotated frame, so
    scan noise and stray points don't inflate the box the way min/max would."""
    xy = pts[:, :2]
    cx, cy = xy.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((xy - [cx, cy]).T))
    yaw = float(np.arctan2(evecs[1, 0], evecs[0, 0]))  # smallest-variance direction = depth
    if np.cos(yaw) * (cx - pole_xy[0]) + np.sin(yaw) * (cy - pole_xy[1]) < 0:
        yaw += np.pi  # face away from the pole
    dep = np.array([np.cos(yaw), np.sin(yaw)])
    wid = np.array([-np.sin(yaw), np.cos(yaw)])
    lo, hi = np.percentile(np.c_[pts[:, :2] @ dep, pts[:, :2] @ wid, pts[:, 2]], [trim, 100 - trim], axis=0)
    size = hi - lo
    mid = (hi + lo) / 2
    centre_xy = mid[0] * dep + mid[1] * wid
    return Cuboid(np.array([*centre_xy, mid[2]]), np.array([size[1], size[0], size[2]]), float(yaw))


# ----------------------------------------------------------------- metrics
def iou_3d(a: Cuboid, b: Cuboid) -> float:
    ra = ((a.center[0], a.center[1]), (a.size[1], a.size[0]), np.rad2deg(a.yaw))
    rb = ((b.center[0], b.center[1]), (b.size[1], b.size[0]), np.rad2deg(b.yaw))
    kind, poly = cv2.rotatedRectangleIntersection(ra, rb)
    if kind == cv2.INTERSECT_NONE or poly is None:
        return 0.0
    inter_xy = cv2.contourArea(cv2.convexHull(poly))
    za = (a.center[2] - a.size[2] / 2, a.center[2] + a.size[2] / 2)
    zb = (b.center[2] - b.size[2] / 2, b.center[2] + b.size[2] / 2)
    inter = inter_xy * max(0.0, min(za[1], zb[1]) - max(za[0], zb[0]))
    return float(inter / (np.prod(a.size) + np.prod(b.size) - inter))


def azimuth_deg(c: Cuboid, pole_xy) -> float:
    return float(np.rad2deg(np.arctan2(c.center[1] - pole_xy[1], c.center[0] - pole_xy[0])) % 360)


def point_iou(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred & true).sum() / max((pred | true).sum(), 1))
