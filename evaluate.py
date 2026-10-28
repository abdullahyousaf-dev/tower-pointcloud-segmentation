"""Train the per-point classifier on synthetic towers, evaluate on held-out towers,
and save a figure of one test tower.

    python evaluate.py                     # 60 train / 30 test towers
    python evaluate.py --train 20 --test 5 # quick run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402

from towerseg import (  # noqa: E402
    ANTENNA, GROUND, LABELS, MOUNT, POLE, RRU, antenna_instances, azimuth_deg, fit_cuboid,
    iou_3d, make_tower, point_features, point_iou, segment,
)

ROOT = Path(__file__).resolve().parent
COLORS = {GROUND: "#b0a080", POLE: "#7a7a7a", MOUNT: "#2a7ab9", ANTENNA: "#e4572e", RRU: "#3fa34d"}


def prepare(seed):
    t = make_tower(seed)
    seg = segment(t.points)
    idx = np.flatnonzero(seg.rest)
    return t, seg, idx, point_features(t.points, seg, idx)


def match_antennas(t, boxes, pole_xy, min_iou=0.25):
    used, rows = set(), []
    for gt in t.antennas:
        scores = [(iou_3d(gt, b), j) for j, b in enumerate(boxes) if j not in used]
        if not scores:
            continue
        best_iou, j = max(scores)
        if best_iou >= min_iou:
            used.add(j)
            b = boxes[j]
            daz = (azimuth_deg(b, pole_xy) - azimuth_deg(gt, t.pole_xy) + 180) % 360 - 180
            dyaw = np.rad2deg(np.angle(np.exp(1j * (b.yaw - gt.yaw))))
            rows.append((best_iou, abs(daz), abs(dyaw), abs(b.center[2] - gt.center[2]),
                         abs(b.size[2] - gt.size[2]), np.linalg.norm(b.center - gt.center)))
    return rows, len(boxes) - len(used)


def plot(t, seg, idx, pred, boxes, out):
    lab = np.full(len(t.points), -1)
    lab[seg.ground], lab[seg.pole] = GROUND, POLE
    lab[idx] = pred
    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sub = np.random.default_rng(0).choice(len(t.points), min(30000, len(t.points)), replace=False)
    for k, col in COLORS.items():
        m = sub[lab[sub] == k]
        ax.scatter(*t.points[m].T, s=0.3, c=col, label=LABELS[k])
    ax.set_box_aspect((1, 1, 2.2))
    ax.set_title("Predicted class per point")
    ax.legend(markerscale=15, loc="upper left", fontsize=8)
    ax.set_axis_off()

    ax2 = fig.add_subplot(1, 2, 2)
    near_top = t.points[:, 2] > seg.top - 4
    for k, col in COLORS.items():
        m = near_top & (lab == k)
        ax2.scatter(t.points[m, 0], t.points[m, 1], s=0.5, c=col)
    for gt in t.antennas:
        ax2.fill(*gt.corners_xy().T, fill=False, ec="black", lw=1.5)
    for b in boxes:
        ax2.fill(*b.corners_xy().T, fill=False, ec="#e4572e", lw=1.2, ls="--")
        c = b.center[:2]
        ax2.annotate(f"{azimuth_deg(b, seg.pole_xy):.0f}°", (c - seg.pole_xy) * 0.3 + c,
                     fontsize=8, ha="center", va="center")
    ax2.plot([], [], "k-", label="ground-truth cuboid")
    ax2.plot([], [], "--", color="#e4572e", label="fitted cuboid (azimuth)")
    reach = max(np.linalg.norm(gt.center[:2] - seg.pole_xy) for gt in t.antennas) + 0.8
    ax2.set_xlim(seg.pole_xy[0] - reach, seg.pole_xy[0] + reach)
    ax2.set_ylim(seg.pole_xy[1] - reach, seg.pole_xy[1] + reach)
    ax2.set_aspect("equal")
    ax2.set_title("Antenna level, top view")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--test", type=int, default=30)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)

    Xtr, ytr = [], []
    for s in range(args.train):
        t, _, idx, X = prepare(s)
        y = t.labels[idx]
        keep = np.flatnonzero(y >= MOUNT)  # outliers and stray ground/pole points are left out
        keep = rng.choice(keep, min(4000, len(keep)), replace=False)
        Xtr.append(X[keep]), ytr.append(y[keep])
    clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0, class_weight="balanced")
    clf.fit(np.vstack(Xtr), np.concatenate(ytr))

    y_true, y_pred, rows, n_gt, false_pos, g_iou, p_iou = [], [], [], 0, 0, [], []
    for i, s in enumerate(range(10_000, 10_000 + args.test)):
        t, seg, idx, X = prepare(s)
        pred = clf.predict(X)
        y = t.labels[idx]
        y_true.append(y[y >= MOUNT]), y_pred.append(pred[y >= MOUNT])
        boxes = [fit_cuboid(t.points[g], seg.pole_xy) for g in antenna_instances(t.points, idx, pred)]
        r, fp = match_antennas(t, boxes, seg.pole_xy)
        rows += r
        n_gt += len(t.antennas)
        false_pos += fp
        g_iou.append(point_iou(seg.ground, t.labels == GROUND))
        p_iou.append(point_iou(seg.pole, t.labels == POLE))
        if i == 0:
            (ROOT / "media").mkdir(exist_ok=True)
            plot(t, seg, idx, pred, boxes, ROOT / "media" / "tower_result.png")

    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
    r = np.array(rows)
    metrics = {
        "train_towers": args.train,
        "test_towers": args.test,
        "ground_point_iou": round(float(np.mean(g_iou)), 3),
        "pole_point_iou": round(float(np.mean(p_iou)), 3),
        "point_macro_f1": round(float(f1_score(yt, yp, average="macro")), 3),
        "antenna_precision": round(len(r) / max(len(r) + false_pos, 1), 3),
        "antenna_recall": round(len(r) / n_gt, 3),
        "antenna_mean_3d_iou": round(float(r[:, 0].mean()), 3),
        "azimuth_mae_deg": round(float(r[:, 1].mean()), 2),
        "facing_yaw_mae_deg": round(float(r[:, 2].mean()), 2),
        "center_error_cm": round(float(r[:, 5].mean() * 100), 1),
        "antenna_height_mae_cm": round(float(r[:, 4].mean() * 100), 1),
    }
    print(classification_report(yt, yp, labels=[MOUNT, ANTENNA, RRU],
                                target_names=["mount", "antenna", "rru"], digits=3))
    for k, v in metrics.items():
        print(f"{k:<24}{v}")
    (ROOT / "media" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDone in {time.time() - t0:.0f} s. Figure: media/tower_result.png")


if __name__ == "__main__":
    main()
