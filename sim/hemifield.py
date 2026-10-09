"""Left and right views for two mushroom bodies on one connectome.

Sector 0 is the right edge of the 110° field (negative yaw) and sector 7 is
the left edge (positive yaw). The camera strip uses that same index, so a
sector stays with its lidar bin. MB_L receives sectors 4–7. MB_R receives
sectors 0–3 mirrored onto those same slots.

One shared projector then sees the same kind of pattern for a dog on either
side. Two independent random maps would make a copied weight set blind on
the side it was not trained on. The fly's two mushroom bodies are mirrors
of one map, with their own KC→MBON weights.
"""

from __future__ import annotations

import numpy as np

from .raw_sense import N_RAW, OFF_BODY, OFF_CLOSE, OFF_FLOOR, OFF_LIDAR, OFF_MOTION

# Ego-left slots. A right-hemifield sector s is written to slot (N_AZ - 1 - s):
# 3→4, 2→5, 1→6, 0→7, so the midline stays the midline.
_LEFT = (4, 5, 6, 7)
_RIGHT_SRC = (3, 2, 1, 0)

YAW_K = 1.0
YAW_EPS = 1e-3
YAW_DEAD = 0.08


def _paint(src: np.ndarray, dst: np.ndarray, src_i: int, dst_i: int) -> None:
    dst[OFF_BODY + dst_i * 3 : OFF_BODY + dst_i * 3 + 3] = src[OFF_BODY + src_i * 3 : OFF_BODY + src_i * 3 + 3]
    dst[OFF_FLOOR + dst_i * 3 : OFF_FLOOR + dst_i * 3 + 3] = src[OFF_FLOOR + src_i * 3 : OFF_FLOOR + src_i * 3 + 3]
    dst[OFF_MOTION + dst_i] = src[OFF_MOTION + src_i]
    dst[OFF_LIDAR + dst_i] = src[OFF_LIDAR + src_i]
    dst[OFF_CLOSE + dst_i] = src[OFF_CLOSE + src_i]


def hemifield(feat: np.ndarray, side: str) -> np.ndarray:
    """72-d view with one hemifield in the left-hand slots and the other half zero."""
    src = np.asarray(feat, dtype=np.float32).ravel()
    if src.shape[0] != N_RAW:
        raise ValueError("feature length %s" % src.shape[0])
    dst = np.zeros(N_RAW, dtype=np.float32)
    if side == "L":
        pairs = zip(_LEFT, _LEFT)
    elif side == "R":
        pairs = zip(_RIGHT_SRC, _LEFT)
    else:
        raise ValueError(side)
    for src_i, dst_i in pairs:
        _paint(src, dst, src_i, dst_i)
    return dst


def bilateral_yaw(r_l: float, r_r: float) -> float:
    """Yaw from the two readouts. Left is positive. A small imbalance is zero.

    ``z = k (R_L - R_R) / (|R_L| + |R_R| + ε)``, then clamped to [-1, 1].
    """
    left = float(r_l)
    right = float(r_r)
    diff = left - right
    denom = abs(left) + abs(right) + YAW_EPS
    z = YAW_K * diff / denom
    if z > 1.0:
        z = 1.0
    elif z < -1.0:
        z = -1.0
    if abs(z) < YAW_DEAD:
        return 0.0
    return float(z)


def format_fly_line(r_l, r_r, diff, z, steer: str) -> str:
    """One monitor line: the two readouts, their difference, and the fly's yaw."""

    def num(value) -> str:
        if value is None:
            return "—"
        return "%+.0f" % float(value)

    def yaw(value) -> str:
        if value is None:
            return "—"
        return "%+.2f" % float(value)

    name = "билатерально" if steer == "bilateral" else "секторы"
    return "муха L %s   R %s   Δ %s   z %s   %s" % (num(r_l), num(r_r), num(diff), yaw(z), name)
