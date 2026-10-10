"""Left and right views for two mushroom bodies on one connectome.

Sector 0 is the right edge of the 110° field (negative yaw) and sector 7 is
the left edge (positive yaw). The camera strip uses that same index, so a
sector stays with its lidar bin. MB_L receives sectors 4–7. MB_R receives
sectors 0–3 mirrored onto those same slots. With ``overlap`` above zero the
two strips share the middle of the field and each is resampled back into
those four slots. The 72-d vector does not grow.

One shared projector then sees the same kind of pattern for a dog on either
side. Two independent random maps would make a copied weight set blind on
the side it was not trained on. The fly's two mushroom bodies are mirrors
of one map, with their own KC→MBON weights.
"""

from __future__ import annotations

import numpy as np

from .raw_sense import N_AZ, N_RAW, OFF_BODY, OFF_CLOSE, OFF_FLOOR, OFF_LIDAR, OFF_MOTION

# Ego-left slots. A right-hemifield sector s is written to slot (N_AZ - 1 - s):
# 3→4, 2→5, 1→6, 0→7, so the midline stays the midline.
_LEFT = (4, 5, 6, 7)
_RIGHT_SRC = (3, 2, 1, 0)
# Default shared fraction of the field. 0 is the hard midline above.
DEFAULT_OVERLAP = 0.40

YAW_K = 1.0
YAW_EPS = 1e-3
YAW_DEAD = 0.08


def _paint(src: np.ndarray, dst: np.ndarray, src_i: int, dst_i: int) -> None:
    dst[OFF_BODY + dst_i * 3 : OFF_BODY + dst_i * 3 + 3] = src[OFF_BODY + src_i * 3 : OFF_BODY + src_i * 3 + 3]
    dst[OFF_FLOOR + dst_i * 3 : OFF_FLOOR + dst_i * 3 + 3] = src[OFF_FLOOR + src_i * 3 : OFF_FLOOR + src_i * 3 + 3]
    dst[OFF_MOTION + dst_i] = src[OFF_MOTION + src_i]
    dst[OFF_LIDAR + dst_i] = src[OFF_LIDAR + src_i]
    dst[OFF_CLOSE + dst_i] = src[OFF_CLOSE + src_i]


def clamp_overlap(overlap: float) -> float:
    """Shared fraction of the field, kept inside [0, 0.5]."""
    value = float(overlap)
    if value < 0.0:
        return 0.0
    if value > 0.5:
        return 0.5
    return value


def overlap_bands(overlap: float) -> tuple[float, float]:
    """Image fractions of the shared zone. Image left is the robot's right eye.

    Returns ``(lo, hi)``. ``[0, lo)`` is П only, ``[lo, hi)`` is Л+П, ``[hi, 1]`` is Л.
    ``overlap`` 0.4 is ``(0.3, 0.7)``. Zero overlap is the midline, ``(0.5, 0.5)``.
    """
    half = 0.5 * clamp_overlap(overlap)
    return 0.5 - half, 0.5 + half


def _gather(src: np.ndarray, index: int) -> np.ndarray:
    i = int(index)
    out = np.empty(9, dtype=np.float64)
    body = OFF_BODY + i * 3
    floor = OFF_FLOOR + i * 3
    out[0:3] = src[body : body + 3]
    out[3:6] = src[floor : floor + 3]
    out[6] = src[OFF_MOTION + i]
    out[7] = src[OFF_LIDAR + i]
    out[8] = src[OFF_CLOSE + i]
    return out


def _scatter(dst: np.ndarray, slot: int, values: np.ndarray) -> None:
    body = OFF_BODY + int(slot) * 3
    floor = OFF_FLOOR + int(slot) * 3
    dst[body : body + 3] = values[0:3]
    dst[floor : floor + 3] = values[3:6]
    dst[OFF_MOTION + slot] = values[6]
    dst[OFF_LIDAR + slot] = values[7]
    dst[OFF_CLOSE + slot] = values[8]


def _coverage(u0: float, u1: float) -> np.ndarray:
    """Area of ``[u0, u1]`` inside each azimuth sector, normalised to sum 1."""
    width = 1.0 / float(N_AZ)
    weights = np.zeros(N_AZ, dtype=np.float64)
    for index in range(N_AZ):
        start = index * width
        stop = start + width
        lo = u0 if u0 > start else start
        hi = u1 if u1 < stop else stop
        if hi > lo:
            weights[index] = hi - lo
    total = float(weights.sum())
    if total > 0.0:
        weights /= total
    return weights


def _resample(src: np.ndarray, dst: np.ndarray, spans) -> None:
    for u0, u1, slot in spans:
        weights = _coverage(float(u0), float(u1))
        acc = np.zeros(9, dtype=np.float64)
        for index in range(N_AZ):
            if weights[index] > 0.0:
                acc += weights[index] * _gather(src, index)
        _scatter(dst, int(slot), acc.astype(np.float32))


def _spans(side: str, overlap: float):
    """Four output bins in slots 4–7. Right view is mirrored so the midline stays slot 4."""
    lo, hi = overlap_bands(overlap)
    if side == "L":
        start = lo
        stop = 1.0
        step = (stop - start) / float(len(_LEFT))
        return [(start + k * step, start + (k + 1) * step, 4 + k) for k in range(len(_LEFT))]
    if side == "R":
        start = 0.0
        stop = hi
        step = (stop - start) / float(len(_LEFT))
        # Slot 4 is the part nearest the midline (high u). Slot 7 is the far right edge.
        return [(stop - (k + 1) * step, stop - k * step, 4 + k) for k in range(len(_LEFT))]
    raise ValueError(side)


def hemifield(feat: np.ndarray, side: str, overlap: float = 0.0) -> np.ndarray:
    """72-d view. One hemifield lands in slots 4–7. Slots 0–3 stay zero.

    ``overlap`` is the shared fraction of the field (0..0.5). Zero copies
    sectors 4–7 to MB_L and mirrors 0–3 onto those same slots for MB_R.
    A wider overlap resamples the strip back into those four slots, so the
    projector and the KC→MBON file keep the same shape.
    """
    src = np.asarray(feat, dtype=np.float32).ravel()
    if src.shape[0] != N_RAW:
        raise ValueError("feature length %s" % src.shape[0])
    if side not in ("L", "R"):
        raise ValueError(side)
    dst = np.zeros(N_RAW, dtype=np.float32)
    shared = clamp_overlap(overlap)
    if shared <= 1e-8:
        pairs = zip(_LEFT, _LEFT) if side == "L" else zip(_RIGHT_SRC, _LEFT)
        for src_i, dst_i in pairs:
            _paint(src, dst, src_i, dst_i)
        return dst
    _resample(src, dst, _spans(side, shared))
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
    """One monitor line: the two readouts, their difference, and the fly's yaw.

    Every number is a fixed-width signed field, so the labels stay put.
    """
    from .tabnum import format_fly_line as _line

    name = "билатерально" if steer == "bilateral" else "секторы"
    return _line(r_l, r_r, diff, z, name)
