"""Two visual roads into Kenyon cells. No random mix, and no action trace.

Medulla features are small and local. They drive KCg-d only.
Lobula features are the whole object, including a YOLO box. They drive
KCab-p only. APL sparsifies each group on its own. Dopamine stores the
value of what is seen on γ-d. Nothing here is a motor command or its result.
"""

from __future__ import annotations

import numpy as np

from .hemifield import overlap_bands
from .mb_runtime import apl_sparsify
from .raw_sense import N_AZ, OFF_BODY, OFF_FLOOR, OFF_MOTION

# The biologist's round counts. The file's own cells are used, not a padded set.
GAMMA_D_ABOUT = 150
ALPHA_BETA_P_ABOUT = 65
GAMMA_D_TYPE = "KCg-d"
ALPHA_BETA_P_TYPE = "KCab-p"

# Medulla, in order: body RGB, floor RGB, body edges, floor edges, motion.
N_MEDULLA = N_AZ * 3 + N_AZ * 3 + N_AZ + N_AZ + N_AZ
# Lobula, in order: occupancy, body-vs-floor, spread, object RGB, YOLO conf/width/height/center.
N_LOBULA = 10
OFF_LOBULA_YOLO = 6


def medulla_channels(feat: np.ndarray) -> np.ndarray:
    """Local color, edges, and motion. Lidar and closing are not read."""
    raw = np.asarray(feat, dtype=np.float32).ravel()
    out = np.zeros(N_MEDULLA, dtype=np.float32)
    out[0 : N_AZ * 3] = raw[OFF_BODY : OFF_BODY + N_AZ * 3]
    out[N_AZ * 3 : N_AZ * 6] = raw[OFF_FLOOR : OFF_FLOOR + N_AZ * 3]
    out[N_AZ * 6 : N_AZ * 7] = _edges(raw, OFF_BODY)
    out[N_AZ * 7 : N_AZ * 8] = _edges(raw, OFF_FLOOR)
    out[N_AZ * 8 : N_AZ * 9] = raw[OFF_MOTION : OFF_MOTION + N_AZ]
    return out


def lobula_channels(feat: np.ndarray, yolo: np.ndarray | None = None) -> np.ndarray:
    """One object for this eye: its shape, its color, and the YOLO box."""
    raw = np.asarray(feat, dtype=np.float32).ravel()
    out = np.zeros(N_LOBULA, dtype=np.float32)
    body = raw[OFF_BODY : OFF_BODY + N_AZ * 3].reshape(N_AZ, 3)
    floor = raw[OFF_FLOOR : OFF_FLOOR + N_AZ * 3].reshape(N_AZ, 3)
    # A hemifield view lives in slots 4–7. Slots 0–3 are the other eye.
    body = body[4:]
    floor = floor[4:]
    body_e = body.sum(axis=1)
    floor_e = floor.sum(axis=1)
    present = (body_e + floor_e) > 1e-6
    n = int(present.sum())
    out[0] = np.float32(n / 4.0)
    bsum = float(body_e.sum())
    fsum = float(floor_e.sum())
    if bsum + fsum > 1e-6:
        out[1] = np.float32(bsum / (bsum + fsum))
    if n >= 2:
        idx = np.flatnonzero(present).astype(np.float32)
        out[2] = np.float32(min(float(idx.std() / 1.5), 1.0))
    if bsum > 1e-6:
        out[3:6] = (body * body_e[:, None]).sum(axis=0) / np.float32(bsum)
    elif fsum > 1e-6:
        out[3:6] = (floor * floor_e[:, None]).sum(axis=0) / np.float32(fsum)
    if yolo is not None:
        vec = np.asarray(yolo, dtype=np.float32).ravel()
        if vec.shape[0] >= 4:
            out[OFF_LOBULA_YOLO : OFF_LOBULA_YOLO + 4] = np.clip(vec[:4], 0.0, 1.0)
    return out


def yolo_vector(boxes, side: str, overlap: float) -> np.ndarray:
    """Strongest box whose centre sits in this eye: conf, width, height, center.

    An empty list is four zeros. Image left is the left eye.
    """
    out = np.zeros(4, dtype=np.float32)
    if not boxes:
        return out
    lo, hi = overlap_bands(overlap)
    if side == "L":
        left, right = 0.0, hi
    elif side == "R":
        left, right = lo, 1.0
    else:
        raise ValueError(side)
    best = None
    best_conf = -1.0
    for box in boxes:
        x0 = float(box.x0)
        x1 = float(box.x1)
        if x1 < x0:
            x0, x1 = x1, x0
        center = 0.5 * (x0 + x1)
        if center < left - 1e-9 or center > right + 1e-9:
            continue
        conf = float(box.conf)
        if conf > best_conf:
            y0 = float(box.y0)
            y1 = float(box.y1)
            if y1 < y0:
                y0, y1 = y1, y0
            best = (conf, x1 - x0, y1 - y0, center)
            best_conf = conf
    if best is None:
        return out
    out[:] = np.clip(np.asarray(best, dtype=np.float32), 0.0, 1.0)
    return out


def _edges(feat: np.ndarray, offset: int) -> np.ndarray:
    """Mean absolute color step to a neighboring sector. A flat field is zero."""
    rgb = feat[offset : offset + N_AZ * 3].reshape(N_AZ, 3)
    edges = np.zeros(N_AZ, dtype=np.float32)
    for i in range(N_AZ):
        diffs = []
        for j in (i - 1, i + 1):
            if j < 0 or j >= N_AZ:
                continue
            if float(rgb[i].sum()) <= 0.0 and float(rgb[j].sum()) <= 0.0:
                continue
            diffs.append(float(np.mean(np.abs(rgb[i] - rgb[j]))))
        if diffs:
            edges[i] = max(diffs)
    return edges


def _group_indices(brain, type_name: str, anatomical_side: str) -> np.ndarray:
    found = []
    for local, global_i in enumerate(brain.kc_idx):
        if str(brain.cell_types[int(global_i)]) != type_name:
            continue
        if str(brain.sides[int(global_i)]) != anatomical_side:
            continue
        found.append(local)
    return np.asarray(found, dtype=np.int32)


def _channels_for(n_cells: int, n_channels: int) -> np.ndarray:
    """Cell i of the group reads channel i mod n. The same cell always reads it."""
    if n_cells <= 0:
        return np.zeros(0, dtype=np.int32)
    return np.arange(n_cells, dtype=np.int32) % int(n_channels)


class VisualPathway:
    """γ-d from the medulla, α/β-p from the lobula, each with its own APL."""

    def __init__(self, brain, anatomical_side: str):
        if anatomical_side not in ("left", "right"):
            raise ValueError(anatomical_side)
        self.side = anatomical_side
        self.n_kc = int(brain.n_kc)
        self.gamma_idx = _group_indices(brain, GAMMA_D_TYPE, anatomical_side)
        self.abp_idx = _group_indices(brain, ALPHA_BETA_P_TYPE, anatomical_side)
        self.gamma_chan = _channels_for(len(self.gamma_idx), N_MEDULLA)
        self.abp_chan = _channels_for(len(self.abp_idx), N_LOBULA)
        self.gamma_mask = np.zeros(self.n_kc, dtype=bool)
        self.abp_mask = np.zeros(self.n_kc, dtype=bool)
        self.gamma_mask[self.gamma_idx] = True
        self.abp_mask[self.abp_idx] = True
        self.n_visual = int(len(self.gamma_idx) + len(self.abp_idx))

    def kenyon(self, feat: np.ndarray, yolo: np.ndarray | None = None) -> np.ndarray:
        """KC rates for one eye. Cells outside the two groups stay silent."""
        kc = np.zeros(self.n_kc, dtype=np.float32)
        medulla = medulla_channels(feat)
        lobula = lobula_channels(feat, yolo)
        if len(self.gamma_idx):
            kc[self.gamma_idx] = medulla[self.gamma_chan]
            kc[self.gamma_idx] = apl_sparsify(kc[self.gamma_idx])
        if len(self.abp_idx):
            kc[self.abp_idx] = lobula[self.abp_chan]
            kc[self.abp_idx] = apl_sparsify(kc[self.abp_idx])
        return kc
