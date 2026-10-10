"""Separate 3D window for the two mushroom bodies.

``connectome_mb_v1.npz`` has FlyWire root ids, types, and synapses, not soma
or skeleton coordinates. Soma positions come from
``artifacts/connectome_mb_v1_neurons.csv.gz``. The edge table has neuropil
names (MB_CA, MB_PED, MB_VL, MB_ML) and counts, not synapse xyz.

Stored soma spans on one hemisphere are about 29 × 24 × 1.9 µm, so z is
~15× flatter than x. FAFB sections are 40 nm against 4 nm in x/y, and there
is no skeleton to recover the neuropil. Display scales z by 10, centers each
hemisphere, and lays calyx, peduncle, and the α/β, α′/β′, γ lobes out from
the cell type plus those neuropil counts. The trainer window stays light.

The trainer only sends a UDP datagram to 127.0.0.1:5473. B stays the beep;
this window is J. The view does not spin on its own.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .npz_compat import open_npz

HOST = "127.0.0.1"
VIEW_PORT = 5473
RATE_HZ = 12.0
KC_CAP = 400
EDGE_CAP = 80

_GAMMA = ("KCg-d", "KCg-s1", "KCg-s2", "KCg-s3", "KCg-m")
_AB = ("KCab-p", "KCab")

# Dark window only. The trainer keeps its light theme.
_BG_EDGE = (14, 15, 18)  # #0E0F12
_BG_CORE = (26, 28, 34)  # #1A1C22
_INK = (196, 198, 204)
_KC_IDLE = (148, 150, 156)
_GLOW_L = (255, 186, 140)
_GLOW_R = (150, 220, 170)
_AMBER = (232, 176, 104)
_PAM = (128, 206, 146)
_PPL = (214, 112, 108)
_WIRE = ((168, 148, 118), (142, 168, 186), (176, 156, 196))

# FAFB voxels are 4×4×40 nm. Stored soma z is ~15× shorter than x.
Z_ANISO = 10.0
# Display units per micrometre. Lobes of ~50 µm stay inside the camera.
_UM = 1.0 / 42.0
# µm in the local frame: medial (toward the midline), dorsal, anterior.
_ANCHOR = {
    # Connected mushroom, µm: calyx behind the peduncle, then the lobes.
    "ca": np.array([2.0, 9.0, 12.0], dtype=np.float64),
    "ped": np.array([5.0, 4.0, 20.0], dtype=np.float64),
    "a": np.array([3.0, 46.0, 26.0], dtype=np.float64),
    "ap": np.array([8.0, 32.0, 32.0], dtype=np.float64),
    "b": np.array([22.0, 6.0, 24.0], dtype=np.float64),
    "bp": np.array([16.0, 14.0, 30.0], dtype=np.float64),
    "g": np.array([18.0, -4.0, 16.0], dtype=np.float64),
}
# 3/4 from above and the side: calyx back, peduncle forward, vertical lobe up.
YAW0 = 0.62
PITCH0 = 0.48
DIST0 = 3.05
# Nearly straight up or down, short of the flip at ±90°.
PITCH_LIM = 89.0 * math.pi / 180.0
PAN_KEY = 0.07


def artifact_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "artifacts"
    return Path(__file__).resolve().parents[1] / "artifacts"


def connectome_npz() -> Path:
    folder = artifact_dir()
    plain = folder / "connectome_mb_v1_np1.npz"
    if plain.is_file():
        return plain
    return folder / "connectome_mb_v1.npz"


def neurons_csv() -> Path:
    return artifact_dir() / "connectome_mb_v1_neurons.csv.gz"


def edges_csv() -> Path:
    return artifact_dir() / "connectome_mb_v1_edges.csv.gz"


def compact_ids(vec, cap: int = KC_CAP) -> list:
    """Local KC indices that fired. Capped so one datagram stays small."""
    if vec is None:
        return []
    # Copy first. A view into a buffer the next step reuses is how the exe died in here.
    arr = np.ravel(np.array(vec, copy=True))
    if arr.size == 0:
        return []
    idx = np.flatnonzero(arr > 0)
    if idx.size > cap:
        step = int(np.ceil(idx.size / float(cap)))
        idx = idx[::step][:cap]
    return [int(i) for i in idx]


def _lobe(name: str) -> int:
    if name in _GAMMA:
        return 0
    if name in _AB:
        return 1
    return 2


def _load_somas(path: Path, root_ids: np.ndarray) -> np.ndarray:
    """Soma xyz in nm, aligned to ``root_ids``. NaN where the table has no row."""
    import csv
    import gzip

    out = np.full((len(root_ids), 3), np.nan, dtype=np.float64)
    if not path.is_file():
        return out
    wanted = {int(r): i for i, r in enumerate(root_ids)}
    with gzip.open(str(path), "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                root = int(row.get("root_id") or 0)
            except ValueError:
                continue
            index = wanted.get(root)
            if index is None:
                continue
            raw = []
            for a, b in (("soma_x", "pos_x"), ("soma_y", "pos_y"), ("soma_z", "pos_z")):
                text = row.get(a) or row.get(b) or ""
                try:
                    raw.append(float(text))
                except ValueError:
                    raw.append(float("nan"))
            out[index] = raw
    return out


def _compartment_weights(path: Path, root_ids: np.ndarray) -> np.ndarray:
    """Synapse counts in MB_CA, MB_PED, MB_VL, MB_ML. No xyz in this file."""
    import csv
    import gzip

    acc = np.zeros((len(root_ids), 4), dtype=np.float64)
    if not path.is_file():
        return acc
    wanted = {int(r): i for i, r in enumerate(root_ids)}
    slots = {"CA": 0, "PED": 1, "VL": 2, "ML": 3}
    with gzip.open(str(path), "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("neuropil") or ""
            if not name.startswith("MB_"):
                continue
            parts = name.split("_")
            slot = slots.get(parts[1] if len(parts) > 1 else "")
            if slot is None:
                continue
            try:
                weight = float(row.get("syn_count") or 0.0)
            except ValueError:
                continue
            for col in ("pre_pt_root_id", "post_pt_root_id"):
                try:
                    index = wanted.get(int(row.get(col) or 0))
                except ValueError:
                    index = None
                if index is not None:
                    acc[index, slot] += weight
    return acc


def _type_anchor(lobe: int, index: int) -> np.ndarray:
    if lobe == 0:
        return _ANCHOR["g"]
    if lobe == 1:
        return _ANCHOR["a"] if (index % 2 == 0) else _ANCHOR["b"]
    return _ANCHOR["ap"] if (index % 2 == 0) else _ANCHOR["bp"]


def _neuropil_anchor(weights, lobe: int, index: int) -> np.ndarray:
    ca, ped, vl, ml = (float(v) for v in weights)
    total = ca + ped + vl + ml
    if total < 1.0:
        return _type_anchor(lobe, index)
    vertical = _ANCHOR["a"] if lobe != 2 else _ANCHOR["ap"]
    medial = _ANCHOR["g"] if lobe == 0 else (_ANCHOR["b"] if lobe == 1 else _ANCHOR["bp"])
    acc = _ANCHOR["ca"] * ca + _ANCHOR["ped"] * ped + vertical * vl + medial * ml
    return acc / total


def _world(medial: float, dorsal: float, anterior: float, left: bool) -> tuple:
    sign = 1.0 if left else -1.0
    x = sign * (-0.64 + medial * _UM)
    y = dorsal * _UM
    z = anterior * _UM
    return (x, y, z)


class Orbit:
    """Left drag orbits; right, middle, or Shift drags pan. The view does not spin."""

    def __init__(self):
        self.yaw = YAW0
        self.pitch = PITCH0
        self.dist = DIST0
        self.home_dist = DIST0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.auto = False
        self.idle = 0.0

    def drag(self, dx: float, dy: float) -> None:
        self.idle = 0.0
        self.yaw += float(dx) * 0.008
        pitched = self.pitch + float(dy) * 0.008
        self.pitch = max(-PITCH_LIM, min(PITCH_LIM, pitched))

    def pan_pixels(self, dx: float, dy: float) -> None:
        """Grab the scene: it follows the mouse."""
        self.idle = 0.0
        sens = 0.0024 * float(self.dist)
        self.pan_x -= float(dx) * sens
        self.pan_y += float(dy) * sens

    def nudge(self, right: float, up: float) -> None:
        """Move the camera in the view plane. Positive up shifts the scene down."""
        self.idle = 0.0
        self.pan_x += float(right)
        self.pan_y += float(up)

    def zoom(self, steps: float) -> None:
        self.dist = max(1.6, min(12.0, self.dist - float(steps) * 0.22))

    def reset(self) -> None:
        self.yaw = YAW0
        self.pitch = PITCH0
        self.dist = self.home_dist
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.idle = 1.0

    def tick(self, dt: float, dragging: bool) -> None:
        """Camera stays until the mouse or a camera key moves it."""
        return


class BrainCloud:
    """Display points. Anatomical left is FlyWire ``side=left`` (Л)."""

    def __init__(self, npz_path: Path | None = None, csv_path: Path | None = None, edge_path: Path | None = None):
        z = open_npz(npz_path or connectome_npz())
        try:
            root_ids = z["root_ids"].astype(np.int64)
            types = np.array([str(t) for t in z["cell_types"]])
            if "sides" in z.files:
                sides = np.array([str(s) for s in z["sides"]])
            else:
                sides = np.array(["right"] * len(root_ids))
            kc_idx = z["kc_idx"].astype(np.int32)
            mbon_idx = z["mbon_idx"].astype(np.int32)
            app = z["dan_appetitive_idx"].astype(np.int32) if "dan_appetitive_idx" in z.files else np.zeros(0, np.int32)
            av = z["dan_aversive_idx"].astype(np.int32) if "dan_aversive_idx" in z.files else np.zeros(0, np.int32)
            pre = z["kc_mbon_pre"].astype(np.int32)
            post = z["kc_mbon_post"].astype(np.int32)
        finally:
            z.close()
        raw = _load_somas(csv_path or neurons_csv(), root_ids)
        weights = _compartment_weights(edge_path or edges_csv(), root_ids)
        finite = np.isfinite(raw).all(axis=1)
        self.n_soma = int(finite.sum())
        self.n_schematic = int((~finite).sum())
        self.z_aniso = float(Z_ANISO)
        self.used_neuropil = bool(weights.sum() > 0)
        left = np.array([str(s) == "left" for s in sides])
        spans = []
        medians = {}
        kc_set = set(int(i) for i in kc_idx)
        for flag in (True, False):
            mask = left & finite if flag else (~left) & finite
            if not np.any(mask):
                medians[flag] = np.zeros(3)
                continue
            block = raw[mask]
            medians[flag] = np.median(block, axis=0)
            kc_rows = [i for i in np.flatnonzero(mask) if int(i) in kc_set]
            if kc_rows:
                spans.append(raw[np.asarray(kc_rows)].max(0) - raw[np.asarray(kc_rows)].min(0))
        if spans:
            span = np.max(np.vstack(spans), axis=0)
        else:
            span = np.zeros(3)
        self.span_nm = (float(span[0]), float(span[1]), float(span[2]))
        if self.n_schematic == 0 and self.n_soma and self.used_neuropil:
            self.source = "flywire-soma+neuropil"
        elif self.n_soma and self.n_schematic == 0:
            self.source = "flywire-soma"
        elif self.n_soma == 0:
            self.source = "schematic"
        else:
            self.source = "flywire-soma+schematic"

        def place(global_i: int, lobe: int) -> tuple:
            is_left = bool(left[global_i])
            if not finite[global_i]:
                ang = (global_i * 0.61803398875) % 1.0
                anchor = _type_anchor(lobe, global_i)
                medial = float(anchor[0]) + (ang - 0.5) * 6.0
                dorsal = float(anchor[1]) + ((global_i % 11) - 5) * 0.8
                anterior = float(anchor[2]) + ((global_i % 7) - 3) * 0.8
                return _world(medial, dorsal, anterior, is_left)
            delta = (raw[global_i] - medians[is_left]) / 1000.0
            delta = delta * np.array([1.0, 1.0, Z_ANISO])
            sign = 1.0 if is_left else -1.0
            jitter = np.array([
                delta[0] * sign * 1.4,
                delta[2] * 2.2,
                delta[1] * 1.4,
            ])
            # A cube clip projects as a square. Keep the residual inside a ball.
            norm = float(np.linalg.norm(jitter))
            if norm > 8.0:
                jitter *= 8.0 / norm
            typed = _type_anchor(lobe, global_i)
            neuropil = _neuropil_anchor(weights[global_i], lobe, global_i)
            # Type keeps the cell in its lobe. Neuropil only tugs it.
            # A full neuropil blend sits between compartments and draws a trail.
            if float(np.sum(weights[global_i])) >= 1.0:
                mixed = 0.72 * typed + 0.28 * neuropil
            else:
                mixed = typed
            local = mixed + jitter
            return _world(float(local[0]), float(local[1]), float(local[2]), is_left)

        def calyx_of(global_i: int) -> tuple:
            is_left = bool(left[global_i])
            anchor = _ANCHOR["ca"]
            if not finite[global_i]:
                return _world(float(anchor[0]), float(anchor[1]), float(anchor[2]), is_left)
            delta = (raw[global_i] - medians[is_left]) / 1000.0
            delta = delta * np.array([1.0, 1.0, Z_ANISO])
            sign = 1.0 if is_left else -1.0
            jitter = np.array([delta[0] * sign * 0.35, delta[2] * 0.45, delta[1] * 0.35])
            norm = float(np.linalg.norm(jitter))
            if norm > 5.0:
                jitter *= 5.0 / norm
            return _world(
                float(anchor[0] + jitter[0]),
                float(anchor[1] + jitter[1]),
                float(anchor[2] + jitter[2]),
                is_left,
            )

        kc_lobe = np.array([_lobe(str(types[int(g)])) for g in kc_idx], dtype=np.uint8)
        kc_pos = np.zeros((len(kc_idx), 3), dtype=np.float32)
        calyx = np.zeros((len(kc_idx), 3), dtype=np.float32)
        for i, g in enumerate(kc_idx):
            kc_pos[i] = place(int(g), int(kc_lobe[i]))
            calyx[i] = calyx_of(int(g))
        self.kc_pos = kc_pos
        self.calyx_pos = calyx
        self.kc_side = left[kc_idx].astype(np.uint8)
        self.kc_lobe = kc_lobe

        def group(idx, lobe_name: str):
            if len(idx) == 0:
                return np.zeros((0, 3), np.float32), np.zeros(0, np.uint8)
            lobe = {"g": 0, "a": 1, "p": 2}[lobe_name]
            pos = np.zeros((len(idx), 3), dtype=np.float32)
            for i, g in enumerate(idx):
                pos[i] = place(int(g), lobe)
            return pos, left[np.asarray(idx, dtype=np.int32)].astype(np.uint8)

        self.mbon_pos, self.mbon_side = group(mbon_idx, "a")
        self.pam_pos, self.pam_side = group(app, "g")
        self.ppl_pos, self.ppl_side = group(av, "a")
        g2kc = {int(g): i for i, g in enumerate(kc_idx)}
        g2mbon = {int(g): i for i, g in enumerate(mbon_idx)}
        lp, lq = [], []
        for i in range(len(pre)):
            a = g2kc.get(int(pre[i]))
            b = g2mbon.get(int(post[i]))
            if a is None or b is None:
                continue
            lp.append(a)
            lq.append(b)
        self.edge_pre = np.asarray(lp, dtype=np.int32)
        self.edge_post = np.asarray(lq, dtype=np.int32)
        self.hulls = _hulls(self)

    @property
    def n_kc(self) -> int:
        return int(len(self.kc_pos))


def _wire_box(pts: np.ndarray):
    if len(pts) < 12:
        return []
    center = np.median(pts, axis=0)
    shifted = pts - center
    cov = np.cov(shifted.T)
    _vals, vecs = np.linalg.eigh(cov)
    local = shifted @ vecs
    lo = np.percentile(local, 10, axis=0)
    hi = np.percentile(local, 90, axis=0)
    corners = []
    for bits in range(8):
        coord = np.array([lo[k] if ((bits >> k) & 1) == 0 else hi[k] for k in range(3)])
        corners.append(center + vecs @ coord)
    segments = []
    for i in range(8):
        for bit in range(3):
            j = i | (1 << bit)
            if j != i and i < j:
                segments.append((corners[i], corners[j]))
    return segments


def _hulls(cloud: "BrainCloud"):
    out = []
    for side in (1, 0):
        mask = cloud.kc_side == side
        calyx = cloud.calyx_pos[mask]
        out.append((side, -1, _wire_box(calyx)))
        for lobe in (0, 1, 2):
            pts = cloud.kc_pos[mask & (cloud.kc_lobe == lobe)]
            out.append((side, lobe, _wire_box(pts)))
    return out


def _gain(value: float) -> float:
    mag = abs(float(value))
    return 0.28 + 0.72 * (mag / (mag + 40.0))


def _mix(a, b, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _project(pos, yaw: float, pitch: float, dist: float, cx: float, cy: float, fov: float, pan_x: float = 0.0, pan_y: float = 0.0):
    """Perspective. Smaller returned depth is closer to the camera."""
    if len(pos) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    x = pos[:, 0].astype(np.float64)
    y = pos[:, 1].astype(np.float64)
    z = pos[:, 2].astype(np.float64)
    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x1 = x * cyaw - z * syaw - float(pan_x)
    z1 = x * syaw + z * cyaw
    y2 = y * cp - z1 * sp - float(pan_y)
    z2 = y * sp + z1 * cp + float(dist)
    scale = float(fov) / np.maximum(z2, 0.25)
    return cx + x1 * scale, cy - y2 * scale, z2


def _depth_t(z2: np.ndarray) -> np.ndarray:
    if len(z2) == 0:
        return z2
    lo = float(np.percentile(z2, 6))
    hi = float(np.percentile(z2, 94))
    return np.clip((z2 - lo) / max(hi - lo, 1e-3), 0.0, 1.0)


def _flags(n: int, ids) -> np.ndarray:
    on = np.zeros(n, dtype=bool)
    if not ids:
        return on
    arr = np.asarray(list(ids), dtype=np.int64)
    arr = arr[(arr >= 0) & (arr < n)]
    on[arr] = True
    return on


def _background(w: int, h: int):
    import pygame

    from .sdl_thread import require_main_thread

    require_main_thread()
    cache = getattr(_background, "_cache", None)
    if cache is not None and cache.get_size() == (w, h):
        return cache
    xs = np.linspace(-1.0, 1.0, max(w, 1), dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, max(h, 1), dtype=np.float32)
    radius = np.sqrt(xs[:, None] ** 2 + (ys[None, :] * 0.92) ** 2)
    tone = np.clip(radius / 0.92, 0.0, 1.0) ** 1.35
    edge = np.array(_BG_EDGE, dtype=np.float32)
    core = np.array(_BG_CORE, dtype=np.float32)
    img = core + (edge - core) * tone[..., None]
    surf = pygame.surfarray.make_surface(img.astype(np.uint8))
    _background._cache = surf
    return surf


def _sphere(color, diameter: int, alpha: int = 255):
    import pygame

    from .sdl_thread import require_main_thread

    require_main_thread()
    diameter = max(4, int(diameter))
    key = (tuple(color), diameter, int(alpha))
    cache = getattr(_sphere, "_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit
    radius = diameter / 2.0
    yy, xx = np.ogrid[:diameter, :diameter]
    x = (xx - radius + 0.5) / radius
    y = (yy - radius + 0.5) / radius
    rr = x * x + y * y
    nz = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))
    light = np.clip((-x) * 0.42 + (-y) * 0.62 + nz * 0.78, 0.0, 1.0)
    body = 0.18 + 0.82 * light
    spec = np.clip(light, 0.0, 1.0) ** 18
    inside = rr <= 1.0
    rgb = np.zeros((diameter, diameter, 3), dtype=np.float32)
    for channel in range(3):
        channel_px = np.clip(color[channel] * body + 255.0 * spec * 0.75, 0, 255)
        rgb[:, :, channel] = np.where(inside, channel_px, 0.0)
    surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    view = pygame.surfarray.pixels3d(surf)
    try:
        view[:] = np.transpose(rgb.astype(np.uint8), (1, 0, 2))
    finally:
        del view
    mask = pygame.surfarray.pixels_alpha(surf)
    try:
        mask[:] = np.where(rr.T <= 1.0, int(alpha), 0).astype(np.uint8)
    finally:
        del mask
    cache[key] = surf
    _sphere._cache = cache
    return surf


def _glow(color, diameter: int):
    import pygame

    from .sdl_thread import require_main_thread

    require_main_thread()
    diameter = max(6, int(diameter))
    key = (tuple(color), diameter)
    cache = getattr(_glow, "_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit
    radius = diameter / 2.0
    yy, xx = np.ogrid[:diameter, :diameter]
    x = (xx - radius + 0.5) / radius
    y = (yy - radius + 0.5) / radius
    rr = np.sqrt(x * x + y * y)
    # BLEND_ADD ignores per-pixel alpha, so the falloff has to live in RGB.
    fall = np.clip(1.0 - rr, 0.0, 1.0) ** 2.1
    rgb = np.zeros((diameter, diameter, 3), dtype=np.float32)
    for channel in range(3):
        rgb[:, :, channel] = color[channel] * fall * 0.34
    surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    view = pygame.surfarray.pixels3d(surf)
    try:
        view[:] = np.transpose(rgb.astype(np.uint8), (1, 0, 2))
    finally:
        del view
    mask = pygame.surfarray.pixels_alpha(surf)
    try:
        mask[:] = (fall.T * 255).astype(np.uint8)
    finally:
        del mask
    cache[key] = surf
    _glow._cache = cache
    return surf


def _scale(color, k: float):
    return tuple(int(max(0, min(255, round(channel * k)))) for channel in color)


def paint(surface, cloud: BrainCloud, packet: dict, yaw: float, pitch: float, dist: float, auto: bool = True, pan_x: float = 0.0, pan_y: float = 0.0, frames: bool = False, fast: bool = False, hint: str | None = None, software_only: bool = False) -> None:
    """Perspective view. FlyWire shell when the cache exists, else the soma cloud."""
    from .mb_flywire import paint_scene
    from .sdl_thread import require_main_thread

    require_main_thread()

    if paint_scene(surface, packet, yaw, pitch, dist, auto=auto, pan_x=pan_x, pan_y=pan_y, frames=frames, fast=fast, hint=hint, software_only=software_only):
        return
    import pygame

    w, h = surface.get_size()
    surface.blit(_background(w, h), (0, 0))
    font = pygame.font.Font(None, 20)
    small = pygame.font.Font(None, 16)
    title = "грибовидное тело"
    surface.blit(font.render(title, True, _INK), (16, 12))
    note = "Л/П полушария   серые KC   свечение — кадр   янтарь MBON   зелёный PAM   красный PPL1"
    surface.blit(small.render(note, True, _INK), (16, 34))
    surface.blit(small.render("чашечка, ножка, доли α/β α′/β′ γ", True, _INK), (16, 52))
    hint = "ЛКМ обзор   ПКМ/СКМ/Shift сдвиг   колёсико зум   стрелки WASD   Home сброс   F каркас"
    if frames:
        hint += " вкл"
    tiny = pygame.font.Font(None, 15)
    surface.blit(tiny.render(hint, True, (168, 170, 176)), (12, h - 20))
    cx, cy, fov = w * 0.50, h * 0.54, min(w, h) * 1.05
    kc_l = _flags(cloud.n_kc, packet.get("kc_l"))
    kc_r = _flags(cloud.n_kc, packet.get("kc_r"))
    r_l = float(packet.get("r_l") or 0.0)
    r_r = float(packet.get("r_r") or 0.0)
    flash_l = str(packet.get("flash_l") or "")
    flash_r = str(packet.get("flash_r") or "")
    rec_l = bool(packet.get("rec_l"))
    rec_r = bool(packet.get("rec_r"))
    sx, sy, depth = _project(cloud.kc_pos, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    fog = _depth_t(depth)
    _draw_grid(surface, cloud, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    if frames:
        _draw_hulls(surface, cloud, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    idle = pygame.Surface((w, h), pygame.SRCALPHA)
    glow = pygame.Surface((w, h), pygame.SRCALPHA)
    order = np.argsort(-depth)
    for i in order:
        if depth[i] <= 0.3:
            continue
        x, y = int(sx[i]), int(sy[i])
        if x < -8 or y < -8 or x >= w + 8 or y >= h + 8:
            continue
        far = float(fog[i])
        left_on = bool(kc_l[i])
        right_on = bool(kc_r[i])
        if left_on or right_on:
            if left_on and right_on:
                color = _mix(_GLOW_L, _GLOW_R, 0.5)
            elif left_on:
                color = _GLOW_L
            else:
                color = _GLOW_R
            color = _fog(color, far * 0.28)
            radius = 2 if far < 0.55 else 1
            pygame.draw.circle(glow, (*_scale(color, 0.16), 255), (x, y), radius + 1)
            pygame.draw.circle(glow, (*_scale(color, 0.38), 255), (x, y), radius)
        else:
            color = _fog(_KC_IDLE, far * 0.22)
            alpha = int(168 - 36 * far)
            radius = 2
            pygame.draw.circle(idle, (*color, max(90, alpha)), (x, y), radius)
    # Calyx is the posterior cup. Draw a thin sample so the wire box is not empty.
    if len(cloud.calyx_pos):
        csx, csy, cdep = _project(cloud.calyx_pos[::2], yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
        for i in range(len(cdep)):
            if cdep[i] <= 0.3:
                continue
            x, y = int(csx[i]), int(csy[i])
            if x < -4 or y < -4 or x >= w + 4 or y >= h + 4:
                continue
            pygame.draw.circle(idle, (*_KC_IDLE, 110), (x, y), 2)
    idle.set_alpha(235)
    surface.blit(idle, (0, 0))
    surface.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)
    _draw_spheres(surface, cloud.mbon_pos, cloud.mbon_side, "mbon", r_l, r_r, flash_l, flash_r, yaw, pitch, dist, cx, cy, fov, w, h, pan_x, pan_y)
    _draw_spheres(surface, cloud.pam_pos, cloud.pam_side, "pam", r_l, r_r, flash_l, flash_r, yaw, pitch, dist, cx, cy, fov, w, h, pan_x, pan_y)
    _draw_spheres(surface, cloud.ppl_pos, cloud.ppl_side, "ppl1", r_l, r_r, flash_l, flash_r, yaw, pitch, dist, cx, cy, fov, w, h, pan_x, pan_y)
    _draw_edges(surface, cloud, packet, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    _draw_eyes(surface, cloud, rec_l, rec_r, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    _draw_side_labels(surface, cloud, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)


def _fog(color, t: float):
    return _mix(color, _BG_EDGE, max(0.0, min(0.85, float(t))))


def _draw_grid(surface, cloud, yaw, pitch, dist, cx, cy, fov, pan_x: float = 0.0, pan_y: float = 0.0) -> None:
    import pygame

    floor = float(np.percentile(cloud.kc_pos[:, 1], 4) - 0.18)
    lines = []
    for i in range(-3, 4):
        x = i * 0.42
        lines.append((np.array([[x, floor, -0.2], [x, floor, 1.7]]),))
        z = -0.2 + i * 0.32
        lines.append((np.array([[-1.5, floor, z], [1.5, floor, z]]),))
    color = (58, 62, 70)
    for (pts,) in lines:
        px, py, dep = _project(pts, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
        if dep[0] <= 0.3 or dep[1] <= 0.3:
            continue
        pygame.draw.line(surface, color, (int(px[0]), int(py[0])), (int(px[1]), int(py[1])), 1)


def _draw_hulls(surface, cloud, yaw, pitch, dist, cx, cy, fov, pan_x: float = 0.0, pan_y: float = 0.0) -> None:
    import pygame

    layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for _side, lobe, segments in cloud.hulls:
        if lobe < 0:
            color = (120, 124, 132, 50)
        else:
            rgb = _WIRE[int(lobe)]
            color = (*rgb, 78)
        for a, b in segments:
            pair = np.vstack((a, b)).astype(np.float64)
            px, py, dep = _project(pair, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
            if dep[0] <= 0.3 or dep[1] <= 0.3:
                continue
            pygame.draw.line(layer, color, (int(px[0]), int(py[0])), (int(px[1]), int(py[1])), 1)
    surface.blit(layer, (0, 0))


def _draw_spheres(surface, pos, side, kind, r_l, r_r, flash_l, flash_r, yaw, pitch, dist, cx, cy, fov, w, h, pan_x: float = 0.0, pan_y: float = 0.0) -> None:
    if len(pos) == 0:
        return
    import pygame

    px, py, dep = _project(pos, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    fog = _depth_t(dep)
    order = np.argsort(-dep)
    halo_pts = {0: [], 1: []}
    for i in order:
        if dep[i] <= 0.3:
            continue
        x, y = int(px[i]), int(py[i])
        if x < -20 or y < -20 or x >= w + 20 or y >= h + 20:
            continue
        far = float(fog[i])
        anatomical_left = int(side[i]) == 1
        readout = r_l if anatomical_left else r_r
        flash = flash_l if anatomical_left else flash_r
        if kind == "mbon":
            color = _fog(_AMBER, far * 0.22)
            hot = _gain(readout)
            diameter = int(round(7 + 3 * hot - 2 * far))
            alpha = int(210 + 30 * hot)
        elif kind == "pam":
            hot = flash == "pam"
            color = _PAM if hot else _fog((78, 96, 84), far * 0.25)
            diameter = 4 if hot else int(round(4 - far))
            alpha = 150 if hot else 130
            if hot:
                halo_pts[1 if anatomical_left else 0].append((x, y))
        else:
            hot = flash == "ppl1"
            color = _PPL if hot else _fog((96, 74, 74), far * 0.25)
            diameter = 4 if hot else int(round(4 - far))
            alpha = 150 if hot else 130
            if hot:
                halo_pts[1 if anatomical_left else 0].append((x, y))
        diameter = max(4, diameter)
        sprite = _sphere(color, diameter, min(255, alpha))
        surface.blit(sprite, (x - diameter // 2, y - diameter // 2))
    halo_color = _PAM if kind == "pam" else (_PPL if kind == "ppl1" else _AMBER)
    for pts in halo_pts.values():
        if len(pts) < 3:
            continue
        hx = int(sum(p[0] for p in pts) / len(pts))
        hy = int(sum(p[1] for p in pts) / len(pts))
        halo = _glow(halo_color, 28)
        surface.blit(halo, (hx - halo.get_width() // 2, hy - halo.get_height() // 2), special_flags=pygame.BLEND_ADD)


def _draw_edges(surface, cloud, packet, yaw, pitch, dist, cx, cy, fov, pan_x: float = 0.0, pan_y: float = 0.0) -> None:
    import pygame

    edges = packet.get("edges") or []
    if not edges:
        return
    layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for item in edges[:EDGE_CAP]:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        pre, post, weight = int(item[0]), int(item[1]), float(item[2])
        if pre < 0 or pre >= cloud.n_kc or post < 0 or post >= len(cloud.mbon_pos):
            continue
        pair = np.vstack((cloud.kc_pos[pre], cloud.mbon_pos[post]))
        px, py, dep = _project(pair, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
        if dep[0] <= 0.3 or dep[1] <= 0.3:
            continue
        color = (*_AMBER, 90) if weight >= 0 else (*_PPL, 90)
        pygame.draw.line(layer, color, (int(px[0]), int(py[0])), (int(px[1]), int(py[1])), 1)
    surface.blit(layer, (0, 0))


def eye_layout(cloud: BrainCloud):
    """Eyes sit lateral to their own hemisphere and look anterior (+z).

    Anatomical left is «глаз Л» / MB_L, the same side as the trainer plaque Л
    (image right, the dog's left eye). Image left is «глаз П».
    """
    eyes = []
    for side, label in ((1, "глаз Л"), (0, "глаз П")):
        mask = cloud.kc_side == side
        if not np.any(mask):
            continue
        cal = np.asarray(cloud.calyx_pos[mask].mean(axis=0), dtype=np.float64)
        kc = cloud.kc_pos[mask]
        lateral = -1.0 if side == 1 else 1.0
        edge = float(kc[:, 0].min() if side == 1 else kc[:, 0].max())
        pos = np.array([
            edge + lateral * 0.42,
            float(cal[1]) + 0.10,
            float(cal[2]) + 0.70,
        ], dtype=np.float64)
        eyes.append({"side": side, "label": label, "pos": pos, "calyx": cal})
    if not eyes:
        return eyes, None
    mid_z = float(np.mean([item["pos"][2] for item in eyes]))
    top = float(np.percentile(cloud.kc_pos[:, 1], 96))
    start = np.array([0.0, top + 0.28, mid_z - 0.15], dtype=np.float64)
    tip = start + np.array([0.0, 0.0, 0.85], dtype=np.float64)
    return eyes, (start, tip)


def _dashed(surface, start, end, color, dash: int = 7, gap: int = 5) -> None:
    import pygame

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 2.0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    travel = 0.0
    while travel < length:
        stop = min(length, travel + dash)
        pygame.draw.line(
            surface,
            color,
            (int(x0 + ux * travel), int(y0 + uy * travel)),
            (int(x0 + ux * stop), int(y0 + uy * stop)),
            1,
        )
        travel += dash + gap


def _draw_eyes(surface, cloud, rec_l: bool, rec_r: bool, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y) -> None:
    import pygame

    eyes, arrow = eye_layout(cloud)
    if not eyes:
        return
    font = pygame.font.Font(None, 18)
    link = (138, 146, 158)
    for item in eyes:
        pair = np.vstack((item["pos"], item["calyx"]))
        px, py, dep = _project(pair, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
        if dep[0] > 0.3 and dep[1] > 0.3:
            _dashed(surface, (px[0], py[0]), (px[1], py[1]), link)
    for item in eyes:
        recognized = rec_l if item["side"] == 1 else rec_r
        pos = item["pos"].reshape(1, 3)
        gaze = (item["pos"] + np.array([0.0, 0.0, 0.22])).reshape(1, 3)
        px, py, dep = _project(pos, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
        gx, gy, _gdep = _project(gaze, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
        if dep[0] <= 0.3:
            continue
        x, y = int(px[0]), int(py[0])
        body = (168, 196, 214) if recognized else (118, 132, 146)
        if recognized:
            halo = _glow((150, 196, 220), 34)
            surface.blit(halo, (x - halo.get_width() // 2, y - halo.get_height() // 2), special_flags=pygame.BLEND_ADD)
        sprite = _sphere(body, 16, 230 if recognized else 200)
        surface.blit(sprite, (x - 8, y - 8))
        dx, dy = float(gx[0]) - float(px[0]), float(gy[0]) - float(py[0])
        norm = math.hypot(dx, dy) or 1.0
        pupil = (int(x + dx / norm * 3.0), int(y + dy / norm * 3.0))
        pygame.draw.circle(surface, (28, 34, 42), pupil, 2)
        label = font.render(item["label"], True, _INK)
        lateral = -1 if item["side"] == 1 else 1
        surface.blit(label, (x + lateral * 14 - (label.get_width() if lateral < 0 else 0), y - 8))
    if arrow is None:
        return
    pts = np.vstack(arrow)
    px, py, dep = _project(pts, yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
    if dep[0] <= 0.3 or dep[1] <= 0.3:
        return
    color = (214, 218, 224)
    x0, y0 = int(px[0]), int(py[0])
    x1, y1 = int(px[1]), int(py[1])
    pygame.draw.line(surface, color, (x0, y0), (x1, y1), 1)
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    left = (int(x1 - ux * 8 - uy * 4), int(y1 - uy * 8 + ux * 4))
    right = (int(x1 - ux * 8 + uy * 4), int(y1 - uy * 8 - ux * 4))
    pygame.draw.line(surface, color, (x1, y1), left, 1)
    pygame.draw.line(surface, color, (x1, y1), right, 1)
    caption = font.render("вперёд", True, _INK)
    surface.blit(caption, (x1 - caption.get_width() // 2, y1 - caption.get_height() - 4))


def _draw_side_labels(surface, cloud, yaw, pitch, dist, cx, cy, fov, pan_x: float = 0.0, pan_y: float = 0.0) -> None:
    import pygame

    font = pygame.font.Font(None, 36)
    for side, text in ((1, "Л"), (0, "П")):
        mask = cloud.kc_side == side
        if not np.any(mask):
            continue
        mean = cloud.kc_pos[mask].mean(axis=0).copy()
        mean[0] += -0.26 if side == 1 else 0.26
        mean[1] += 0.22
        px, py, dep = _project(mean.reshape(1, 3), yaw, pitch, dist, cx, cy, fov, pan_x, pan_y)
        if dep[0] <= 0.3:
            continue
        label = font.render(text, True, _INK)
        surface.blit(label, (int(px[0]) - label.get_width() // 2, int(py[0]) - label.get_height() // 2))


def render_frame(cloud: BrainCloud, packet: dict, size=(960, 700), yaw: float = YAW0, pitch: float = PITCH0, dist: float | None = None, auto: bool = True, pan_x: float = 0.0, pan_y: float = 0.0, frames: bool = False, fast: bool = False):
    import pygame

    from .mb_flywire import load_scene
    from .sdl_thread import require_main_thread

    require_main_thread()

    if dist is None:
        scene = load_scene()
        dist = float(scene.fit_dist) if scene is not None else DIST0
    pygame.font.init()
    surface = pygame.Surface(size)
    paint(surface, cloud, packet, yaw, pitch, dist, auto=auto, pan_x=pan_x, pan_y=pan_y, frames=frames, fast=fast)
    return surface


def demo_packet(cloud: BrainCloud) -> dict:
    left = np.flatnonzero(cloud.kc_side == 1)
    right = np.flatnonzero(cloud.kc_side == 0)
    return {
        "t": 0.0,
        "r_l": 80.0,
        "r_r": -30.0,
        "kc_l": [int(i) for i in left[::40][:KC_CAP]],
        "kc_r": [int(i) for i in right[::40][:KC_CAP]],
        "flash_l": "pam",
        "flash_r": "ppl1",
        "rec_l": True,
        "rec_r": False,
        "edges": [],
    }


def viewer_command(port: int = VIEW_PORT) -> list:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--mb-view3d", "--port", str(int(port))]
    return [sys.executable, "-m", "sim.mb_view3d", "--port", str(int(port))]


class BrainView:
    """Owned by the trainer. Spawn and send never wait on a frame."""

    def __init__(self, port: int = VIEW_PORT):
        self.port = int(port)
        self.proc = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self._next = 0.0

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _ensure_sock(self) -> None:
        if self.sock is not None:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def _stop_proc(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            proc.kill()

    def toggle(self) -> str:
        if self.alive:
            self._stop_proc()
            return "окно мозга закрыто"
        self.spawn()
        return "окно мозга открыто"

    def spawn(self) -> None:
        self._stop_proc()
        self._ensure_sock()
        self.proc = subprocess.Popen(viewer_command(self.port), stdin=subprocess.DEVNULL)
        self._next = 0.0

    def send(self, packet: dict, now: float | None = None, force: bool = False) -> None:
        if self.sock is None or (not force and not self.alive):
            return
        now = time.monotonic() if now is None else float(now)
        if not force and now < self._next:
            return
        self._next = now + (1.0 / RATE_HZ)
        raw = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        if len(raw) > 60000:
            return
        try:
            self.sock.sendto(raw, (HOST, self.port))
        except OSError:
            pass

    def close(self) -> None:
        self._stop_proc()
        sock = self.sock
        self.sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _sample_edges(brain, active, side: int) -> list:
    if brain is None or not active:
        return []
    ids = np.array(active, dtype=np.int32, copy=True)
    mask = np.isin(brain.kc_mbon_pre, ids)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    if idx.size > EDGE_CAP:
        idx = idx[:EDGE_CAP]
    out = []
    for i in idx:
        out.append([int(brain.kc_mbon_pre[i]), int(brain.kc_mbon_post[i]), round(float(brain.kc_mbon_w[i]), 4), int(side)])
    return out


def _flash_kind(session, side: str, now: float) -> str:
    show = str(getattr(session, "teacher_flash_%s_show" % side, "") or "")
    if show in ("pam", "ppl1"):
        return show
    mb = getattr(session, "mb", None)
    flash = None
    if mb is not None:
        flash = mb.flash if side == "l" else getattr(mb, "flash_r", None)
    if flash is not None and str(getattr(flash, "kind", "")) in ("pam", "ppl1"):
        if now - float(getattr(flash, "t", -1e9)) <= 0.5:
            return str(flash.kind)
    if not getattr(session, "onboard", False):
        return ""
    remote = getattr(session, "remote", None)
    if not isinstance(remote, dict):
        return ""
    data = remote.get("flash" if side == "l" else "flash_r")
    if not isinstance(data, dict):
        return ""
    kind = str(data.get("kind") or "")
    if kind not in ("pam", "ppl1"):
        return ""
    token = (kind, data.get("t"), data.get("n_syn"))
    seen = getattr(session, "_brain_flash_seen", None)
    if seen is None:
        seen = {}
        session._brain_flash_seen = seen
    untils = getattr(session, "_brain_flash_until", None)
    if untils is None:
        untils = {}
        session._brain_flash_until = untils
    if seen.get(side) != token:
        seen[side] = token
        untils[side] = now + 0.5
    if now <= float(untils.get(side) or 0.0):
        return kind
    return ""


def _as_ids(value) -> list:
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value[:KC_CAP]:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def packet_from_session(session, now: float) -> dict:
    """KC lists, readouts, compartment flashes. Weights only from a local brain."""
    remote = getattr(session, "remote", None) if getattr(session, "onboard", False) else None
    mb = getattr(session, "mb", None)
    edges: list = []
    if isinstance(remote, dict) and remote:
        kc_l = _as_ids(remote.get("kc_l"))
        kc_r = _as_ids(remote.get("kc_r"))
        r_l = float(remote.get("r_l") or 0.0)
        r_r = float(remote.get("r_r") or 0.0)
    elif mb is not None:
        kc_l = compact_ids(None if mb.last_fwd is None else mb.last_fwd.kc)
        kc_r = compact_ids(None if mb.last_fwd_r is None else mb.last_fwd_r.kc)
        r_l = float(mb.r_l)
        r_r = float(mb.r_r)
        edges = _sample_edges(mb.brain, kc_l, 0) + _sample_edges(mb.brain_r, kc_r, 1)
        edges = edges[:EDGE_CAP]
    else:
        kc_l, kc_r, r_l, r_r = [], [], 0.0, 0.0
    return {
        "t": float(now),
        "r_l": r_l,
        "r_r": r_r,
        "kc_l": kc_l,
        "kc_r": kc_r,
        "flash_l": _flash_kind(session, "l", now),
        "flash_r": _flash_kind(session, "r", now),
        "rec_l": _hemi_recognized(session, "l"),
        "rec_r": _hemi_recognized(session, "r"),
        "edges": edges,
    }


def _hemi_recognized(session, side: str) -> bool:
    """Same bits as the trainer plaques. Л is the dog's left eye."""
    name = "eye_l_recognized" if side == "l" else "eye_r_recognized"
    if hasattr(session, name):
        return bool(getattr(session, name))
    mb = getattr(session, "mb", None)
    if mb is not None and hasattr(mb, "eye_recognized"):
        try:
            left, right = mb.eye_recognized()
        except (TypeError, ValueError):
            return False
        return bool(left if side == "l" else right)
    remote = getattr(session, "remote", None) if getattr(session, "onboard", False) else None
    if isinstance(remote, dict):
        return bool(remote.get("recognized_L" if side == "l" else "recognized_R", False))
    return False


def drive_brain(session, inp, now: float, mon=None) -> None:
    if getattr(session, "learner_kind", "") != "mb":
        return
    link = getattr(session, "_brain_view", None)
    if link is None:
        link = BrainView()
        session._brain_view = link
    if getattr(inp, "brain_toggle", False):
        session._log(link.toggle())
    packet = packet_from_session(session, now)
    if link.alive:
        link.send(packet, now)
    if mon is not None:
        mon.brain_open = bool(link.alive)
        mon.brain_packet = packet


def close_brain(session) -> None:
    link = getattr(session, "_brain_view", None)
    if link is not None:
        link.close()


def _recv_latest(sock) -> dict | None:
    latest = None
    while True:
        try:
            raw, _addr = sock.recvfrom(65535)
        except BlockingIOError:
            break
        except OSError:
            break
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            latest = data
    return latest


def _saved_viewer_size() -> tuple:
    """Restore the 3D window. Default 960×700, never below 320×240."""
    import os

    from .ui_settings import load_brain_window

    geom = load_brain_window()
    try:
        width = max(320, int(geom.get("w") or 960))
    except (TypeError, ValueError):
        width = 960
    try:
        height = max(240, int(geom.get("h") or 700))
    except (TypeError, ValueError):
        height = 700
    x, y = geom.get("x"), geom.get("y")
    if x is not None and y is not None:
        try:
            os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (int(x), int(y))
        except (TypeError, ValueError):
            pass
    return width, height


def _remember_viewer(window) -> None:
    from .ui_settings import save_brain_window

    box = {"x": None, "y": None, "w": max(320, int(window.get_width())), "h": max(240, int(window.get_height()))}
    try:
        from pygame._sdl2.video import Window

        win = Window.from_display_module()
        pos = win.position
        size = win.size
        box["x"], box["y"] = int(pos[0]), int(pos[1])
        box["w"], box["h"] = max(320, int(size[0])), max(240, int(size[1]))
    except Exception:
        pass
    save_brain_window(box)


def run_viewer(port: int = VIEW_PORT) -> int:
    import os

    import pygame

    from .sdl_thread import require_main_thread

    require_main_thread()
    pygame.init()
    pygame.display.set_caption("Грибовидное тело")
    width, height = _saved_viewer_size()
    window = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    window.fill(_BG_EDGE)
    window.blit(font.render("загрузка координат FlyWire…", True, _INK), (24, 24))
    pygame.display.flip()
    cloud = BrainCloud()
    from .mb_flywire import load_scene

    scene = load_scene()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((HOST, int(port)))
    except OSError as exc:
        print("окно мозга: порт %s занят (%s)" % (port, exc), file=sys.stderr)
        try:
            sock.close()
        except OSError:
            pass
        return 1
    sock.setblocking(False)
    packet = {"r_l": 0.0, "r_r": 0.0, "kc_l": [], "kc_r": [], "flash_l": "", "flash_r": "", "rec_l": False, "rec_r": False, "edges": []}
    orbit = Orbit()
    if scene is not None:
        orbit.home_dist = float(scene.fit_dist)
        orbit.dist = float(scene.fit_dist)
    frames = [False]
    drag = None
    shot = os.environ.get("MB_VIEW_SHOT") or ""
    shot_done = False
    pygame.key.set_repeat(180, 40)
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_j):
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    _keys(orbit, event, frames)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 2, 3):
                    shifted = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    drag = ("pan" if event.button != 1 or shifted else "orbit", event.pos)
                    orbit.idle = 0.0
                elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 2, 3):
                    drag = None
                elif event.type == pygame.MOUSEMOTION and drag is not None:
                    dx = event.pos[0] - drag[1][0]
                    dy = event.pos[1] - drag[1][1]
                    shifted = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    mode = "pan" if drag[0] == "pan" or shifted or event.buttons[1] or event.buttons[2] else "orbit"
                    drag = (mode, event.pos)
                    if mode == "pan":
                        orbit.pan_pixels(dx, dy)
                    elif event.buttons[0]:
                        orbit.drag(dx, dy)
                elif event.type == pygame.MOUSEWHEEL:
                    orbit.zoom(event.y)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
                    orbit.zoom(1 if event.button == 4 else -1)
                elif event.type == pygame.VIDEORESIZE:
                    window = pygame.display.set_mode((max(320, event.w), max(240, event.h)), pygame.RESIZABLE)
                    _remember_viewer(window)
                elif event.type == getattr(pygame, "WINDOWMOVED", -11):
                    _remember_viewer(window)
            if not running:
                break
            fresh = _recv_latest(sock)
            if fresh is not None:
                packet = fresh
            dt = clock.tick(30) / 1000.0
            orbit.tick(dt, drag is not None)
            paint(window, cloud, packet, orbit.yaw, orbit.pitch, orbit.dist, auto=orbit.auto, pan_x=orbit.pan_x, pan_y=orbit.pan_y, frames=frames[0], fast=True)
            pygame.display.flip()
            if shot and not shot_done:
                try:
                    dest = Path(shot)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    pygame.image.save(window, str(dest))
                except OSError:
                    pass
                shot_done = True
    finally:
        try:
            _remember_viewer(window)
        except Exception:
            pass
        try:
            sock.close()
        except OSError:
            pass
    return 0


def _keys(orbit: Orbit, event, frames=None) -> None:
    """Camera keys for this window only. Trainer hotkeys live in the other process."""
    import pygame

    key = event.key
    repeat = bool(getattr(event, "repeat", False))
    if key == pygame.K_f and not repeat:
        if frames is not None:
            frames[0] = not frames[0]
        return
    step = PAN_KEY
    if key in (pygame.K_LEFT, pygame.K_a):
        orbit.nudge(-step, 0.0)
    elif key in (pygame.K_RIGHT, pygame.K_d):
        orbit.nudge(step, 0.0)
    elif key in (pygame.K_UP, pygame.K_w, pygame.K_PAGEUP):
        orbit.nudge(0.0, step)
    elif key in (pygame.K_DOWN, pygame.K_s, pygame.K_PAGEDOWN):
        orbit.nudge(0.0, -step)
    elif key in (pygame.K_HOME, pygame.K_r) and not repeat:
        orbit.reset()


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="3D mushroom body window")
    parser.add_argument("--mb-view3d", action="store_true")
    parser.add_argument("--port", type=int, default=VIEW_PORT)
    parser.add_argument("--shot", default="")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)
    if args.shot:
        import pygame

        pygame.init()
        cloud = BrainCloud()
        packet = demo_packet(cloud)
        frame = render_frame(cloud, packet)
        dest = Path(args.shot)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(frame, str(dest))
        return 0
    return run_viewer(args.port)


if __name__ == "__main__":
    sys.exit(main())
