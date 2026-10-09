"""Egocentric camera + lidar features with no conspecific label.

Pixels and range bins are projected by a fixed random sparse map onto PN.
Nothing in the vector is named 'dog'; agents and distractors differ only by
the raw pattern they produce (body row vs floor row, size, speed, color).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_AZ = 8
FOV = np.deg2rad(110.0)
SEE_M = 3.2
N_RAW = 72  # body RGB 24 + floor RGB 24 + motion 8 + lidar near 8 + closing 8

OFF_BODY = 0
OFF_FLOOR = 24
OFF_MOTION = 48
OFF_LIDAR = 56
OFF_CLOSE = 64

RECOG_CAMERA_LABEL = "узнавание: только камера"
RECOG_LIDAR_LABEL = "узнавание: камера и лидар"


def drop_lidar(feat: np.ndarray) -> np.ndarray:
    """Copy with lidar near and closing cleared. Camera channels stay.

    The caller keeps the original: distance, the 1 m stop, and the map still
    read those channels. Recognition must not.
    """
    raw = np.asarray(feat, dtype=np.float32).ravel()
    out = np.array(raw, dtype=np.float32, copy=True)
    out[OFF_LIDAR:N_RAW] = 0.0
    return out


def recog_label(camera_only: bool) -> str:
    return RECOG_CAMERA_LABEL if camera_only else RECOG_LIDAR_LABEL


def add_recog_camera_only_arg(parser) -> None:
    """Default on. ``--no-recog-camera-only`` puts lidar back into PN→KC."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--recog-camera-only",
        dest="recog_camera_only",
        action="store_true",
        help="Zero lidar channels before PN→KC. Default.",
    )
    group.add_argument(
        "--no-recog-camera-only",
        dest="recog_camera_only",
        action="store_false",
        help="Feed lidar near and closing into recognition as well.",
    )
    parser.set_defaults(recog_camera_only=True)


@dataclass
class SenseHit:
    dog: bool
    distractor: bool


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def bearing_bin(dx: float, dy: float, yaw: float) -> int | None:
    ang = _wrap(float(np.arctan2(dy, dx)) - yaw)
    if abs(ang) > FOV / 2:
        return None
    u = (ang + FOV / 2) / FOV
    return int(np.clip(np.floor(u * N_AZ), 0, N_AZ - 1))


def wall_range(x: float, y: float, yaw: float, bin_i: int, width: float, height: float) -> float:
    ang = yaw + (-FOV / 2 + (bin_i + 0.5) * FOV / N_AZ)
    dx, dy = float(np.cos(ang)), float(np.sin(ang))
    hits = []
    if dx > 1e-6:
        hits.append((width - x) / dx)
    elif dx < -1e-6:
        hits.append((0.0 - x) / dx)
    if dy > 1e-6:
        hits.append((height - y) / dy)
    elif dy < -1e-6:
        hits.append((0.0 - y) / dy)
    positive = [t for t in hits if t > 0]
    return float(min(positive)) if positive else SEE_M


def _paint_object(feat: np.ndarray, *, bx: int, dist: float, radius: float, color, speed: float, closing: float, body: bool) -> None:
    near = float(np.clip(1.0 - dist / SEE_M, 0.0, 1.0))
    if near <= 0:
        return
    sector = FOV / N_AZ
    half = int(np.floor((radius / max(dist, 1e-3)) / sector))
    rgb_off = OFF_BODY if body else OFF_FLOOR
    for bi in range(max(0, bx - half), min(N_AZ, bx + half + 1)):
        for c in range(3):
            feat[rgb_off + bi * 3 + c] = max(feat[rgb_off + bi * 3 + c], float(color[c]) * near)
        feat[OFF_MOTION + bi] = max(feat[OFF_MOTION + bi], float(np.clip(speed / 0.8, 0.0, 1.0)) * near)
        feat[OFF_LIDAR + bi] = max(feat[OFF_LIDAR + bi], near)
        feat[OFF_CLOSE + bi] = max(feat[OFF_CLOSE + bi], float(np.clip(closing / 0.8, 0.0, 1.0)) * near)


def render_view(agent, world, include_agents: bool = True) -> tuple[np.ndarray, SenseHit]:
    """Coarse onboard view. `include_agents=False` is the blind-peer control."""
    feat = np.zeros(N_RAW, dtype=np.float32)
    dog = False
    distractor = False
    yaw = float(agent.yaw)
    svx, svy = float(getattr(agent, "vx", 0.0)), float(getattr(agent, "vy", 0.0))

    if include_agents:
        for other in world.agents:
            if other.agent_id == agent.agent_id:
                continue
            dx, dy = other.x - agent.x, other.y - agent.y
            dist = float(np.hypot(dx, dy))
            bx = bearing_bin(dx, dy, yaw)
            if bx is None or dist > SEE_M or dist < 1e-3:
                continue
            rel_vx = float(getattr(other, "vx", 0.0)) - svx
            rel_vy = float(getattr(other, "vy", 0.0)) - svy
            closing = -((rel_vx * dx + rel_vy * dy) / dist)
            color = tuple(c / 255.0 for c in other.color)
            _paint_object(
                feat,
                bx=bx,
                dist=dist,
                radius=0.28,
                color=color,
                speed=float(np.hypot(getattr(other, "vx", 0.0), getattr(other, "vy", 0.0))),
                closing=closing,
                body=True,
            )
            dog = True

    for obj in getattr(world, "distractors", []):
        dx, dy = obj.x - agent.x, obj.y - agent.y
        dist = float(np.hypot(dx, dy))
        bx = bearing_bin(dx, dy, yaw)
        if bx is None or dist > SEE_M or dist < 1e-3:
            continue
        rel_vx, rel_vy = obj.vx - svx, obj.vy - svy
        closing = -((rel_vx * dx + rel_vy * dy) / dist)
        _paint_object(
            feat,
            bx=bx,
            dist=dist,
            radius=obj.radius,
            color=obj.color,
            speed=float(np.hypot(obj.vx, obj.vy)),
            closing=closing,
            body=False,
        )
        distractor = True

    # Floor color of the zone the body is standing on (still not a class label).
    for zone in world.zones.values():
        if not zone.contains(agent.x, agent.y):
            cx, cy = zone.centroid()
            dx, dy = cx - agent.x, cy - agent.y
            dist = float(np.hypot(dx, dy))
            bx = bearing_bin(dx, dy, yaw)
            if bx is None or dist > 4.0:
                continue
            gain = float(np.clip(1.0 - dist / 4.0, 0.0, 1.0)) * 0.35
            color = tuple(c / 255.0 for c in zone.color)
            for c in range(3):
                feat[OFF_FLOOR + bx * 3 + c] = max(feat[OFF_FLOOR + bx * 3 + c], color[c] * gain)
            continue
        color = tuple(c / 255.0 for c in zone.color)
        for bi in range(N_AZ):
            for c in range(3):
                feat[OFF_FLOOR + bi * 3 + c] = max(feat[OFF_FLOOR + bi * 3 + c], color[c] * 0.85)

    for bi in range(N_AZ):
        if feat[OFF_LIDAR + bi] > 0.15:
            continue
        dist = wall_range(agent.x, agent.y, yaw, bi, world.cfg.width, world.cfg.height)
        feat[OFF_LIDAR + bi] = max(feat[OFF_LIDAR + bi], float(np.clip(1.0 - dist / SEE_M, 0.0, 1.0)) * 0.25)

    return feat, SenseHit(dog=dog, distractor=distractor)


def probe_features(kind: str) -> np.ndarray:
    """Canonical object straight ahead, same range, no zone odor.

    `dog` fills the body row and moves faster. `distractor` fills only the
    floor row, is narrower and slower. Neither vector has a class id.
    """
    feat = np.zeros(N_RAW, dtype=np.float32)
    if kind == "empty":
        return feat
    bx = N_AZ // 2
    dist = 1.4
    if kind == "dog":
        _paint_object(
            feat,
            bx=bx,
            dist=dist,
            radius=0.28,
            color=(0.45, 0.62, 0.95),
            speed=0.45,
            closing=0.2,
            body=True,
        )
    elif kind == "distractor":
        _paint_object(
            feat,
            bx=bx,
            dist=dist,
            radius=0.12,
            color=(0.55, 0.38, 0.16),
            speed=0.12,
            closing=0.05,
            body=False,
        )
    else:
        raise ValueError(kind)
    return feat


class RawProjector:
    """Fixed random sparse feature → PN map. Same seed ⇒ same glomeruli."""

    def __init__(self, n_pn: int, seed: int, k: int = 10):
        rng = np.random.default_rng(seed)
        self.n_pn = n_pn
        self.idx = np.zeros((N_RAW, k), dtype=np.int32)
        self.w = np.zeros((N_RAW, k), dtype=np.float32)
        for i in range(N_RAW):
            self.idx[i] = rng.choice(n_pn, size=k, replace=False)
            self.w[i] = rng.uniform(0.6, 1.0, size=k).astype(np.float32)

    def project(self, features: np.ndarray) -> np.ndarray:
        pn = np.zeros(self.n_pn, dtype=np.float32)
        hot = np.nonzero(features > 0)[0]
        for i in hot:
            pn[self.idx[i]] += features[i] * self.w[i]
        np.clip(pn, 0.0, 1.0, out=pn)
        return pn
