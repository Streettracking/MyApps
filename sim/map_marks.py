"""Place conspecific marks on the lidar map from the same mushroom body.

The full frame still owns the «УЗНАЮ СОРОДИЧА» word. Localization is extra
read-only passes of that same KC→MBON readout on horizontal sectors. Nothing
here is a detector with its own weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mb_train import MbTrainer
from .raw_sense import FOV, N_AZ, N_RAW, OFF_BODY, OFF_CLOSE, OFF_FLOOR, OFF_LIDAR, OFF_MOTION, SEE_M

SECTOR_WIDTH = 3
FADE_S = 2.5
NEAR_POINT = 0.15
# Sim lidar draws SEE_M at 0.42 of the image half-width from the robot.
SIM_REACH = 0.42


def bin_angle(index: int) -> float:
    """Ego bearing of a sector center. Positive is left of forward. Bin 0 is right."""
    return float(-FOV / 2.0 + (index + 0.5) * FOV / N_AZ)


def sector_feature(feat: np.ndarray, center: int, width: int = SECTOR_WIDTH) -> np.ndarray:
    """Keep one horizontal window of the 72-d view and zero the other sectors."""
    src = np.asarray(feat, dtype=np.float32).ravel()
    out = np.zeros(N_RAW, dtype=np.float32)
    half = width // 2
    for b in range(max(0, center - half), min(N_AZ, center + half + 1)):
        out[OFF_BODY + b * 3 : OFF_BODY + b * 3 + 3] = src[OFF_BODY + b * 3 : OFF_BODY + b * 3 + 3]
        out[OFF_FLOOR + b * 3 : OFF_FLOOR + b * 3 + 3] = src[OFF_FLOOR + b * 3 : OFF_FLOOR + b * 3 + 3]
        out[OFF_MOTION + b] = src[OFF_MOTION + b]
        out[OFF_LIDAR + b] = src[OFF_LIDAR + b]
        out[OFF_CLOSE + b] = src[OFF_CLOSE + b]
    return out


def feature_range(feat: np.ndarray, center: int, width: int = SECTOR_WIDTH) -> float | None:
    """Meters from the lidar-near channels. None when that window has no return."""
    src = np.asarray(feat, dtype=np.float32).ravel()
    half = width // 2
    near = 0.0
    for b in range(max(0, center - half), min(N_AZ, center + half + 1)):
        near = max(near, float(src[OFF_LIDAR + b]))
    if near <= NEAR_POINT:
        return None
    return float((1.0 - near) * SEE_M)


def scan_range(scan: dict | None, center: int, width: int = SECTOR_WIDTH) -> float | None:
    if not scan:
        return None
    ranges = scan.get("ranges_m")
    if not isinstance(ranges, list):
        return None
    half = width // 2
    vals = []
    for b in range(max(0, center - half), min(len(ranges), center + half + 1)):
        if ranges[b] is not None:
            vals.append(float(ranges[b]))
    if not vals:
        return None
    return min(vals)


def _scan_ready(scan: dict | None) -> bool:
    if not scan:
        return False
    px = scan.get("robot_px")
    if not isinstance(px, (list, tuple)) or len(px) < 2:
        return False
    if scan.get("yaw_rad") is None or scan.get("meters_per_px") is None:
        return False
    try:
        float(scan["yaw_rad"])
        float(scan["meters_per_px"])
        float(px[0])
        float(px[1])
    except (TypeError, ValueError):
        return False
    return float(scan["meters_per_px"]) > 0


@dataclass
class MapMark:
    bin_i: int
    percent: float
    kind: str
    nx: float
    ny: float
    ox: float
    oy: float
    nx_l: float
    ny_l: float
    nx_r: float
    ny_r: float
    born: float = 0.0
    alpha: float = 1.0


def _ego_tip(ang: float, frac: float, ox: float = 0.5, oy: float = 0.5) -> tuple[float, float]:
    return ox - math.sin(ang) * frac, oy - math.cos(ang) * frac


def project_mark(
    index: int,
    percent: float,
    dist: float | None,
    *,
    mode: str,
    scan: dict | None,
    width: int = SECTOR_WIDTH,
) -> MapMark:
    ang = bin_angle(index)
    half = (width * (FOV / N_AZ)) / 2.0
    kind = "point" if dist is not None else "cone"
    if mode != "sim" and _scan_ready(scan):
        assert scan is not None
        yaw = float(scan["yaw_rad"])
        mpp = float(scan["meters_per_px"])
        rx, ry = float(scan["robot_px"][0]), float(scan["robot_px"][1])
        iw = float(scan.get("image_w") or 1.0)
        ih = float(scan.get("image_h") or 1.0)
        reach = dist if dist is not None else min(2.4, float(scan.get("span_m") or 8.0) * 0.35)

        def tip(delta: float) -> tuple[float, float]:
            world = yaw + ang + delta
            px = rx + math.cos(world) * reach / mpp
            py = ry - math.sin(world) * reach / mpp
            return px / iw, py / ih

        nx, ny = tip(0.0)
        nx_l, ny_l = tip(half)
        nx_r, ny_r = tip(-half)
        ox, oy = rx / iw, ry / ih
    else:
        if dist is None:
            frac = 0.38
        else:
            frac = float(np.clip(dist / SEE_M, 0.0, 1.0)) * SIM_REACH
        nx, ny = _ego_tip(ang, frac)
        nx_l, ny_l = _ego_tip(ang + half, frac if dist is not None else 0.38)
        nx_r, ny_r = _ego_tip(ang - half, frac if dist is not None else 0.38)
        ox, oy = 0.5, 0.5
    return MapMark(
        bin_i=index,
        percent=float(percent),
        kind=kind,
        nx=float(nx),
        ny=float(ny),
        ox=float(ox),
        oy=float(oy),
        nx_l=float(nx_l),
        ny_l=float(ny_l),
        nx_r=float(nx_r),
        ny_r=float(ny_r),
    )


def sector_table(trainer: MbTrainer, feat: np.ndarray, width: int = SECTOR_WIDTH) -> list[tuple[int, float, float]]:
    """Per-sector (bin, readout, percent). Does not teach or move the calibrator."""
    rows = []
    for index in range(N_AZ):
        readout = trainer.score_feature(sector_feature(feat, index, width))
        percent = trainer.cal.score(readout).percent
        rows.append((index, float(readout), float(percent)))
    return rows


def choose_bearings(rows: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """Best sector, plus up to two more that clearly share its peak."""
    if not rows:
        return []
    ranked = sorted(rows, key=lambda row: row[1], reverse=True)
    best = ranked[0]
    chosen = [best]
    floor = ranked[-1][1]
    span = best[1] - floor
    if span <= 1e-3:
        return chosen
    for row in ranked[1:3]:
        if (row[1] - floor) >= 0.72 * span:
            chosen.append(row)
    return chosen


@dataclass
class MarkLayer:
    alive: dict[int, MapMark] = field(default_factory=dict)

    def consider(
        self,
        trainer: MbTrainer | None,
        feat: np.ndarray | None,
        now: float,
        *,
        recognized: bool,
        mode: str,
        scan: dict | None = None,
    ) -> list[MapMark]:
        if trainer is None or feat is None or not recognized:
            return self.visible(now)
        rows = sector_table(trainer, feat)
        chosen = []
        for index, _readout, percent in choose_bearings(rows):
            dist = scan_range(scan, index) if mode != "sim" else None
            if dist is None:
                dist = feature_range(feat, index)
            mark = project_mark(index, percent, dist, mode=mode, scan=scan)
            mark.born = now
            self.alive[index] = mark
            chosen.append(index)
        # Bearings that lost the peak this frame start fading from now only if
        # they were not refreshed. Their previous born time stays.
        return self.visible(now)

    def visible(self, now: float) -> list[MapMark]:
        dead = []
        out = []
        for index, mark in self.alive.items():
            age = now - mark.born
            if age >= FADE_S:
                dead.append(index)
                continue
            mark.alpha = float(np.clip(1.0 - age / FADE_S, 0.0, 1.0))
            out.append(mark)
        for index in dead:
            del self.alive[index]
        out.sort(key=lambda mark: mark.bin_i)
        return out
