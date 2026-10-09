"""Who may send Move: the mushroom body, the operator, or nobody.

Autonomy searches by turning in place until the fly MB says «УЗНАЮ», then
walks toward that camera sector and stops near one metre. Arrows do not
steal autonomy. Only the takeover key does, and only the autonomy button
gives it back. Every velocity is clamped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .map_marks import bin_angle, feature_range
from .raw_sense import FOV, N_AZ

X_MAX = 0.4
Z_MAX = 1.0
STOP_M = 1.0
SAFE_M = 1.2
MANUAL_HOLD_S = 0.30
LINK_HOLD_S = 1.0
TEACH_PERIOD_S = 0.45
SEARCH_TURN = 0.35
SEARCH_TURN_S = 1.10
SEARCH_PAUSE_S = 0.45


def clamp_velocity(x: float, z: float) -> tuple[float, float]:
    x = float(x)
    z = float(z)
    if x > X_MAX:
        x = X_MAX
    elif x < -X_MAX:
        x = -X_MAX
    if z > Z_MAX:
        z = Z_MAX
    elif z < -Z_MAX:
        z = -Z_MAX
    return x, z


def forward_clearance(feat: np.ndarray) -> float | None:
    """Nearest return in the two sectors that face forward. None if neither has one."""
    found = []
    for index in (3, 4):
        dist = feature_range(feat, index, width=1)
        if dist is not None:
            found.append(dist)
    if not found:
        return None
    return float(min(found))


def ego_sector_ranges(points_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> list[float | None]:
    """Nearest return in each camera sector. Same bins as the lidar marks.

    ``points_xy`` and ``origin_xy`` are world metres, x right and y up, yaw
    from the robot odometry. Empty sectors stay None.
    """
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    origin = np.asarray(origin_xy, dtype=np.float32).reshape(2)
    ranges: list[float | None] = [None] * N_AZ
    if len(points) == 0:
        return ranges
    delta = points - origin
    dist = np.linalg.norm(delta, axis=1)
    ang = np.arctan2(delta[:, 1], delta[:, 0]) - float(yaw)
    ang = (ang + np.pi) % (2.0 * np.pi) - np.pi
    half = float(FOV) / (2.0 * N_AZ)
    for index in range(N_AZ):
        center = bin_angle(index)
        delta_ang = (ang - center + np.pi) % (2.0 * np.pi) - np.pi
        picked = (np.abs(delta_ang) <= half) & (dist > 0.35) & (dist < 12.0)
        if np.any(picked):
            ranges[index] = float(dist[picked].min())
    return ranges


def search_yaw(age: float) -> float:
    """Slow sweep, then a pause so the camera frame can settle."""
    half = SEARCH_TURN_S + SEARCH_PAUSE_S
    u = float(age) % (2.0 * half)
    if u < SEARCH_TURN_S:
        return SEARCH_TURN
    if u < half:
        return 0.0
    if u < half + SEARCH_TURN_S:
        return -SEARCH_TURN
    return 0.0


def seek_velocity(
    recognized: bool,
    sector: int | None,
    dist_m: float | None,
    forward_m: float | None,
    search_age: float,
) -> tuple[float, float, str]:
    """Return clamped (x, z, phase). x is m/s forward, z is rad/s, left positive."""
    if not recognized or sector is None:
        return 0.0, search_yaw(search_age), "search"
    ang = bin_angle(int(sector))
    turn = ang / 0.55
    if turn > 1.0:
        turn = 1.0
    elif turn < -1.0:
        turn = -1.0
    close = dist_m is not None and dist_m <= STOP_M
    blocked = dist_m is None and (forward_m is None or forward_m < SAFE_M)
    if close or blocked:
        # Stop the walk. A small yaw still recenters the dog. It does not close range.
        yaw = turn * 0.35 if abs(ang) > 0.22 else 0.0
        return clamp_velocity(0.0, yaw) + ("hold",)
    if dist_m is None:
        forward = 0.15
    elif dist_m > 1.6:
        forward = 0.35
    else:
        forward = 0.22
    if abs(ang) > 0.45:
        forward = 0.08 if forward > 0.08 else forward
    x, z = clamp_velocity(forward, turn)
    return x, z, "approach"


@dataclass
class DriveCommand:
    x: float = 0.0
    z: float = 0.0
    stop: bool = True
    phase: str = "stop"
    who: str = "никто"
    hint: str = ""


class Pilot:
    """Autonomy, manual takeover, and stop. Default is manual, autonomy off."""

    def __init__(self, return_auto_s: float = 0.0):
        self.mode = "manual"  # manual | auto | estop
        self.took_over = False
        self.held_stop = False
        self.return_auto_s = float(return_auto_s)
        self.phase = "stop"
        self.who = "оператор"
        self.hint = ""
        self._search_from = 0.0
        self._idle_from: float | None = None

    @property
    def autonomy(self) -> bool:
        return self.mode == "auto"

    def estop(self) -> None:
        self.mode = "estop"
        self.took_over = False
        self.held_stop = False
        self.phase = "stop"
        self.who = "никто"
        self.hint = ""
        self._idle_from = None

    def space(self) -> None:
        """Stop now. Autonomy does not resume by itself."""
        if self.mode == "estop":
            self.phase = "stop"
            self.who = "никто"
            return
        self.mode = "manual"
        self.took_over = False
        self.held_stop = True
        self.phase = "stop"
        self.who = "никто"
        self.hint = ""
        self._idle_from = None

    def takeover(self) -> None:
        """Explicit manual grab. Also the way out of E-STOP into manual."""
        self.mode = "manual"
        self.took_over = True
        self.held_stop = False
        self.phase = "stop"
        self.who = "оператор"
        self.hint = ""
        self._idle_from = None

    def start_auto(self, now: float) -> bool:
        if self.mode == "estop":
            return False
        self.mode = "auto"
        self.took_over = False
        self.held_stop = False
        self._search_from = float(now)
        self._idle_from = None
        self.who = "мозг"
        self.hint = ""
        return True

    def stop_auto(self) -> None:
        if self.mode == "estop":
            return
        self.mode = "manual"
        self.took_over = False
        self.held_stop = False
        self.phase = "stop"
        self.who = "оператор"
        self.hint = ""
        self._idle_from = None

    def command(
        self,
        now: float,
        arrows: tuple[float, float],
        *,
        focused: bool,
        frames_ok: bool,
        link_ok: bool,
        recognized: bool,
        sector: int | None,
        dist_m: float | None,
        forward_m: float | None,
        manual_axes: tuple[float, float] | None = None,
    ) -> DriveCommand:
        ax = float(arrows[0]) if arrows else 0.0
        az = float(arrows[1]) if arrows else 0.0
        self.hint = ""
        if self.mode == "estop" or not link_ok:
            self.phase = "stop"
            self.who = "никто"
            return DriveCommand(0.0, 0.0, True, "stop", self.who, "")
        if not focused:
            self.phase = "stop"
            self.who = "никто"
            return DriveCommand(0.0, 0.0, True, "stop", self.who, "")
        if self.mode == "auto":
            if abs(ax) + abs(az) > 0 or manual_axes is not None and (abs(manual_axes[0]) + abs(manual_axes[1]) > 1e-6):
                self.hint = "нажми ПЕРЕХВАТ"
            if not frames_ok:
                self.phase = "stop"
                self.who = "мозг"
                return DriveCommand(0.0, 0.0, True, "stop", self.who, self.hint)
            x, z, phase = seek_velocity(
                recognized,
                sector if sector is None else int(sector) % N_AZ,
                dist_m,
                forward_m,
                float(now) - self._search_from,
            )
            self.phase = phase
            self.who = "мозг"
            stop = abs(x) < 1e-6 and abs(z) < 1e-6
            return DriveCommand(x, z, stop, phase, self.who, self.hint)
        # Manual. A quiet stick stops the dog and does not restart autonomy.
        if manual_axes is not None:
            x, z = clamp_velocity(manual_axes[0], manual_axes[1])
        else:
            x, z = clamp_velocity(0.4 * ax, 1.0 * az if az else 0.0)
            if az > 0:
                z = Z_MAX
            elif az < 0:
                z = -Z_MAX
            x, z = clamp_velocity(x, z)
        if abs(x) < 1e-6 and abs(z) < 1e-6:
            if self.return_auto_s > 0 and self.took_over and not self.held_stop:
                if self._idle_from is None:
                    self._idle_from = float(now)
                elif float(now) - self._idle_from >= self.return_auto_s:
                    self.start_auto(now)
                    return self.command(
                        now,
                        (0.0, 0.0),
                        focused=focused,
                        frames_ok=frames_ok,
                        link_ok=link_ok,
                        recognized=recognized,
                        sector=sector,
                        dist_m=dist_m,
                        forward_m=forward_m,
                        manual_axes=None,
                    )
            self.phase = "stop"
            self.who = "никто" if self.held_stop else "оператор"
            return DriveCommand(0.0, 0.0, True, "stop", self.who, "")
        self._idle_from = None
        self.held_stop = False
        self.phase = "manual"
        self.who = "оператор"
        return DriveCommand(x, z, False, "manual", self.who, "")

    def label(self) -> str:
        if self.mode == "estop" or self.held_stop:
            return "СТОП"
        if self.mode == "auto":
            return "АВТОНОМИЯ"
        if self.took_over:
            return "РУЧНОЕ · ПЕРЕХВАТ"
        return "РУЧНОЕ"


class TeachRepeater:
    """Holding T or X pulses the DAN at a limited rate. Learning off is the caller's choice."""

    def __init__(self, period: float = TEACH_PERIOD_S):
        self.period = float(period)
        self._last = -1e9
        self._kind: str | None = None

    def poll(self, now: float, treat_edge: bool, punish_edge: bool, treat_down: bool, punish_down: bool) -> str | None:
        kind = None
        if treat_edge:
            kind = "pam"
        elif punish_edge:
            kind = "ppl1"
        elif treat_down:
            kind = "pam"
        elif punish_down:
            kind = "ppl1"
        if kind is None:
            self._kind = None
            return None
        if kind != self._kind or treat_edge or punish_edge or float(now) - self._last >= self.period:
            self._kind = kind
            self._last = float(now)
            return kind
        return None
