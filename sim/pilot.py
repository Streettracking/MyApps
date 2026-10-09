"""Who may send Move: the mushroom body, the operator, or nobody.

Autonomy searches by turning left in place until the fly MB says «УЗНАЮ»
on enough frames, then walks toward the median of those sectors and stops
near one metre. The raw sector is not smoothed inside the mushroom body.
Arrows do not steal autonomy. Only the takeover key does, and only the
autonomy button gives it back. Every velocity is clamped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hemifield import bilateral_yaw
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
# Points inside this radius are the robot's own body, legs, or mount.
SELF_RADIUS_M = 0.6
BODY_HALF_M = 0.22
BODY_REAR_M = 0.30
BODY_NOSE_M = 0.50
# Approach only after this many recognized frames in the window, then keep
# the bearing for COAST_S after the word drops. The mushroom body is not smoothed.
HYST_WINDOW = 8
HYST_NEED = 5
SECTOR_MEMORY = 5
COAST_S = 1.0
SLOW_X = 0.20
SIDE_EMA = 0.45
STEER_MODES = ("bilateral", "sectors")
# Second eye must agree within this long, or the first eye was a false alarm.
EYE_CONFIRM_S = 3.0
# Drop out of the walk after this many frames with one eye dark.
EYE_LOSE_N = 5


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


def scrub_range(dist: float | None, self_radius: float = SELF_RADIUS_M) -> float | None:
    """Drop a range that is still inside the body. None means no reliable return."""
    if dist is None:
        return None
    value = float(dist)
    if value < float(self_radius):
        return None
    return value


def forward_clearance(feat: np.ndarray, self_radius: float = SELF_RADIUS_M) -> float | None:
    """Forward return in the two sectors that face ahead, past the body.

    The feature channel stores one near-value per sector, so a body hit
    saturates it. Those ranges are discarded rather than treated as the target.
    """
    found = []
    for index in (3, 4):
        dist = scrub_range(feature_range(feat, index, width=1), self_radius)
        if dist is not None:
            found.append(dist)
    if not found:
        return None
    return float(min(found))


def ego_sector_ranges(
    points_xy: np.ndarray,
    origin_xy: np.ndarray,
    yaw: float,
    self_radius: float = SELF_RADIUS_M,
) -> list[float | None]:
    """10th-percentile return in each camera sector, past the robot's body.

    ``points_xy`` and ``origin_xy`` are world metres, x right and y up, yaw
    from the robot odometry. Empty sectors stay None. The nearest point is
    not used: a single return on the leg would look like a target at 0.35 m.
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
    fwd = dist * np.cos(ang)
    left = dist * np.sin(ang)
    body = (np.abs(left) <= BODY_HALF_M) & (fwd >= -BODY_REAR_M) & (fwd <= BODY_NOSE_M)
    keep = (~body) & (dist >= float(self_radius)) & (dist < 12.0) & np.isfinite(dist)
    half = float(FOV) / (2.0 * N_AZ)
    for index in range(N_AZ):
        center = bin_angle(index)
        delta_ang = (ang - center + np.pi) % (2.0 * np.pi) - np.pi
        picked = (np.abs(delta_ang) <= half) & keep
        if not np.any(picked):
            continue
        ranges[index] = float(np.percentile(dist[picked], 10))
    return ranges


def cloud_forward(ranges) -> float | None:
    """Nearer forward sector. Each value is already a percentile past the body."""
    if not ranges:
        return None
    found = []
    for index in (3, 4):
        if index < len(ranges) and ranges[index] is not None:
            found.append(float(ranges[index]))
    if not found:
        return None
    return float(min(found))


def search_yaw(age: float) -> float:
    """Slow turn to the left, then a pause so the camera frame can settle.

    The sign does not flip. The old left-right sweep was this function, not
    the mushroom body.
    """
    period = SEARCH_TURN_S + SEARCH_PAUSE_S
    u = float(age) % period
    if u < SEARCH_TURN_S:
        return SEARCH_TURN
    return 0.0


def phase_label(phase: str, steer: str = "bilateral") -> str:
    """Short Russian name for a pilot phase. Sector mode keeps the old approach word."""
    if phase == "approach" and steer == "sectors":
        return "подход"
    names = {
        "search": "поиск",
        "align_l": "доворот (Л)",
        "align_r": "доворот (П)",
        "approach": "подтверждено — иду",
        "hold": "стоп 1 м",
        "manual": "ручное",
        "stop": "стоп",
    }
    return names.get(phase, phase)


class EyeConfirm:
    """Walk only when both hemispheres recognize. One eye turns in place.

    Each eye's ``recognized`` bit already carries its own calibrator hysteresis.
    A lone eye yaws toward that side at the search rate and never sends forward
    speed. If the other eye is still dark after ``EYE_CONFIRM_S``, the sighting
    is a false alarm: search resumes and that eye cannot start another turn
    until it goes dark. Losing one eye for ``EYE_LOSE_N`` frames during the
    walk drops back to the same in-place turn.
    """

    def __init__(self, timeout_s: float = EYE_CONFIRM_S, lose_n: int = EYE_LOSE_N):
        self.timeout_s = float(timeout_s)
        self.lose_n = int(lose_n)
        self.reset()

    def reset(self) -> None:
        self.phase = "search"
        self._align_from: float | None = None
        self._miss_l = 0
        self._miss_r = 0
        self._suppress: str | None = None
        self.recognized_l = False
        self.recognized_r = False

    def update(self, now: float, recognized_l: bool, recognized_r: bool) -> str:
        now = float(now)
        left = bool(recognized_l)
        right = bool(recognized_r)
        self.recognized_l = left
        self.recognized_r = right
        if self._suppress == "L" and not left:
            self._suppress = None
        elif self._suppress == "R" and not right:
            self._suppress = None
        if left and right:
            self._suppress = None
            self._align_from = None
            self._miss_l = 0
            self._miss_r = 0
            self.phase = "approach"
            return "approach"
        if self.phase == "approach":
            self._miss_l = 0 if left else self._miss_l + 1
            self._miss_r = 0 if right else self._miss_r + 1
            if self._miss_l < self.lose_n and self._miss_r < self.lose_n:
                return "approach"
            self._miss_l = 0
            self._miss_r = 0
            self.phase = "search"
        see_l = left and self._suppress != "L"
        see_r = right and self._suppress != "R"
        if see_l and not see_r:
            return self._align(now, "align_l")
        if see_r and not see_l:
            return self._align(now, "align_r")
        self._align_from = None
        self.phase = "search"
        return "search"

    def _align(self, now: float, phase: str) -> str:
        if self.phase != phase or self._align_from is None:
            self._align_from = float(now)
        self.phase = phase
        if float(now) - float(self._align_from) >= self.timeout_s:
            self._suppress = "L" if phase == "align_l" else "R"
            self._align_from = None
            self.phase = "search"
            return "search"
        return phase

    def velocity(
        self,
        now: float,
        recognized_l: bool,
        recognized_r: bool,
        dist_m: float | None,
        forward_m: float | None,
        search_age: float,
        fly_z: float,
    ) -> tuple[float, float, str]:
        phase = self.update(now, recognized_l, recognized_r)
        if phase == "align_l":
            return 0.0, SEARCH_TURN, "align_l"
        if phase == "align_r":
            return 0.0, -SEARCH_TURN, "align_r"
        if phase == "search":
            return 0.0, search_yaw(search_age), "search"
        x, z, stepped = seek_velocity(
            True,
            None,
            dist_m,
            forward_m,
            search_age,
            steer="bilateral",
            fly_z=float(fly_z),
        )
        return x, z, stepped


def half_sector() -> float:
    """Half a camera sector. Inside this, the target counts as centred."""
    return float(FOV) / (2.0 * float(N_AZ))


def yaw_for_bearing(ang: float) -> float:
    """Yaw rate from a bearing. Left is positive. A centred target is z = 0.

    Sector 0 is the right edge of the 110° view (negative z). Sector 7 is
    the left edge (positive z). Sectors 3 and 4 straddle straight ahead.
    """
    half = half_sector()
    ang = float(ang)
    if abs(ang) <= half + 1e-5:
        return 0.0
    excess = ang - float(np.copysign(half, ang))
    return clamp_velocity(0.0, excess / 0.85)[1]


def seek_velocity(
    recognized: bool,
    sector: int | None,
    dist_m: float | None,
    forward_m: float | None,
    search_age: float,
    steer: str = "sectors",
    fly_z: float = 0.0,
) -> tuple[float, float, str]:
    """Return clamped (x, z, phase). x is m/s forward, z is rad/s, left positive.

    ``recognized`` and ``sector`` here are the controller's latched values,
    not the raw mushroom-body frame. Ranges inside the body radius are dropped.
    ``steer='bilateral'`` takes yaw from ``fly_z`` (already deadbanded). Search
    with the word off stays a one-way left turn in either mode.
    """
    dist_m = scrub_range(dist_m)
    forward_m = scrub_range(forward_m)
    bilateral = steer == "bilateral"
    if not recognized or (not bilateral and sector is None):
        return 0.0, search_yaw(search_age), "search"
    if bilateral:
        yaw = float(fly_z)
        centred = abs(yaw) <= 1e-6
        wide = abs(yaw) > 0.45
    else:
        ang = bin_angle(int(sector))
        yaw = yaw_for_bearing(ang)
        centred = abs(ang) <= half_sector() + 1e-5
        wide = abs(ang) > 0.45
    close = dist_m is not None and dist_m <= STOP_M
    blocked = forward_m is not None and forward_m < STOP_M
    if close or blocked:
        return clamp_velocity(0.0, yaw) + ("hold",)
    if dist_m is None:
        forward = SLOW_X
    elif centred or not wide:
        forward = X_MAX
    else:
        forward = SLOW_X
    x, z = clamp_velocity(forward, yaw)
    return x, z, "approach"


def format_range_line(dist, forward, sector, sector_smooth, hysteresis) -> str:
    """One monitor line: filtered ranges, raw sector, smoothed sector."""

    def metres(value) -> str:
        if value is None:
            return "—"
        return "%.2f м" % float(value)

    raw = "—" if sector is None else str(int(sector))
    smooth = "—" if sector_smooth is None else str(int(sector_smooth))
    gate = ""
    if isinstance(hysteresis, dict) and hysteresis.get("window"):
        gate = "  %s/%s" % (int(hysteresis.get("votes") or 0), int(hysteresis["window"]))
        if hysteresis.get("coast"):
            gate += " держу"
        elif hysteresis.get("latched"):
            gate += " подход"
    return "дальн %s   вперёд %s   сектор %s→%s%s" % (metres(dist), metres(forward), raw, smooth, gate)


class ApproachTrack:
    """Smooth the brain's sector and the recognize bit. Does not teach.

    ``sector`` on the status line stays the raw strongest window. ``sector_smooth``
    is the median of the last few raw sectors. Approach starts after
    ``HYST_NEED`` of the last ``HYST_WINDOW`` frames, and the last bearing is
    kept for ``COAST_S`` after the word drops.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.history: list[bool] = []
        self.sectors: list[int] = []
        self.votes = 0
        self.latched = False
        self.coast = False
        self._smooth: int | None = None
        self._coast_until = -1e9
        self.raw_recognized = False
        self.raw_sector: int | None = None
        self.r_l = 0.0
        self.r_r = 0.0
        self.r_diff = 0.0
        self.yaw_z = 0.0
        self._ema_l: float | None = None
        self._ema_r: float | None = None

    def note_sides(self, r_l: float, r_r: float) -> None:
        """EMA of the two readouts. The mushroom bodies themselves are not smoothed."""
        left = float(r_l)
        right = float(r_r)
        if self._ema_l is None:
            self._ema_l = left
            self._ema_r = right
        else:
            self._ema_l = SIDE_EMA * left + (1.0 - SIDE_EMA) * self._ema_l
            self._ema_r = SIDE_EMA * right + (1.0 - SIDE_EMA) * float(self._ema_r)
        self.r_l = float(self._ema_l)
        self.r_r = float(self._ema_r)
        self.r_diff = self.r_l - self.r_r
        self.yaw_z = bilateral_yaw(self.r_l, self.r_r)

    def update(self, now: float, recognized: bool, sector: int | None) -> None:
        now = float(now)
        seen = bool(recognized) and sector is not None
        self.raw_recognized = seen
        self.raw_sector = int(sector) % N_AZ if seen else None
        self.history.append(seen)
        if len(self.history) > HYST_WINDOW:
            self.history.pop(0)
        self.votes = int(sum(1 for item in self.history if item))
        if seen:
            assert sector is not None
            self.sectors.append(int(sector) % N_AZ)
            if len(self.sectors) > SECTOR_MEMORY:
                self.sectors.pop(0)
            self._smooth = int(np.round(float(np.median(np.asarray(self.sectors, dtype=np.float64)))))
            if self.votes >= HYST_NEED:
                self.latched = True
            if self.latched:
                self._coast_until = now + COAST_S
        if self.latched and not seen:
            if now <= self._coast_until:
                self.coast = True
            else:
                self.latched = False
                self.coast = False
                self.sectors.clear()
                self._smooth = None
        else:
            self.coast = False

    @property
    def sector_smooth(self) -> int | None:
        return self._smooth

    def as_dict(self) -> dict:
        return {
            "votes": int(self.votes),
            "need": HYST_NEED,
            "window": HYST_WINDOW,
            "latched": bool(self.latched),
            "coast": bool(self.coast),
        }


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

    def __init__(self, return_auto_s: float = 0.0, self_radius: float = SELF_RADIUS_M):
        self.mode = "manual"  # manual | auto | estop
        self.took_over = False
        self.held_stop = False
        self.return_auto_s = float(return_auto_s)
        self.self_radius = float(self_radius)
        self.phase = "stop"
        self.who = "оператор"
        self.hint = ""
        self.track = ApproachTrack()
        self.eyes = EyeConfirm()
        self.steer = "bilateral"
        self._search_from = 0.0
        self._idle_from: float | None = None

    def set_steer(self, mode: str) -> str:
        if mode not in STEER_MODES:
            raise ValueError(mode)
        if mode != self.steer:
            self.eyes.reset()
        self.steer = mode
        return mode

    def toggle_steer(self) -> str:
        self.steer = "sectors" if self.steer == "bilateral" else "bilateral"
        return self.steer

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
        self.track.reset()
        self.eyes.reset()
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
        r_l: float = 0.0,
        r_r: float = 0.0,
        recognized_l: bool | None = None,
        recognized_r: bool | None = None,
    ) -> DriveCommand:
        ax = float(arrows[0]) if arrows else 0.0
        az = float(arrows[1]) if arrows else 0.0
        self.hint = ""
        self.track.note_sides(r_l, r_r)
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
            dist_m = scrub_range(dist_m, self.self_radius)
            forward_m = scrub_range(forward_m, self.self_radius)
            raw_sector = None if sector is None else int(sector) % N_AZ
            self.track.update(float(now), bool(recognized), raw_sector)
            if recognized_l is None:
                recognized_l = bool(recognized)
            if recognized_r is None:
                recognized_r = bool(recognized)
            if self.steer == "bilateral":
                x, z, phase = self.eyes.velocity(
                    float(now),
                    bool(recognized_l),
                    bool(recognized_r),
                    dist_m,
                    forward_m,
                    float(now) - self._search_from,
                    self.track.yaw_z,
                )
            else:
                use_sector = self.track.sector_smooth if self.track.latched else None
                x, z, phase = seek_velocity(
                    self.track.latched,
                    use_sector,
                    dist_m,
                    forward_m,
                    float(now) - self._search_from,
                    steer=self.steer,
                    fly_z=self.track.yaw_z,
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
                        r_l=r_l,
                        r_r=r_r,
                        recognized_l=recognized_l,
                        recognized_r=recognized_r,
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
