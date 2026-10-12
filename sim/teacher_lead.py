"""Teacher drive mixed with the fly. The mushroom-body split is not touched.

Image left is the dog's right, so a box on the left of the frame yaws right
(negative z). Image right yaws left. A centred box walks forward and stops
at one metre. A lost box searches toward the side where it was last seen.

The sent command is ``weight * teacher + (1 - weight) * fly``. Weight 0 is
the fly, weight 1 is the teacher. E-STOP and a close forward lidar return
override that mix. Nothing here is written into the weight file.
"""

from __future__ import annotations

from .pilot import STOP_M, X_MAX, Z_MAX, SLOW_X, clamp_velocity, scrub_range, search_yaw

TEACHER_WEIGHT_DEFAULT = 0.0
MATCH_WINDOW_S = 8.0
MATCH_TOL_X = 0.05
MATCH_TOL_Z = 0.08
CENTER_BAND = 0.06
WIDE_OFFSET = 0.18
AIM_HOLD_S = 0.50
PAM_GAP_S = 0.50


def clamp_weight(weight: float) -> float:
    value = float(weight)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def lead_mode(weight: float) -> str:
    """``teacher``, ``mix``, or ``fly``."""
    value = clamp_weight(weight)
    if value >= 0.995:
        return "teacher"
    if value <= 0.005:
        return "fly"
    return "mix"


def lead_label(weight: float) -> str:
    mode = lead_mode(weight)
    if mode == "teacher":
        return "учитель"
    if mode == "fly":
        return "муха"
    return "смешанно"


def lead_plaque(weight: float) -> str:
    """``ведёт: учитель``, ``ведёт: смешанно NN/MM``, or ``ведёт: муха``."""
    mode = lead_mode(weight)
    if mode == "teacher":
        return "ведёт: учитель"
    if mode == "fly":
        return "ведёт: муха"
    teacher = int(clamp_weight(weight) * 100.0 + 0.5)
    if teacher < 1:
        teacher = 1
    elif teacher > 99:
        teacher = 99
    return "ведёт: смешанно %d/%d" % (teacher, 100 - teacher)


def mix_velocity(
    fly_x: float,
    fly_z: float,
    teacher_x: float,
    teacher_z: float,
    weight: float,
) -> tuple[float, float]:
    """``weight`` of the teacher plus the rest of the fly. Both axes clamped."""
    share = clamp_weight(weight)
    return clamp_velocity(
        share * float(teacher_x) + (1.0 - share) * float(fly_x),
        share * float(teacher_z) + (1.0 - share) * float(fly_z),
    )


def lidar_blocks(forward_m: float | None) -> bool:
    """Forward return inside the stop distance, past the body radius."""
    dist = scrub_range(forward_m)
    return dist is not None and dist < STOP_M


def apply_guards(
    x: float,
    z: float,
    estop: bool,
    forward_m: float | None,
) -> tuple[float, float, bool]:
    """E-STOP stops both axes. A close lidar return stops forward speed.

    Yaw is kept on a lidar stop so the dog can still turn. The returned
    flag is true when nothing is left to send.
    """
    if estop:
        return 0.0, 0.0, True
    speed, yaw = clamp_velocity(x, z)
    if lidar_blocks(forward_m):
        speed = 0.0
    stopped = abs(speed) < 1e-6 and abs(yaw) < 1e-6
    return speed, yaw, stopped


def commands_match(
    fly_x: float,
    fly_z: float,
    teacher_x: float,
    teacher_z: float,
    tol_x: float = MATCH_TOL_X,
    tol_z: float = MATCH_TOL_Z,
) -> bool:
    return abs(float(fly_x) - float(teacher_x)) <= float(tol_x) and abs(float(fly_z) - float(teacher_z)) <= float(tol_z)


def match_reinforce(matched: bool, hit_l: bool, hit_r: bool) -> tuple[str | None, str | None]:
    """PAM on an eye that sees the dog, and only when the commands agree."""
    if not matched:
        return None, None
    return ("pam" if hit_l else None), ("pam" if hit_r else None)


def sign_for_center(cx: float | None) -> float | None:
    """Image left is the dog's right, so the search sign is negative there."""
    if cx is None:
        return None
    offset = float(cx) - 0.5
    if offset < -1e-6:
        return -1.0
    if offset > 1e-6:
        return 1.0
    return None


def teacher_velocity(
    cx: float | None,
    dist_m: float | None,
    forward_m: float | None,
    search_age: float,
    search_sign: float,
) -> tuple[float, float]:
    """Turn toward the box, walk once it is centred, stop at ``STOP_M``.

    ``cx`` is the box centre as a fraction of the image. None means the
    target is lost and the dog searches on ``search_sign``.
    """
    if cx is None:
        return 0.0, search_yaw(float(search_age), float(search_sign))
    offset = float(cx) - 0.5
    if abs(offset) <= CENTER_BAND:
        yaw = 0.0
        centred = True
    else:
        span = 0.5 - CENTER_BAND
        excess = offset - (CENTER_BAND if offset > 0.0 else -CENTER_BAND)
        yaw = clamp_velocity(0.0, excess / span * Z_MAX)[1]
        centred = False
    dist = scrub_range(dist_m)
    ahead = scrub_range(forward_m)
    close = dist is not None and dist <= STOP_M
    blocked = ahead is not None and ahead < STOP_M
    if close or blocked:
        return clamp_velocity(0.0, yaw)
    wide = abs(offset) > WIDE_OFFSET
    if centred:
        speed = X_MAX if dist is not None else SLOW_X
    elif wide:
        speed = 0.0
    else:
        speed = SLOW_X
    return clamp_velocity(speed, yaw)


def best_center(boxes, conf_min: float = 0.5) -> float | None:
    """Image-x of the strongest box. Coordinates are fractions of the frame."""
    chosen = None
    best = float(conf_min) - 1e-9
    for box in boxes or ():
        conf = float(getattr(box, "conf", 0.0))
        if conf < float(conf_min) or conf < best:
            continue
        best = conf
        chosen = 0.5 * (float(box.x0) + float(box.x1))
    return chosen


class MatchWindow:
    """Share of matching samples inside the last ``horizon`` seconds."""

    def __init__(self, horizon: float = MATCH_WINDOW_S):
        self.horizon = float(horizon)
        self._samples: list[tuple[float, bool]] = []

    def note(self, now: float, matched: bool) -> None:
        now = float(now)
        self._samples.append((now, bool(matched)))
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = float(now) - self.horizon
        if self._samples and self._samples[0][0] < cutoff:
            self._samples = [item for item in self._samples if item[0] >= cutoff]

    def fraction(self, now: float) -> float:
        self._trim(float(now))
        if not self._samples:
            return 0.0
        hits = 0
        for _stamp, matched in self._samples:
            if matched:
                hits += 1
        return float(hits) / float(len(self._samples))


class PulseGate:
    """One PAM per eye per gap. Keeps a matching stream from teaching every frame."""

    def __init__(self, gap: float = PAM_GAP_S):
        self.gap = float(gap)
        self.last: dict[str, float] = {}

    def allow(self, now: float, side: str) -> bool:
        prev = self.last.get(side, -1e9)
        if float(now) - prev < self.gap - 1e-9:
            return False
        self.last[side] = float(now)
        return True


class TeacherLead:
    """Last box, the search that follows it, and the recent match rate."""

    def __init__(self) -> None:
        self.weight = TEACHER_WEIGHT_DEFAULT
        self.teacher_x = 0.0
        self.teacher_z = 0.0
        self.fly_x = 0.0
        self.fly_z = 0.0
        self.center: float | None = None
        self.seen = False
        self.hit_l = False
        self.hit_r = False
        self.search_sign = 1.0
        self._last_side_cx: float | None = None
        self._search_locked = False
        self._search_from = 0.0
        self._remote_t = -1e9
        self._remote_x = 0.0
        self._remote_z = 0.0
        self._remote_hit_l = False
        self._remote_hit_r = False
        self.window = MatchWindow()
        self.gate = PulseGate()

    def update_from_boxes(self, now: float, boxes, dist_m, forward_m, conf_min: float = 0.5) -> tuple[float, float]:
        cx = best_center(boxes, conf_min)
        self.center = cx
        self.seen = cx is not None
        if cx is None:
            if not self._search_locked:
                sign = sign_for_center(self._last_side_cx)
                self.search_sign = 1.0 if sign is None else float(sign)
                self._search_locked = True
                self._search_from = float(now)
            age = float(now) - self._search_from
            speed, yaw = teacher_velocity(None, dist_m, forward_m, age, self.search_sign)
        else:
            self._search_locked = False
            sign = sign_for_center(cx)
            if sign is not None:
                self._last_side_cx = float(cx)
                self.search_sign = float(sign)
            speed, yaw = teacher_velocity(cx, dist_m, forward_m, 0.0, self.search_sign)
        self.teacher_x = float(speed)
        self.teacher_z = float(yaw)
        return self.teacher_x, self.teacher_z

    def set_remote(self, now: float, x: float, z: float, hit_l: bool, hit_r: bool, weight: float) -> None:
        self.weight = clamp_weight(weight)
        self._remote_t = float(now)
        self._remote_x = float(x)
        self._remote_z = float(z)
        self.teacher_x = float(x)
        self.teacher_z = float(z)
        self._remote_hit_l = bool(hit_l)
        self._remote_hit_r = bool(hit_r)

    def remote(self, now: float) -> tuple[float, float, bool, bool]:
        """Last posted teacher command. A stale post is a stop with no eye."""
        if float(now) - self._remote_t > AIM_HOLD_S:
            return 0.0, 0.0, False, False
        return self._remote_x, self._remote_z, self._remote_hit_l, self._remote_hit_r

    def note(self, now: float, matched: bool) -> None:
        self.window.note(float(now), bool(matched))

    def fraction(self, now: float) -> float:
        return self.window.fraction(float(now))

    def action_kinds(
        self,
        now: float,
        matched: bool,
        learning: bool,
        teacher_on: bool,
        already_l: bool = False,
        already_r: bool = False,
    ) -> tuple[str | None, str | None]:
        if not learning or not teacher_on:
            return None, None
        kind_l, kind_r = match_reinforce(
            bool(matched),
            bool(self.hit_l) and not already_l,
            bool(self.hit_r) and not already_r,
        )
        if kind_l and not self.gate.allow(now, "L"):
            kind_l = None
        if kind_r and not self.gate.allow(now, "R"):
            kind_r = None
        return kind_l, kind_r
