"""Teacher drive mixed with the fly. The mushroom-body split is not touched.

Image left is the dog's left, so a box on the left of the frame yaws left
(positive z). Image right yaws right. A centred box walks forward and stops
at one metre. A lost box searches toward the side where it was last seen.

The sent command is ``weight * teacher + (1 - weight) * fly``. Weight 0 is
the fly, weight 1 is the teacher. E-STOP and a close forward lidar return
override that mix. Nothing here is written into the weight file.

The teacher only steers the dog into the situation. The reward is the
situation, not whether the fly's command matched. PAM when the conspecific
approaches or sits in the centre of the frame. PPL1 when the target is lost
or the approach is inside the stop distance. A command match is a status
metric only.
"""

from __future__ import annotations

from .pilot import STOP_M, X_MAX, Z_MAX, SLOW_X, clamp_velocity, scrub_range, search_yaw
from .stimuli import Stimulus, camera_stimuli, reinforcement_value, signed_reinforcement

TEACHER_WEIGHT_DEFAULT = 0.0
MATCH_WINDOW_S = 8.0
MATCH_TOL_X = 0.05
MATCH_TOL_Z = 0.08
CENTER_BAND = 0.06
WIDE_OFFSET = 0.18
AIM_HOLD_S = 0.50
PAM_GAP_S = 0.50
# Trainer-side approach law. The onboard brain does not call these.
CAMERA_HFOV_DEG = 120.0
AIM_CONE_DEG = 15.0
STOP_BOX_H_DEFAULT = 0.45
STOP_BOX_H_MIN = 0.15
STOP_BOX_H_MAX = 0.90
TEACHER_X_CAP = 0.40
TEACHER_X_CAP_MAX = 0.40
TEACHER_YAW_MAX = 1.0
TEACHER_YAW_GAIN = 4.0
TEACHER_YAW_MIN_OUT = 0.35
TEACHER_YAW_MIN = 0.15
YAW_DEADBAND_DEG = 3.0


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


def gated_outcome(
    gate: PulseGate,
    now: float,
    hit_l: bool,
    hit_r: bool,
    centered: bool,
    approaching: bool,
    lost: bool,
    too_close: bool,
    last_l: bool,
    last_r: bool,
    learning: bool,
    enabled: bool,
    already_l: str | None = None,
    already_r: str | None = None,
) -> tuple[str | None, str | None]:
    """One rate-limited outcome. A side already taught this frame is left alone."""
    if not learning or not enabled:
        return None, None
    kind_l, kind_r = outcome_kinds(
        bool(hit_l),
        bool(hit_r),
        bool(centered),
        bool(approaching),
        bool(lost),
        bool(too_close),
        bool(last_l),
        bool(last_r),
    )
    if kind_l and already_l == kind_l:
        kind_l = None
    if kind_r and already_r == kind_r:
        kind_r = None
    if kind_l and not gate.allow(now, "L", kind_l):
        kind_l = None
    if kind_r and not gate.allow(now, "R", kind_r):
        kind_r = None
    return kind_l, kind_r


def outcome_kinds(
    hit_l: bool,
    hit_r: bool,
    centered: bool,
    approaching: bool,
    lost: bool,
    too_close: bool,
    last_l: bool = False,
    last_r: bool = False,
) -> tuple[str | None, str | None]:
    """PAM for an approach or a centred dog. PPL1 for a loss or a large box.

    The eye that sees the dog gets the pulse. A loss uses the eye that saw
    it last. A box that fills the frame wins over a treat. Lidar range is
    not an input. Command agreement is not an input.
    """
    if lost:
        return ("ppl1" if last_l else None), ("ppl1" if last_r else None)
    if too_close:
        if hit_l or hit_r:
            return ("ppl1" if hit_l else None), ("ppl1" if hit_r else None)
        return ("ppl1" if last_l else None), ("ppl1" if last_r else None)
    if approaching or centered:
        return ("pam" if hit_l else None), ("pam" if hit_r else None)
    return None, None


def sign_for_center(cx: float | None) -> float | None:
    """Image left is the dog's left, so the search sign is positive there."""
    if cx is None:
        return None
    offset = float(cx) - 0.5
    if offset < -1e-6:
        return 1.0
    if offset > 1e-6:
        return -1.0
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
        yaw = clamp_velocity(0.0, -excess / span * Z_MAX)[1]
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
    """One pulse per eye and kind per gap. PAM and PPL1 keep separate clocks."""

    def __init__(self, gap: float = PAM_GAP_S):
        self.gap = float(gap)
        self.last: dict[str, float] = {}

    def allow(self, now: float, side: str, kind: str) -> bool:
        key = "%s:%s" % (side, kind)
        prev = self.last.get(key, -1e9)
        if float(now) - prev < self.gap - 1e-9:
            return False
        self.last[key] = float(now)
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
        self.last_hit_l = False
        self.last_hit_r = False
        self.centered = False
        self.approaching = False
        self.lost = False
        self.too_close = False
        self._prev_dist: float | None = None
        self._prev_area: float | None = None
        self.search_sign = 1.0
        self._last_side_cx: float | None = None
        self._search_locked = False
        self._search_from = 0.0
        self._remote_t = -1e9
        self._remote_x = 0.0
        self._remote_z = 0.0
        self._remote_hit_l = False
        self._remote_hit_r = False
        self._remote_centered = False
        self._remote_approaching = False
        self._remote_lost = False
        self._remote_too_close = False
        self._remote_last_l = False
        self._remote_last_r = False
        self.window = MatchWindow()
        self.gate = PulseGate()

    def update_from_boxes(self, now: float, boxes, dist_m, forward_m, conf_min: float = 0.5) -> tuple[float, float]:
        cx = best_center(boxes, conf_min)
        was_seen = bool(self.seen)
        self.center = cx
        self.seen = cx is not None
        if cx is None:
            self.lost = was_seen
            self.centered = False
            self.approaching = False
            self.too_close = False
        else:
            dist = scrub_range(dist_m)
            if dist is not None:
                self._prev_dist = dist
            self.centered = abs(float(cx) - 0.5) <= CENTER_BAND + 1e-9
            # Closeness and approach are camera stimuli. Lidar still stops
            # the wheels in teacher_velocity and apply_guards, not here.
            close, grew, area = camera_stimuli(boxes, self._prev_area, conf_min)
            self.too_close = close
            self.approaching = grew
            if area is not None:
                self._prev_area = area
            self.lost = False
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

    def set_hits(self, hit_l: bool, hit_r: bool) -> None:
        """Eyes that see the dog now. A visible dog remembers its side for a later loss."""
        self.hit_l = bool(hit_l)
        self.hit_r = bool(hit_r)
        if self.seen:
            self.last_hit_l = self.hit_l
            self.last_hit_r = self.hit_r

    def set_remote(
        self,
        now: float,
        x: float,
        z: float,
        hit_l: bool,
        hit_r: bool,
        weight: float,
        centered: bool = False,
        approaching: bool = False,
        lost: bool = False,
        too_close: bool = False,
        last_l: bool = False,
        last_r: bool = False,
    ) -> None:
        self.weight = clamp_weight(weight)
        self._remote_t = float(now)
        self._remote_x = float(x)
        self._remote_z = float(z)
        self.teacher_x = float(x)
        self.teacher_z = float(z)
        self._remote_hit_l = bool(hit_l)
        self._remote_hit_r = bool(hit_r)
        self._remote_centered = bool(centered)
        self._remote_approaching = bool(approaching)
        self._remote_lost = bool(lost)
        self._remote_too_close = bool(too_close)
        self._remote_last_l = bool(last_l)
        self._remote_last_r = bool(last_r)

    def _apply_remote(self, fresh: bool) -> None:
        if not fresh:
            self.hit_l = False
            self.hit_r = False
            self.centered = False
            self.approaching = False
            self.lost = False
            self.too_close = False
            self.last_hit_l = False
            self.last_hit_r = False
            return
        self.hit_l = self._remote_hit_l
        self.hit_r = self._remote_hit_r
        self.centered = self._remote_centered
        self.approaching = self._remote_approaching
        self.lost = self._remote_lost
        self.too_close = self._remote_too_close
        self.last_hit_l = self._remote_last_l
        self.last_hit_r = self._remote_last_r

    def remote(self, now: float) -> tuple[float, float, bool, bool]:
        """Last posted teacher command. A stale post is a stop with no eye."""
        fresh = float(now) - self._remote_t <= AIM_HOLD_S
        self._apply_remote(fresh)
        if not fresh:
            return 0.0, 0.0, False, False
        return self._remote_x, self._remote_z, self._remote_hit_l, self._remote_hit_r

    def note(self, now: float, matched: bool) -> None:
        self.window.note(float(now), bool(matched))

    def fraction(self, now: float) -> float:
        return self.window.fraction(float(now))

    def action_kinds(
        self,
        now: float,
        learning: bool,
        teacher_on: bool,
        already_l: str | None = None,
        already_r: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Outcome pulses. ``already_*`` is a kind already applied to that eye this frame."""
        return gated_outcome(
            self.gate,
            now,
            self.hit_l,
            self.hit_r,
            self.centered,
            self.approaching,
            self.lost,
            self.too_close,
            self.last_hit_l,
            self.last_hit_r,
            learning,
            teacher_on,
            already_l,
            already_r,
        )

    def aim_fresh(self, now: float) -> bool:
        """A teacher post is still inside the hold window."""
        return float(now) - self._remote_t <= AIM_HOLD_S


class ConspecificStimulus(Stimulus):
    """The one reinforcement source wired today: a conspecific seen by YOLO.

    Approach or a centred box is appetitive. A loss or a box that fills the
    frame is aversive. The mushroom body never names this class.
    """

    def __init__(self) -> None:
        Stimulus.__init__(self, "conspecific")
        self.gate = PulseGate()
        self.enabled = False
        self.hit_l = False
        self.hit_r = False
        self.centered = False
        self.approaching = False
        self.lost = False
        self.too_close = False
        self.last_l = False
        self.last_r = False

    def observe(
        self,
        enabled: bool,
        hit_l: bool,
        hit_r: bool,
        centered: bool,
        approaching: bool,
        lost: bool,
        too_close: bool,
        last_l: bool,
        last_r: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.active = bool(enabled)
        self.hit_l = bool(hit_l)
        self.hit_r = bool(hit_r)
        self.centered = bool(centered)
        self.approaching = bool(approaching)
        self.lost = bool(lost)
        self.too_close = bool(too_close)
        self.last_l = bool(last_l)
        self.last_r = bool(last_r)

    def offer(self, now: float, learning: bool, skip_l: str | None = None, skip_r: str | None = None) -> tuple[float, float]:
        kind_l, kind_r = gated_outcome(
            self.gate,
            now,
            self.hit_l,
            self.hit_r,
            self.centered,
            self.approaching,
            self.lost,
            self.too_close,
            self.last_l,
            self.last_r,
            learning,
            self.enabled,
            skip_l,
            skip_r,
        )
        self.value = reinforcement_value(kind_l, kind_r)
        return signed_reinforcement(kind_l), signed_reinforcement(kind_r)


def attach_recognition_stimuli(board) -> None:
    """Plug in the only source used for recognition. Further sources attach the same way."""
    if board.get("conspecific") is None:
        board.attach(ConspecificStimulus())


def observe_recognition(board, lead: TeacherLead, enabled: bool) -> None:
    """Hand the latest YOLO situation to the conspecific source, if it is attached."""
    stim = board.get("conspecific")
    if stim is None:
        return
    stim.observe(
        enabled,
        lead.hit_l,
        lead.hit_r,
        lead.centered,
        lead.approaching,
        lead.lost,
        lead.too_close,
        lead.last_hit_l,
        lead.last_hit_r,
    )


def clamp_stop_h(value) -> float:
    """Box height (share of the frame) at which the teacher stands."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return STOP_BOX_H_DEFAULT
    if v != v:
        return STOP_BOX_H_DEFAULT
    return min(STOP_BOX_H_MAX, max(STOP_BOX_H_MIN, v))


def box_bearing_deg(cx: float, hfov: float = CAMERA_HFOV_DEG) -> float:
    """Signed bearing of an image-x. Positive is image right."""
    return (float(cx) - 0.5) * float(hfov)


def best_box(boxes, conf_min: float = 0.5):
    """Strongest box at or above ``conf_min``, or None."""
    chosen = None
    best = float(conf_min) - 1e-9
    for box in boxes or ():
        conf = float(getattr(box, "conf", 0.0))
        if conf < best:
            continue
        best = conf
        chosen = box
    return chosen


def teacher_yaw(offset: float) -> float:
    """Strong proportional turn, ±1 rad/s. Outside the ±15° cone never below the minimum.

    Image right (positive offset) yaws right (negative z). Inside a ~3° deadband
    no yaw, so the walk does not wobble.
    """
    off = float(offset)
    bearing = abs(box_bearing_deg(0.5 + off))
    if bearing <= YAW_DEADBAND_DEG:
        return 0.0
    mag = min(TEACHER_YAW_MAX, max(TEACHER_YAW_MIN, TEACHER_YAW_GAIN * abs(off)))
    if bearing > AIM_CONE_DEG + 1e-9:
        mag = max(mag, TEACHER_YAW_MIN_OUT)
    return -mag if off > 0.0 else mag


def approach_velocity(box, forward_m, stop_h: float = STOP_BOX_H_DEFAULT, x_cap: float = TEACHER_X_CAP):
    """``(x, z, state)``: turn first, walk inside the cone, stand when the box is tall.

    state is ``no_box``, ``turn``, ``walk``, ``near`` or ``lidar``. No box is a
    full stop: no search spin, no coasting. Lidar is a safety stop only.
    """
    if box is None:
        return 0.0, 0.0, "no_box"
    x0, x1 = float(box.x0), float(box.x1)
    y0, y1 = float(box.y0), float(box.y1)
    cx = 0.5 * (x0 + x1)
    height = abs(y1 - y0)
    offset = cx - 0.5
    yaw = teacher_yaw(offset)
    if height >= clamp_stop_h(stop_h):
        return 0.0, 0.0, "near"
    if abs(box_bearing_deg(cx)) > AIM_CONE_DEG + 1e-9:
        return 0.0, float(yaw), "turn"
    if lidar_blocks(forward_m):
        return 0.0, float(yaw), "lidar"
    cap = max(0.0, min(float(x_cap), X_MAX))
    return cap, float(yaw), "walk"


TEACHER_STATUS = {
    "no_query": "YOLO не опрашивается",
    "stale": "кадр старый",
    "no_box": "нет рамки",
    "turn": "поворот",
    "near": "рядом, стою",
    "lidar": "препятствие, стою",
    "halt": "стоп",
}


def teacher_status_line(state: str, kin: str = "") -> str:
    if state == "walk":
        return "иду к %s" % (kin or "сородичу")
    return TEACHER_STATUS.get(str(state or ""), "")
