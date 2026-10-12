"""Pluggable reinforcement, plus the camera measures a stimulus may read.

The mushroom body does not name a stimulus. It sums signed values from
whatever is attached: positive is appetitive, negative is aversive.
Lidar range is not a source. A close return still stops the dog in the
pilot and in ``apply_guards``; that reflex never enters this registry.
"""

from __future__ import annotations

# A confident box this tall, or covering this much of the frame, is too close.
CLOSE_HEIGHT = 0.70
CLOSE_AREA = 0.18
# Sim distance at which a conspecific fills the frame the way CLOSE_HEIGHT does.
# Nearer than this, the sim auto-teacher sends PPL1 instead of PAM.
TOO_CLOSE_M = 0.55
# Growth of the tracked box, as a fraction of the frame, counts as an approach.
APPROACH_GROWTH = 0.02

# Learning reads the YOLO box only. Lidar is display and a motor reflex.
LEARNING_SOURCES = ("yolo_bbox",)


def learning_sources() -> tuple[str, ...]:
    return LEARNING_SOURCES


def signed_reinforcement(kind: str | None) -> float:
    """+1 appetitive, -1 aversive, 0 when this side is silent."""
    if kind == "pam":
        return 1.0
    if kind == "ppl1":
        return -1.0
    return 0.0


def reinforcement_kind(signed: float) -> str | None:
    """The dopamine a summed value asks for. Zero stays silent."""
    if float(signed) > 1e-9:
        return "pam"
    if float(signed) < -1e-9:
        return "ppl1"
    return None


def reinforcement_value(kind_l: str | None, kind_r: str | None) -> float:
    """One status value for a stimulus. Agreeing eyes stay at +1 or -1."""
    signs = []
    for kind in (kind_l, kind_r):
        signed = signed_reinforcement(kind)
        if signed != 0.0:
            signs.append(signed)
    if not signs:
        return 0.0
    if all(item > 0.0 for item in signs):
        return 1.0
    if all(item < 0.0 for item in signs):
        return -1.0
    return float(sum(signs)) / float(len(signs))


class Stimulus:
    """One reinforcement source. The brain reads ``name``, ``active``, and ``value``."""

    def __init__(self, name: str):
        self.name = str(name)
        self.active = False
        self.value = 0.0

    def status(self) -> dict:
        return {"name": self.name, "active": bool(self.active), "value": float(self.value)}

    def offer(self, now: float, learning: bool, skip_l: str | None = None, skip_r: str | None = None) -> tuple[float, float]:
        """Signed reinforcement for the left eye and the right eye. Silence is (0, 0)."""
        return 0.0, 0.0


class StimulusSet:
    """The plug. Attach a source; the brain never looks up its name."""

    def __init__(self) -> None:
        self._items: list[Stimulus] = []
        self.muted = False

    def attach(self, stimulus: Stimulus) -> None:
        name = stimulus.name
        self._items = [item for item in self._items if item.name != name]
        self._items.append(stimulus)

    def get(self, name: str) -> Stimulus | None:
        for item in self._items:
            if item.name == name:
                return item
        return None

    def status(self) -> list:
        return [item.status() for item in self._items]

    def reinforcement(
        self,
        now: float,
        learning: bool,
        skip_l: str | None = None,
        skip_r: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Sum every attached source, then turn the sign into PAM or PPL1.

        ``muted`` drops every source. The gates do not advance.
        """
        if self.muted:
            return None, None
        left = 0.0
        right = 0.0
        for item in self._items:
            part_l, part_r = item.offer(now, learning, skip_l, skip_r)
            left += float(part_l)
            right += float(part_r)
        return reinforcement_kind(left), reinforcement_kind(right)


def _metrics(box) -> tuple[float, float, float]:
    x0 = float(box.x0)
    x1 = float(box.x1)
    y0 = float(getattr(box, "y0", 0.0))
    y1 = float(getattr(box, "y1", 1.0))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    width = x1 - x0
    height = y1 - y0
    if width < 0.0:
        width = 0.0
    if height < 0.0:
        height = 0.0
    return width, height, width * height


def camera_stimuli(boxes, prev_area: float | None, conf_min: float = 0.5) -> tuple[bool, bool, float | None]:
    """``(too_close, approaching, area)`` from YOLO boxes. Lidar is not an argument.

    ``too_close`` is a box whose height reaches ``CLOSE_HEIGHT`` or whose area
    reaches ``CLOSE_AREA``. ``approaching`` is the strongest box growing by
    ``APPROACH_GROWTH``. ``area`` is that box, or None when nothing qualifies.
    """
    best_area = None
    best_conf = float(conf_min) - 1e-9
    close = False
    for box in boxes or ():
        conf = float(getattr(box, "conf", 0.0))
        if conf < float(conf_min):
            continue
        _width, height, area = _metrics(box)
        if height + 1e-12 >= CLOSE_HEIGHT or area + 1e-12 >= CLOSE_AREA:
            close = True
        if conf >= best_conf:
            best_conf = conf
            best_area = area
    grew = False
    if best_area is not None and prev_area is not None and best_area > float(prev_area) + APPROACH_GROWTH:
        grew = True
    return close, grew, best_area
