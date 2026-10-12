"""Innate search. It sits outside the mushroom body and does not teach.

While YOLO has no box the body turns slowly in place. The moment a target
is in the frame, yaw comes from the two approach-minus-avoidance drives.
"""

from __future__ import annotations

from .hemifield import live_turn
from .pilot import STOP_M, X_MAX, DriveCommand

# Slower than the old burst (SEARCH_TURN is 0.35, then a pause). One
# direction, no pause, so a target behind the body eventually crosses the view.
SEARCH_REFLEX_YAW = 0.18


def yolo_boxes_seen(boxes) -> bool:
    """True when the detector returned at least one box."""
    return bool(boxes)


def cover_command(cmd, seen: bool, r_l: float, r_r: float, dist_m, forward_m, base_l=None, base_r=None):
    """Autonomy gate. Returns ``(command, searching)``.

    A stop is left alone. An empty frame is the slow scan and ``searching``
    is true, which is the caller's cue to skip plasticity. A target in view
    hands the yaw to ``live_turn`` and walks until the stop distance.
    ``base_*`` is each hemisphere's birth drive for this view.
    """
    hint = getattr(cmd, "hint", "") or ""
    if getattr(cmd, "phase", "") == "stop":
        return cmd, False
    if not seen:
        return DriveCommand(0.0, SEARCH_REFLEX_YAW, False, "search", "мозг", hint), True
    yaw = float(live_turn(float(r_l), float(r_r), base_l, base_r))
    dist = None if dist_m is None else float(dist_m)
    ahead = None if forward_m is None else float(forward_m)
    close = dist is not None and dist <= float(STOP_M)
    blocked = ahead is not None and ahead < float(STOP_M)
    if close or blocked:
        return DriveCommand(0.0, yaw, abs(yaw) < 1e-6, "hold", "мозг", hint), False
    return DriveCommand(float(X_MAX), yaw, False, "approach", "мозг", hint), False
