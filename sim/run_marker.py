"""Operator mark for one training pass: stage and which side is the target.

The journal receives one line when the mark changes. Every frame appends the
raw left and right hemisphere drives under the mark that was active then.
The clock is the caller's time. This module does not read the wall clock.
"""

from __future__ import annotations

from collections import deque

STAGES = ("до", "обучение", "после")
SIDES = ("слева", "справа")
DRIVE_LOG_MAX = 240

_STAGE_ALIAS = {
    "до": "до",
    "before": "до",
    "pre": "до",
    "обучение": "обучение",
    "train": "обучение",
    "training": "обучение",
    "после": "после",
    "after": "после",
    "post": "после",
}
_SIDE_ALIAS = {
    "слева": "слева",
    "left": "слева",
    "l": "слева",
    "л": "слева",
    "справа": "справа",
    "right": "справа",
    "r": "справа",
    "п": "справа",
}


def canon_stage(value) -> str | None:
    if value is None:
        return None
    return _STAGE_ALIAS.get(str(value).strip().lower())


def canon_side(value) -> str | None:
    if value is None:
        return None
    return _SIDE_ALIAS.get(str(value).strip().lower())


def run_label(stage: str, side: str) -> str:
    return "%s · %s" % (stage, side)


class RunMarker:
    def __init__(self) -> None:
        self.stage = "до"
        self.side = "слева"
        self.drive_log: deque = deque(maxlen=DRIVE_LOG_MAX)

    def label(self) -> str:
        return run_label(self.stage, self.side)

    def set_stage(self, value) -> bool:
        mapped = canon_stage(value)
        if mapped is None or mapped == self.stage:
            return False
        self.stage = mapped
        return True

    def set_side(self, value) -> bool:
        mapped = canon_side(value)
        if mapped is None or mapped == self.side:
            return False
        self.side = mapped
        return True

    def note(self, t: float, drive_l: float, drive_r: float) -> None:
        """One frame. The label is the mark at this frame, not a later edit."""
        self.drive_log.append(
            {
                "t": float(t),
                "label": self.label(),
                "stage": self.stage,
                "side": self.side,
                "drive_l": float(drive_l),
                "drive_r": float(drive_r),
            }
        )

    def adopt_status(self, status: dict) -> bool:
        """Copy a robot status onto this mark. Missing keys leave the local mark.

        True when the label changed, so the trainer can write one journal line.
        """
        if not isinstance(status, dict):
            return False
        before = self.label()
        stage = canon_stage(status.get("run_stage"))
        side = canon_side(status.get("run_side"))
        if stage:
            self.stage = stage
        if side:
            self.side = side
        rows = status.get("drive_log")
        if isinstance(rows, list):
            self.drive_log.clear()
            for row in rows[-DRIVE_LOG_MAX:]:
                if isinstance(row, dict):
                    self.drive_log.append(dict(row))
        return self.label() != before

    def status(self) -> dict:
        return {
            "run_stage": self.stage,
            "run_side": self.side,
            "run_label": self.label(),
            "drive_log": [dict(row) for row in self.drive_log],
        }


def apply_marker_fields(session, stage, side) -> bool:
    """Set the mark. One journal line when either field actually changes."""
    marker = getattr(session, "run_marker", None)
    if marker is None:
        return False
    changed = False
    if stage:
        changed = marker.set_stage(stage) or changed
    if side:
        changed = marker.set_side(side) or changed
    if changed and hasattr(session, "_log"):
        session._log("заход: %s" % marker.label())
    return changed


def apply_marker_input(session, inp, onboard=None) -> bool:
    """Trainer click. A robot link receives the same mark for its frame log."""
    stage = getattr(inp, "run_stage", None)
    side = getattr(inp, "run_side", None)
    if not stage and not side:
        return False
    changed = apply_marker_fields(session, stage, side)
    if onboard is not None:
        marker = getattr(session, "run_marker", None)
        onboard.post(
            "run_marker",
            stage=marker.stage if marker is not None else (stage or "до"),
            side=marker.side if marker is not None else (side or "слева"),
        )
    return changed
