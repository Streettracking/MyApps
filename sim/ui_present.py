"""Presentation-only labels for the trainer window.

The brain panel already says who leads (``lead_plaque`` / teacher weight).
The header repeats that, and names the operator when the stick is theirs.
Nothing here changes a command, a weight, or the onboard link.
"""

from __future__ import annotations

DRIVER_FLY = "МУХА"
DRIVER_TEACHER = "УЧИТЕЛЬ"
DRIVER_HUMAN = "ЧЕЛОВЕК"
DRIVER_MIX = "СМЕШАННО"


def _text(view, name: str) -> str:
    return str(getattr(view, name, "") or "")


def operator_has_stick(view) -> bool:
    """Arrow keys, ПЕРЕХВАТ, or manual control. E-STOP is not a driver."""
    mode = _text(view, "pilot_mode")
    who = _text(view, "pilot_who")
    phase = _text(view, "pilot_phase")
    command = _text(view, "last_command").strip()
    arrows = bool(command) and command not in ("—", "стоп")
    if mode == "СТОП":
        return False
    if "ПЕРЕХВАТ" in mode:
        return True
    if mode.startswith("РУЧНОЕ") or phase == "manual" or who == "оператор":
        return True
    if arrows and not bool(getattr(view, "autonomy_on", False)):
        return True
    return False


def driver_name(view) -> str:
    """``МУХА`` / ``УЧИТЕЛЬ`` / ``СМЕШАННО`` from the brain plaque, or ``ЧЕЛОВЕК``."""
    if operator_has_stick(view):
        return DRIVER_HUMAN
    plaque = _text(view, "lead_plaque")
    if "смешан" in plaque:
        return DRIVER_MIX
    if "учитель" in plaque:
        return DRIVER_TEACHER
    if "муха" in plaque:
        return DRIVER_FLY
    try:
        weight = float(getattr(view, "teacher_weight", 0.0) or 0.0)
    except (TypeError, ValueError):
        weight = 0.0
    if weight >= 0.995:
        return DRIVER_TEACHER
    if weight > 0.005:
        return DRIVER_MIX
    return DRIVER_FLY


def driver_banner(view) -> str:
    return "ВЕДЁТ: %s" % driver_name(view)


def driver_journal_line(name: str) -> str:
    return "ведёт: %s" % name


def known_weights_name(state_path) -> str:
    """YOLO file the trainer already resolved, else the brain state file name."""
    import os
    from pathlib import Path

    yolo = os.environ.get("RECOGNIZER_YOLO_WEIGHTS", "").strip()
    if yolo:
        path = Path(yolo)
        if path.is_file():
            return path.name
    if state_path:
        name = Path(str(state_path)).name
        if name and name not in (".", ".."):
            return name
    return ""


def weights_caption(name: str) -> str:
    text = str(name or "").strip()
    return "веса: %s" % text if text else "веса: ?"
