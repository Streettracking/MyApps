"""Fields the brain panel paints, shared by the simulator and the onboard status.

The laptop draws one panel. The simulator fills it from the local mushroom
body. The robot publishes the same names on ``GET /status``, and the onboard
client maps that dict onto the same panel fields. A missing status, a status
older than ``BRAIN_STATUS_STALE_S``, or a missing field leaves the block
offline instead of holding the last numbers.
"""

from __future__ import annotations

import numpy as np

# How long a /status body may stand in for a live brain. Older than this, the
# panel stops painting numbers.
BRAIN_STATUS_STALE_S = 2.0

BRAIN_OFFLINE_TITLE = "Мозг на .213 не запущен"

# How many samples the panel actually draws. The robot keeps the same caps.
CURVE_KEEP = 400
DRIFT_KEEP = 120
EVENT_KEEP = 240

# Names on the wire. The panel model below is the only reader.
BRAIN_PANEL_KEYS = (
    "dog_in_frame",
    "dog_in_frame_series",
    "no_dog",
    "no_dog_series",
    "kc_mbon_drift",
    "kc_mbon_drift_series",
    "treats_session",
    "treats_total",
    "kc_bins",
    "kc_on",
    "kc_n",
    "pam_l",
    "pam_r",
    "ppl1_l",
    "ppl1_r",
)

_REQUIRED_NUMBERS = (
    "kc_mbon_drift",
    "treats_session",
    "treats_total",
    "kc_on",
    "kc_n",
    "pam_l",
    "pam_r",
    "ppl1_l",
    "ppl1_r",
)


def brain_header(offline: bool) -> str:
    """Card title. Offline replaces the live title instead of leaving it."""
    if offline:
        return BRAIN_OFFLINE_TITLE
    from .dock_layout import BLOCK_TITLES

    return BLOCK_TITLES["brain"]


def offline_plate(card, dim) -> tuple:
    """Flat grey for a brain card that has nothing current to show."""
    return tuple(int(round(float(card[i]) * 0.45 + float(dim[i]) * 0.55)) for i in range(3))


class PanelTrace:
    """Dog / no-dog readout history kept on the robot. The sim keeps its own."""

    def __init__(self) -> None:
        self.dog_series: list[tuple[float, float]] = []
        self.none_series: list[tuple[float, float]] = []
        self.dog_value: float | None = None
        self.none_value: float | None = None
        self._drift_t = -1.0

    def note(self, t: float, value: float, dog: bool, mb) -> None:
        t = float(t)
        value = float(value)
        if dog:
            _push(self.dog_series, t, value, CURVE_KEEP)
            self.dog_value = value
        else:
            _push(self.none_series, t, value, CURVE_KEEP)
            self.none_value = value
        if self._drift_t < 0.0 or t - self._drift_t >= 1.0:
            mb.note_drift(t)
            self._drift_t = t
            if len(mb.drift_curve) > DRIFT_KEEP:
                del mb.drift_curve[:-DRIFT_KEEP]
        if len(mb.dan_events) > EVENT_KEEP:
            del mb.dan_events[:-EVENT_KEEP]


def widget_status(mb, trace: PanelTrace) -> dict:
    """The brain-panel slice of ``GET /status``. Plain lists, no numpy."""
    return {
        "dog_in_frame": _opt_float(trace.dog_value),
        "dog_in_frame_series": _rows_json(trace.dog_series, CURVE_KEEP),
        "no_dog": _opt_float(trace.none_value),
        "no_dog_series": _rows_json(trace.none_series, CURVE_KEEP),
        "kc_mbon_drift": float(mb.last_drift),
        "kc_mbon_drift_series": _rows_json(mb.drift_curve, DRIFT_KEEP),
        "treats_session": int(mb.n_pam),
        "treats_total": int(mb.progress.base_pam + mb.n_pam),
        "kc_bins": _plain_floats(mb.last_kc_bins),
        "kc_on": int(mb.last_kc_on),
        "kc_n": int(mb.brain.n_kc),
        "pam_l": int(mb.teacher_pam_l),
        "pam_r": int(mb.teacher_pam_r),
        "ppl1_l": int(mb.teacher_ppl1_l),
        "ppl1_r": int(mb.teacher_ppl1_r),
        "dan_events": _events_json(mb.dan_events, EVENT_KEEP),
    }


def scale_readout_curves(dog, none, keep: int = CURVE_KEEP):
    """Map both readout curves into 0..1 from their shared range.

    The same stretch the simulator uses for «собака в кадре» and «нет собаки».
    An empty side stays empty.
    """
    dog_rows = [(float(t), float(v)) for t, v in list(dog)[-int(keep) :]]
    none_rows = [(float(t), float(v)) for t, v in list(none)[-int(keep) :]]
    vals = [v for _, v in dog_rows] + [v for _, v in none_rows]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 1.0
    span = hi - lo if hi > lo else 1.0

    def conv(rows):
        return [(t, (v - lo) / span) for t, v in rows]

    return conv(dog_rows), conv(none_rows)


def panel_status_fresh(status, age_s) -> bool:
    """True when this body can drive the panel. Age equal to the limit still counts."""
    if not isinstance(status, dict):
        return False
    try:
        age = float(age_s)
    except (TypeError, ValueError):
        return False
    if age < 0.0 or age > BRAIN_STATUS_STALE_S:
        return False
    for key in BRAIN_PANEL_KEYS:
        if key not in status:
            return False
    if _pairs(status.get("dog_in_frame_series")) is None:
        return False
    if _pairs(status.get("no_dog_series")) is None:
        return False
    if _pairs(status.get("kc_mbon_drift_series")) is None:
        return False
    if not isinstance(status.get("kc_bins"), (list, tuple)):
        return False
    for key in _REQUIRED_NUMBERS:
        if _required_number(status.get(key)) is None:
            return False
    if not _optional_number_ok(status.get("dog_in_frame")):
        return False
    if not _optional_number_ok(status.get("no_dog")):
        return False
    return True


def brain_panel_from_status(status, age_s) -> tuple:
    """``(ok, monitor_fields)``. Offline returns no numbers, so nothing freezes on screen."""
    if not panel_status_fresh(status, age_s):
        return False, {}
    dog, none = scale_readout_curves(
        _pairs(status["dog_in_frame_series"]),
        _pairs(status["no_dog_series"]),
    )
    from .yolo_teacher import counts_line

    fields = {
        "peer_curve": dog,
        "other_curve": none,
        "drift": float(status["kc_mbon_drift"]),
        "drift_curve": _pairs(status["kc_mbon_drift_series"])[-DRIFT_KEEP:],
        "n_pam": int(status["treats_session"]),
        "total_pam": int(status["treats_total"]),
        "lifetime_known": True,
        "kc_on": int(status["kc_on"]),
        "kc_n": int(status["kc_n"]),
        "kc_bins": np.asarray(list(status["kc_bins"]), dtype=np.float32),
        "teacher_counts": counts_line(
            int(status["pam_l"]),
            int(status["pam_r"]),
            int(status["ppl1_l"]),
            int(status["ppl1_r"]),
        ),
    }
    if "n_ppl1" in status and _required_number(status.get("n_ppl1")) is not None:
        fields["n_ppl1"] = int(status["n_ppl1"])
    if "n_ppl1_total" in status and _required_number(status.get("n_ppl1_total")) is not None:
        fields["total_ppl1"] = int(status["n_ppl1_total"])
    events = _events(status.get("dan_events"))
    if events is not None:
        fields["dan_events"] = events[-EVENT_KEEP:]
    return True, fields


def _push(rows: list, t: float, value: float, keep: int) -> None:
    rows.append((float(t), float(value)))
    if len(rows) > int(keep):
        del rows[:-int(keep)]


def _opt_float(value):
    if value is None:
        return None
    return float(value)


def _plain_floats(values) -> list:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        raw = values.tolist()
    else:
        raw = list(values)
    return [float(item) for item in raw]


def _rows_json(rows, keep: int) -> list:
    out = []
    for item in list(rows)[-int(keep) :]:
        out.append([float(item[0]), float(item[1])])
    return out


def _events_json(rows, keep: int) -> list:
    out = []
    for item in list(rows)[-int(keep) :]:
        out.append([float(item[0]), str(item[1])])
    return out


def _pairs(raw):
    if not isinstance(raw, (list, tuple)):
        return None
    out = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        try:
            out.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            return None
    return out


def _events(raw):
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    out = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        try:
            out.append((float(item[0]), str(item[1])))
        except (TypeError, ValueError):
            return None
    return out


def _required_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_number_ok(value) -> bool:
    if value is None:
        return True
    return _required_number(value) is not None
