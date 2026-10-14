"""«БЕЗ .128» / «С .128»: two independent 15 s averages of УЗНАЮ.

The user places the dog and presses when ready. There is no pause and no
automatic step between the buttons. A press while board status is not
calibrated shows «борт не готов: калибровка X/200» and does not start.
Otherwise ОБУЧЕНИЕ and АВТО go off for the 15 s, the lit button and the
prompt count down, and both return to the states they had before the press.
The same button again cancels. The other button cancels the current run
without storing it and starts its own 15 s.

The strip keeps only the latest value of each side:
«без X% · с Y% · разница Z». A new measurement replaces that side.
Each finished result is written to the session log, which the trainer
journal stamps into journal.log.

Board status never shows the trainer's own 200-frame counter. When the
board sends confidence_calib_n / confidence_calib_need (and the _L / _R
pairs), the strip shows «калибровка n/need» and БЕЗ / С start only once
every reported window is full. A board without those keys shows
«борт: готов» or «борт: не готов» from confidence_ready and does not
block the buttons.
"""

from __future__ import annotations

import time

PHASE_S = 15.0

# Input flags stay measure_before / measure_after. The run id is the side.
WHICH = {"bez": "без .128", "s": "с .128"}
BUTTON = {"bez": "БЕЗ .128", "s": "С .128"}


def _window_pair(remote, suffix: str):
    """``(n, need)`` for one confidence window, or None when that pair is absent."""
    n_key = "confidence_calib_n" + suffix
    need_key = "confidence_calib_need" + suffix
    if n_key not in remote or need_key not in remote:
        return None
    try:
        n = int(remote[n_key])
        need = int(remote[need_key])
    except (TypeError, ValueError):
        return None
    if need <= 0:
        return None
    return max(0, n), need


def _confidence_ready(session, remote) -> bool:
    if isinstance(remote, dict) and "confidence_ready" in remote:
        return bool(remote.get("confidence_ready"))
    conf = getattr(getattr(session, "mb", None), "conf", None)
    if conf is not None and hasattr(conf, "ready"):
        return bool(conf.ready)
    return not bool(getattr(session, "onboard", False))


def board_calib(session):
    """Board status for the measure strip. Never the trainer's 200 counter.

    A confidence window (``confidence_calib_n`` / ``confidence_calib_need``,
    plus ``_L`` / ``_R`` when the board sends them) is
    ``("window", n, need, full)``. ``n/need`` is the window that is furthest
    from full; ``full`` is true only when every reported window is filled.
    Without those keys the result is ``("flag", ready)`` from
    ``confidence_ready``, and the buttons are not gated on it.
    ``calib_n`` is the trainer counter and is ignored. A direct ``(n, need)``
    tuple is the same window, for tests.
    """
    remote = getattr(session, "remote", None)
    remote = remote if isinstance(remote, dict) else {}
    if getattr(session, "onboard", False):
        pairs = []
        for suffix in ("", "_L", "_R"):
            pair = _window_pair(remote, suffix)
            if pair is not None:
                pairs.append((suffix, pair[0], pair[1]))
        if pairs:
            incomplete = [item for item in pairs if item[1] < item[2]]
            if incomplete:
                _suffix, n, need = min(incomplete, key=lambda item: (item[1] / item[2], item[1]))
                return ("window", n, need, False)
            for suffix in ("", "_L", "_R"):
                for item in pairs:
                    if item[0] == suffix:
                        return ("window", item[1], item[2], True)
    return ("flag", _confidence_ready(session, remote))


def _as_window(board):
    """``(n, need, full)`` when ``board`` is a confidence window, else None."""
    if isinstance(board, tuple) and len(board) == 2:
        n, need = board
        try:
            n, need = int(n), int(need)
        except (TypeError, ValueError):
            return None
        if need <= 0:
            return None
        return n, need, n >= need
    if isinstance(board, tuple) and len(board) == 4 and board[0] == "window":
        return int(board[1]), int(board[2]), bool(board[3])
    return None


def board_ready(board) -> bool:
    """Buttons run unless a reported confidence window is still filling."""
    window = _as_window(board)
    if window is None:
        return True
    _n, _need, full = window
    return full


def board_phrase(board) -> str:
    """«калибровка n/need» for a window, else «борт: готов» / «борт: не готов»."""
    window = _as_window(board)
    if window is not None:
        n, need, _full = window
        return "калибровка %d/%d" % (n, need)
    if isinstance(board, tuple) and len(board) >= 2 and board[0] == "flag":
        return "борт: готов" if board[1] else "борт: не готов"
    return "борт: не готов"


def board_wait(board) -> str:
    """Why a press did not start."""
    return "борт не готов: " + board_phrase(board)


def uznayu_value(session) -> float | None:
    """Joint УЗНАЮ as 0..1, None while calibrating or unknown."""
    if getattr(session, "onboard", False):
        r = getattr(session, "remote", None) or {}
        if "confidence" not in r or not r.get("confidence_ready", True):
            return None
        try:
            return float(r.get("confidence") or 0.0) / 100.0
        except (TypeError, ValueError):
            return None
    conf = getattr(getattr(session, "mb", None), "conf", None)
    if conf is None or not getattr(conf, "ready", False):
        return None
    return float(getattr(conf, "percent", 0.0)) / 100.0


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _points(v):
    if v is None:
        return None
    return int(round(100.0 * float(v)))


def _pct(v) -> str:
    """Monospace percent cell: `` 10%``, ``100%``, or ``   —``."""
    n = _points(v)
    if n is None:
        return "   —"
    return "%3d%%" % n


def _pct_log(v) -> str:
    n = _points(v)
    if n is None:
        return "—"
    return "%d%%" % n


def _delta(bez, s) -> str:
    """Signed percentage points, с minus без. ``   —`` until both sides exist."""
    b, a = _points(bez), _points(s)
    if b is None or a is None:
        return "   —"
    return "%+d%%" % (a - b)


class Measure:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.which: str | None = None  # "bez" / "s" while that button is running
        self.t0 = 0.0
        self.samples: list[float] = []
        # Latest finished average of each side. A new run replaces only its side.
        self.latest: dict[str, float | None] = {}
        self.blocked = ""
        # learn/auto as they were at the first press of this run. Restored
        # when the run ends or the same button cancels it.
        self.saved = None
        self.pending_restore = False

    @property
    def running(self) -> bool:
        return self.which is not None

    def seconds_left(self) -> int:
        if self.which is None:
            return 0
        left = PHASE_S - (float(self.clock()) - self.t0)
        return max(0, int(left + 0.999))

    def press(self, which: str, log, board=(200, 200)) -> str:
        """"start", "switch", "cancel", or "blocked".

        "start" is the first press of an idle ready board: the caller turns
        ОБУЧЕНИЕ and АВТО off. "switch" replaces the other running side and
        leaves them off. "cancel" is the same button again. "blocked" does
        not start and does not touch ОБУЧЕНИЕ / АВТО.
        """
        self.board = board
        if self.which == which:
            self.which = None
            self.samples = []
            self.blocked = ""
            self.pending_restore = True
            log("замер отменён")
            return "cancel"
        if not board_ready(board):
            if self.which is None:
                self.blocked = board_wait(board)
                log("замер %s: %s" % (WHICH[which], self.blocked))
            return "blocked"
        self.blocked = ""
        switching = self.which is not None
        if switching:
            log("замер отменён")
        self.which = which
        self.t0 = float(self.clock())
        self.samples = []
        self.pending_restore = False
        log("замер %s: старт" % WHICH[which])
        return "switch" if switching else "start"

    def prompt(self) -> str:
        if self.which == "bez":
            return "БЕЗ .128: %d с" % self.seconds_left()
        if self.which == "s":
            return "С .128: %d с" % self.seconds_left()
        return self.blocked

    def tick(self, value: float | None, log, board=(200, 200)) -> str | None:
        """Once per frame. Returns the journal result when a run finishes.

        ``board`` is ``(n, total)`` from board status, or None when the robot
        has not reported calibration. The trainer's own counter is not read.
        A run that already started keeps averaging if the board later drops.
        """
        self.board = board
        if self.blocked and board_ready(board):
            self.blocked = ""
        if self.which is None:
            return None
        if value is not None:
            self.samples.append(float(value))
        if float(self.clock()) - self.t0 < PHASE_S:
            return None
        which, self.which = self.which, None
        mean = _mean(self.samples)
        self.samples = []
        self.latest[which] = mean
        self.pending_restore = True
        line = "замер %s: %s  %s" % (WHICH[which], _pct_log(mean), self.line())
        log(line)
        return line

    def button_text(self, which: str) -> str:
        label = BUTTON[which]
        if self.which == which:
            return "%s  %d с" % (label, self.seconds_left())
        return label

    def line(self) -> str:
        bez = self.latest.get("bez")
        s = self.latest.get("s")
        return "без %s · с %s · разница %s" % (_pct(bez), _pct(s), _delta(bez, s))

    def lines(self) -> tuple[str, ...]:
        return (self.line(),)


def measure_of(session) -> Measure:
    m = getattr(session, "_measure", None)
    if m is None:
        m = session._measure = Measure()
    return m


def _restore(session, inp, m: Measure) -> None:
    """Flip ОБУЧЕНИЕ / АВТО back through their buttons when the run let them go."""
    saved = m.saved
    m.saved = None
    m.pending_restore = False
    if not saved:
        return
    from .trainer_journal import _state

    state = _state(session)
    if bool(state["learn"]) != bool(saved["learn"]):
        inp.pause_learn = True
    if bool(state["auto"]) != bool(saved["auto"]):
        inp.autonomy_toggle = True


def drive_measure(session, inp) -> None:
    """Before the keys: a press turns ОБУЧЕНИЕ / АВТО off; the end turns them back."""
    from .trainer_journal import _state

    m = measure_of(session)
    log = session._log
    pressed = bool(getattr(inp, "measure_before", False) or getattr(inp, "measure_after", False))
    if m.pending_restore and not m.running and not pressed:
        _restore(session, inp, m)
    board = board_calib(session)
    for which, flag in (("bez", "measure_before"), ("s", "measure_after")):
        if not getattr(inp, flag, False):
            continue
        action = m.press(which, log, board)
        if action == "start" and m.saved is None:
            m.saved = dict(_state(session))
            if m.saved["learn"]:
                inp.pause_learn = True
            if m.saved["auto"]:
                inp.autonomy_toggle = True
        elif action == "cancel" and m.saved is not None and not m.running:
            _restore(session, inp, m)


def measure_frame(session, view) -> None:
    """After the frame: collect, finish, and put the texts on the view."""
    m = measure_of(session)
    board = board_calib(session)
    m.tick(uznayu_value(session), session._log, board)
    view.measure_run = m.which or ""
    view.measure_before = m.button_text("bez")
    view.measure_after = m.button_text("s")
    view.measure_prompt = m.prompt()
    view.measure_board = board_phrase(board)
    view.measure_line = m.line()
    view.measure_lines = m.lines()
