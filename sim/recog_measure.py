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
journal stamps into journal.log. The block shows board calibration
(«калибровка X/200» or «готов»), never the trainer's own 200-frame counter.
"""

from __future__ import annotations

import time

PHASE_S = 15.0

# Input flags stay measure_before / measure_after. The run id is the side.
WHICH = {"bez": "без .128", "s": "с .128"}
BUTTON = {"bez": "БЕЗ .128", "s": "С .128"}


def board_calib(session) -> tuple[int, int] | None:
    """``(n, total)`` from board status. Never the trainer's own counter.

    A simulator has no separate board, so the measure block reads «готов»
    and does not paint ``mb.calib_progress``. On the robot, only
    ``remote["calib_n"]`` counts. A status without that field is not ready.
    """
    if not getattr(session, "onboard", False):
        return (200, 200)
    remote = getattr(session, "remote", None)
    if not isinstance(remote, dict) or "calib_n" not in remote:
        return None
    try:
        n = int(remote["calib_n"])
        total = int(remote.get("calib_total") or 200)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        total = 200
    return max(0, n), total


def board_ready(board) -> bool:
    if board is None:
        return False
    n, total = board
    return int(total) > 0 and int(n) >= int(total)


def board_phrase(board) -> str:
    """«калибровка X/200» while the board is filling, «готов» when it is done."""
    if board_ready(board):
        return "готов"
    if board is None:
        return "калибровка —/200"
    n, total = board
    return "калибровка %d/%d" % (int(n), int(total))


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
