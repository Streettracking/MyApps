"""«ЗАМЕР ДО» / «ЗАМЕР ПОСЛЕ»: УЗНАЮ without and with the .128 dog in frame.

Press: ОБУЧЕНИЕ and АВТО go off through their usual buttons. The run then
waits until the board reports calibration ready («УБЕРИ .128 ИЗ КАДРА»; while
it is not ready the prompt is «жду калибровку борта…» and the clock stays
put). Then 15 s «БЕЗ .128» (every ready frame's УЗНАЮ is averaged), 5 s
«ПОСТАВЬ .128 В КАДР» (not averaged), 15 s «С .128». The same button again
cancels. A measurement never turns training back on.

Each step start, step end, and the averages go to the session log, which the
trainer journal stamps into journal.log.
"""

from __future__ import annotations

import time

PHASE_S = 15.0
PLACE_S = 5.0
# «узнаёт»: с is clearly above без. «выучил фон»: both sit high.
CLEAR = 0.15
HIGH = 0.50

WHICH = {"before": "ДО", "after": "ПОСЛЕ"}


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


def _pct(v) -> str:
    """Monospace percent cell: `` 10%``, ``100%``, or ``   —``."""
    if v is None:
        return "   —"
    return "%3d%%" % int(round(100.0 * float(v)))


def _pct_log(v) -> str:
    if v is None:
        return "—"
    return "%d%%" % int(round(100.0 * float(v)))


def verdict(bez, s) -> str:
    """«узнаёт» when с is clearly above без, «выучил фон» when both are high."""
    if bez is None or s is None:
        return ""
    if float(bez) >= HIGH and float(s) >= HIGH:
        return "выучил фон"
    if float(s) - float(bez) > CLEAR:
        return "узнаёт"
    return "нет разницы"


class Measure:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.which: str | None = None  # "before" / "after" while running
        self.t0 = 0.0
        self.armed = False
        self.bez: list[float] = []
        self.s: list[float] = []
        self.results: dict[str, dict] = {}
        self.word = ""
        self._mark = ""
        self._saw_unready = False

    @property
    def running(self) -> bool:
        return self.which is not None

    def press(self, which: str, log) -> bool:
        """Returns True when a run starts (caller turns ОБУЧЕНИЕ/АВТО off)."""
        if self.which == which:
            self.which = None
            self.armed = False
            log("замер отменён")
            return False
        if self.which is not None:
            log("замер отменён")
        self.which = which
        self.t0 = 0.0
        self.armed = False
        self.bez = []
        self.s = []
        self._mark = "wait"
        self._saw_unready = False
        log("замер %s: старт убери .128" % WHICH[which])
        return True

    def phase(self) -> tuple[str, float]:
        """("wait"|"bez"|"place"|"s"|"done"|"idle", seconds left in that phase)."""
        if self.which is None:
            return "idle", 0.0
        if not self.armed:
            return "wait", 0.0
        el = float(self.clock()) - self.t0
        for name, end in (("bez", PHASE_S), ("place", PHASE_S + PLACE_S), ("s", 2 * PHASE_S + PLACE_S)):
            if el < end:
                return name, end - el
        return "done", 0.0

    def prompt(self) -> str:
        if self.which is None:
            return ""
        name, left = self.phase()
        sec = max(0, int(left + 0.999))
        if name == "wait":
            if self._saw_unready:
                return "жду калибровку борта…"
            return "УБЕРИ .128 ИЗ КАДРА"
        if name == "bez":
            return "БЕЗ .128: %d с" % sec
        if name == "place":
            return "ПОСТАВЬ .128 В КАДР: %d с" % sec
        if name == "s":
            return "С .128: %d с" % sec
        return ""

    def _arm(self, log) -> None:
        self.armed = True
        self.t0 = float(self.clock())
        self._mark = "bez"
        log("замер %s: конец убери .128" % WHICH[self.which])
        log("замер %s: старт без .128" % WHICH[self.which])

    def _cross(self, name: str, log) -> None:
        """Log the end of the phase we left and the start of the one we entered."""
        order = ("bez", "place", "s", "done")
        if name == self._mark or self._mark not in order:
            return
        while self._mark != name:
            if self._mark == "bez":
                log("замер %s: конец без .128 среднее=%s" % (WHICH[self.which], _pct_log(_mean(self.bez))))
            elif self._mark == "place":
                log("замер %s: конец поставь .128" % WHICH[self.which])
            elif self._mark == "s":
                log("замер %s: конец с .128 среднее=%s" % (WHICH[self.which], _pct_log(_mean(self.s))))
            nxt = order[order.index(self._mark) + 1]
            self._mark = nxt
            if nxt == "place":
                log("замер %s: старт поставь .128" % WHICH[self.which])
            elif nxt == "s":
                log("замер %s: старт с .128" % WHICH[self.which])
            if nxt == "done" or nxt == name:
                break

    def tick(self, value: float | None, log) -> str | None:
        """Once per frame. Returns the result line when a run finishes."""
        if self.which is None:
            return None
        if not self.armed:
            if value is None:
                if not self._saw_unready:
                    self._saw_unready = True
                    log("замер %s: жду калибровку борта" % WHICH[self.which])
                return None
            self._arm(log)
        name, _left = self.phase()
        self._cross(name, log)
        if name == "bez" and value is not None:
            self.bez.append(float(value))
        elif name == "s" and value is not None:
            self.s.append(float(value))
        if name != "done":
            return None
        which, self.which = self.which, None
        self.armed = False
        res = {"bez": _mean(self.bez), "s": _mean(self.s)}
        self.results[which] = res
        line = "замер %s: без=%s с=%s" % (WHICH[which], _pct_log(res["bez"]), _pct_log(res["s"]))
        log(line)
        self.word = verdict(res["bez"], res["s"])
        if self.word:
            log(self.word)
        return line

    def button_text(self, which: str) -> str:
        return "ЗАМЕР " + WHICH[which]

    def line(self) -> str:
        parts = []
        for which, word in (("before", "до"), ("after", "после")):
            r = self.results.get(which, {})
            parts.append("%s: без %s · с %s" % (word, _pct(r.get("bez")), _pct(r.get("s"))))
        text = "   ".join(parts)
        if self.word:
            text += "   " + self.word
        return text

    def lines(self) -> tuple[str, ...]:
        return (self.line(),)


def measure_of(session) -> Measure:
    m = getattr(session, "_measure", None)
    if m is None:
        m = session._measure = Measure()
    return m


def drive_measure(session, inp) -> None:
    """Before the keys: a press turns ОБУЧЕНИЕ / АВТО off through their own buttons."""
    from .trainer_journal import _state

    m = measure_of(session)
    log = session._log
    for which, flag in (("before", "measure_before"), ("after", "measure_after")):
        if getattr(inp, flag, False) and m.press(which, log):
            state = _state(session)
            if state["learn"]:
                inp.pause_learn = True
            if state["auto"]:
                inp.autonomy_toggle = True


def measure_frame(session, view) -> None:
    """After the frame: collect, finish, and put the texts on the view."""
    m = measure_of(session)
    m.tick(uznayu_value(session), session._log)
    view.measure_run = m.which or ""
    view.measure_before = m.button_text("before")
    view.measure_after = m.button_text("after")
    view.measure_prompt = m.prompt()
    view.measure_line = m.line()
    view.measure_lines = m.lines()
