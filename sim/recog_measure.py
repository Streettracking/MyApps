"""«ЗАМЕР ДО» / «ЗАМЕР ПОСЛЕ»: УЗНАЮ without and with the .128 dog in frame.

Press: ОБУЧЕНИЕ and АВТО go off through their usual buttons. Then 15 s «без .128»
(every frame's УЗНАЮ is collected), 5 s «поставь .128» (not collected), 15 s
«с .128». The result goes to the journal and under the buttons. The same button
again cancels. A measurement never turns training back on.
"""

from __future__ import annotations

import time

PHASE_S = 15.0
PLACE_S = 5.0
# Verdict thresholds, УЗНАЮ as a fraction 0..1.
GAIN_MIN = 0.05  # после.разница must beat до.разница by more than this
BG_KEEP = 0.05  # после.без within до.без + this: the background stayed quiet
BG_RISE = 0.10  # после.без above до.без + this: it learned the background

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


def _num(v) -> str:
    return "—" if v is None else "%.2f" % v


def verdict(before: dict, after: dict) -> str:
    bb, bw, ab, aw = before.get("bez"), before.get("s"), after.get("bez"), after.get("s")
    if None in (bb, bw, ab, aw):
        return "итог: нет данных"
    if (aw - ab) > (bw - bb) + GAIN_MIN and ab <= bb + BG_KEEP:
        return "итог: узнаёт"
    if ab > bb + BG_RISE:
        return "итог: выучил фон"
    return "итог: без изменений"


class Measure:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.which: str | None = None  # "before" / "after" while running
        self.t0 = 0.0
        self.bez: list[float] = []
        self.s: list[float] = []
        self.results: dict[str, dict] = {}
        self._placed = False

    @property
    def running(self) -> bool:
        return self.which is not None

    def press(self, which: str, log) -> bool:
        """Returns True when a run starts (caller turns ОБУЧЕНИЕ/АВТО off)."""
        if self.which == which:
            self.which = None
            log("замер отменён")
            return False
        if self.which is not None:
            log("замер отменён")
        self.which, self.t0, self.bez, self.s, self._placed = which, float(self.clock()), [], [], False
        log("замер %s: без .128, %d с" % (WHICH[which], PHASE_S))
        return True

    def phase(self) -> tuple[str, float]:
        """("bez"|"place"|"s"|"done", seconds left in that phase)."""
        el = float(self.clock()) - self.t0
        for name, end in (("bez", PHASE_S), ("place", PHASE_S + PLACE_S), ("s", 2 * PHASE_S + PLACE_S)):
            if el < end:
                return name, end - el
        return "done", 0.0

    def tick(self, value: float | None, log) -> str | None:
        """Once per frame. Returns the result line when a run finishes."""
        if self.which is None:
            return None
        name, _left = self.phase()
        if name == "place" and not self._placed:
            self._placed = True
            log("поставь .128 в кадр")
        if name == "s" and not self._placed:
            self._placed = True
        if name == "bez" and value is not None:
            self.bez.append(float(value))
        elif name == "s" and value is not None:
            self.s.append(float(value))
        if name != "done":
            return None
        which, self.which = self.which, None
        res = {"bez": _mean(self.bez), "s": _mean(self.s)}
        self.results[which] = res
        diff = None if None in (res["bez"], res["s"]) else res["s"] - res["bez"]
        line = "замер %s: без=%s с=%s разница=%s" % (
            WHICH[which], _num(res["bez"]), _num(res["s"]), "—" if diff is None else "%+.2f" % diff
        )
        log(line)
        if which == "after" and "before" in self.results:
            log(verdict(self.results["before"], res))
        return line

    def button_text(self, which: str) -> str:
        base = "ЗАМЕР " + WHICH[which]
        if self.which != which:
            return base
        name, left = self.phase()
        word = {"bez": "без .128", "place": "поставь .128", "s": "с .128"}.get(name, "с .128")
        return "%s: %2d с" % (word, max(0, int(left + 0.999)))

    def lines(self) -> tuple[str, str]:
        out = []
        for which, word in (("before", "до:    "), ("after", "после: ")):
            r = self.results.get(which, {})
            out.append("%sбез %s · с %s" % (word, _num(r.get("bez")).rjust(4), _num(r.get("s")).rjust(4)))
        return out[0], out[1]


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
    view.measure_lines = m.lines()
