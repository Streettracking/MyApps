"""Trainer journal: every operator intervention and every negative (PPL1) signal.

One line per event, wall-clock time, appended to ``journal.log`` beside
``trainer.log`` (%APPDATA%/recognize_trainer on Windows). Never raises.
"""

from __future__ import annotations

import time
from pathlib import Path

JOURNAL_NAME = "journal.log"


def journal_path() -> Path:
    from .ui_settings import trainer_log_path

    return trainer_log_path().with_name(JOURNAL_NAME)


class Journal:
    def __init__(self, path: Path | None = None, clock=time.time):
        self.path = path
        self.clock = clock
        self.lines: list[str] = []

    def write(self, text: str) -> str:
        now = float(self.clock())
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + ".%03d" % int((now % 1.0) * 1000)
        line = "%s  %s" % (stamp, text)
        self.lines.append(line)
        del self.lines[:-200]
        if self.path is None:
            return line
        try:
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        return line


def journal_of(session) -> Journal:
    # Memory only unless the trainer loop installed a file journal (tests never write to APPDATA).
    j = getattr(session, "journal", None)
    if j is None:
        j = Journal(None)
        session.journal = j
    return j


def note_ppl1_sent(session, side_l, side_r, how: str = "teach_sides") -> None:
    sides = [s for s, k in (("L", side_l), ("R", side_r)) if k == "ppl1"]
    if sides:
        journal_of(session).write("PPL1 отправлен (%s) %s" % (how, "+".join(sides)))


def _ppl1_total(session) -> int | None:
    remote = getattr(session, "remote", None) or {}
    if getattr(session, "onboard", False):
        for key in ("n_ppl1_total", "ppl1_count", "n_ppl1"):
            if remote.get(key) is not None:
                try:
                    return int(remote[key])
                except (TypeError, ValueError):
                    return None
        return None
    mb = getattr(session, "mb", None)
    return None if mb is None else int(getattr(mb, "n_ppl1", 0))


def _state(session) -> dict:
    remote = getattr(session, "remote", None) or {}
    teacher = getattr(session, "teacher", None)
    pilot = getattr(session, "pilot", None)
    onboard = bool(getattr(session, "onboard", False))
    return {
        "teacher": bool(teacher is not None and getattr(teacher, "on", False)),
        "learn": bool(remote.get("learning")) if onboard else bool(getattr(session, "learn", False)),
        "auto": bool(remote.get("autonomy")) if onboard else bool(getattr(pilot, "autonomy", False)),
    }


_STATE_WORDS = {
    "teacher": ("учитель включён", "учитель выключен"),
    "learn": ("обучение включено", "обучение выключено"),
    "auto": ("автономия включена", "автономия выключена"),
}


def journal_frame(session, inp) -> None:
    """Call once per frame after the keys and the teacher ran."""
    j = journal_of(session)
    if getattr(inp, "estop", False):
        j.write("E-STOP")
    if getattr(inp, "stop", False):
        j.write("стоп (пробел)")
    if getattr(inp, "takeover", False):
        j.write("перехват (ручное)")
    if getattr(inp, "punish", False):
        j.write("ошибка (X)")
    if getattr(inp, "treat", False):
        j.write("лакомство (T)")
    if getattr(inp, "teacher_toggle", False):
        j.write("кнопка учителя")
    if getattr(inp, "pause_learn", False):
        j.write("кнопка обучения")
    if getattr(inp, "autonomy_toggle", False):
        j.write("кнопка автономии")
    arrows = abs(float(getattr(inp, "steer_x", 0.0) or 0.0)) + abs(float(getattr(inp, "steer_z", 0.0) or 0.0)) > 0
    if arrows and not getattr(session, "_j_arrows", False):
        j.write("стрелки x=%+.0f z=%+.0f" % (float(inp.steer_x), float(inp.steer_z)))
    session._j_arrows = arrows
    focus_halt = bool(getattr(session, "_teacher_halt", False)) and not bool(getattr(inp, "focused", True))
    try:
        from .recognize_train_live import effective_focus

        focus_halt = focus_halt and not effective_focus(session, inp)
    except Exception:
        pass
    if focus_halt != bool(getattr(session, "_j_focus", False)):
        j.write("фокус потерян: учитель стоит" if focus_halt else "фокус вернулся")
    session._j_focus = focus_halt
    state = _state(session)
    prev = getattr(session, "_j_state", None)
    if prev is not None:
        for key, (on_word, off_word) in _STATE_WORDS.items():
            if state[key] != prev[key]:
                j.write(on_word if state[key] else off_word)
    session._j_state = state
    total = _ppl1_total(session)
    last = getattr(session, "_j_ppl1", None)
    if total is not None and last is not None and total > last:
        j.write("PPL1 (негатив) %s +%d, всего %d" % ("на борту" if getattr(session, "onboard", False) else "в мозге", total - last, total))
    if total is not None:
        session._j_ppl1 = total