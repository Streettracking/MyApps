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


def stamp_now(now: float | None = None) -> str:
    now = time.time() if now is None else float(now)
    return time.strftime("%H:%M:%S", time.localtime(now)) + ".%03d" % int((now % 1.0) * 1000)


def install_journal(session, where: str) -> Journal:
    """File journal for a real trainer loop (sim or robot). Every line is flushed."""
    j = Journal(journal_path())
    session.journal = j
    j.write("тренер запущен (%s)" % where)
    return j


def mirror_log(session, text: str) -> None:
    """session._log text: also into journal.log and, stamped, into trainer.log (stdout)."""
    j = getattr(session, "journal", None)
    if j is None or j.path is None:
        return
    j.write("· " + str(text))
    try:
        print("%s  %s" % (stamp_now(), text), flush=True)
    except Exception:
        pass


def _pct(conf) -> str:
    if conf is None:
        return "—"
    if not getattr(conf, "ready", False):
        return "калибровка"
    return "%3.0f%%%s" % (float(getattr(conf, "percent", 0.0)), " ✓" if getattr(conf, "recognized", False) else "")


def uznayu_line(session) -> str | None:
    """«УЗНАЮ» joint and per eye, from the dog's status or the local brain."""
    if getattr(session, "onboard", False):
        r = getattr(session, "remote", None) or {}
        if "confidence" not in r:
            return None

        def one(key, ready, rec):
            if not r.get(ready, True):
                return "калибровка"
            return "%3.0f%%%s" % (float(r.get(key) or 0.0), " ✓" if r.get(rec) else "")

        return "УЗНАЮ %s  L %s  R %s" % (
            one("confidence", "confidence_ready", "recognized"),
            one("confidence_L", "confidence_L_ready", "recognized_L"),
            one("confidence_R", "confidence_R_ready", "recognized_R"),
        )
    mb = getattr(session, "mb", None)
    if mb is None:
        return None
    return "УЗНАЮ %s  L %s  R %s" % (_pct(getattr(mb, "conf", None)), _pct(getattr(mb, "conf_l", None)), _pct(getattr(mb, "conf_r", None)))


def _pam_total(session) -> int | None:
    if getattr(session, "onboard", False):
        r = getattr(session, "remote", None) or {}
        for key in ("n_pam_total", "pam_count", "n_pam"):
            if r.get(key) is not None:
                try:
                    return int(r[key])
                except (TypeError, ValueError):
                    return None
        return None
    mb = getattr(session, "mb", None)
    return None if mb is None else int(getattr(mb, "n_pam", 0))


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
    focus_lost = not bool(getattr(inp, "focused", True))
    if focus_lost:
        try:
            from .recognize_train_live import effective_focus

            focus_lost = not effective_focus(session, inp)
        except Exception:
            pass
    if focus_lost != bool(getattr(session, "_j_focus", False)):
        j.write("фокус потерян: стоп" if focus_lost else "фокус вернулся")
    session._j_focus = focus_lost
    pilot = getattr(session, "pilot", None)
    mode = str(getattr(pilot, "mode", "") or "")
    if mode and mode != getattr(session, "_j_mode", None) and getattr(session, "_j_mode", None) is not None:
        j.write("режим: %s" % mode)
    session._j_mode = mode
    now = float(j.clock())
    if now - float(getattr(session, "_j_uz_t", -1e9)) >= 1.0:
        session._j_uz_t = now
        line = uznayu_line(session)
        if line:
            j.write(line)
    pam = _pam_total(session)
    last_pam = getattr(session, "_j_pam", None)
    if pam is not None and last_pam is not None and pam > last_pam:
        j.write("PAM +%d, всего %d" % (pam - last_pam, pam))
    if pam is not None:
        session._j_pam = pam
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