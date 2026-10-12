import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sim.train_monitor import MonitorInput
from sim.trainer_journal import Journal, journal_frame, note_ppl1_sent


def _session(tmp):
    s = SimpleNamespace(onboard=True, remote={"learning": False, "autonomy": False, "n_ppl1_total": 3},
                        teacher=SimpleNamespace(on=False), pilot=SimpleNamespace(autonomy=False), learn=False,
                        _teacher_halt=False)
    s.journal = Journal(Path(tmp) / "journal.log", clock=lambda: 1760000000.25)
    return s


class JournalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def text(self):
        return (Path(self.tmp) / "journal.log").read_text(encoding="utf-8")

    def test_interventions_and_states(self):
        s = _session(self.tmp)
        journal_frame(s, MonitorInput())
        journal_frame(s, MonitorInput(estop=True, stop=True, takeover=True, punish=True, steer_z=1.0))
        journal_frame(s, MonitorInput(steer_z=1.0))
        s.teacher.on = True
        s.remote = {"learning": True, "autonomy": True, "n_ppl1_total": 5}
        journal_frame(s, MonitorInput())
        t = self.text()
        for word in ("E-STOP", "стоп (пробел)", "перехват", "ошибка (X)", "учитель включён", "обучение включено", "автономия включена", "PPL1 (негатив) на борту +2, всего 5"):
            self.assertIn(word, t)
        self.assertEqual(t.count("стрелки"), 1)
        self.assertRegex(t.splitlines()[0], r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.250  ")

    def test_focus_halt_edge(self):
        s = _session(self.tmp)
        s._teacher_halt = True
        journal_frame(s, MonitorInput(focused=False))
        journal_frame(s, MonitorInput(focused=False))
        s._teacher_halt = False
        journal_frame(s, MonitorInput())
        t = self.text()
        self.assertEqual(t.count("фокус потерян"), 1)
        self.assertIn("фокус вернулся", t)

    def test_ppl1_sent(self):
        s = _session(self.tmp)
        note_ppl1_sent(s, "ppl1", None)
        note_ppl1_sent(s, "pam", "pam")
        t = self.text()
        self.assertIn("PPL1 отправлен (teach_sides) L", t)
        self.assertEqual(t.count("PPL1"), 1)


if __name__ == "__main__":
    unittest.main()