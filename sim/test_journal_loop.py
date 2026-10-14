"""The real sim loop (_run_gui) writes journal.log lines while it runs."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class SimLoopJournalTest(unittest.TestCase):
    def test_events_reach_journal_before_exit(self):
        import pygame

        from sim import trainer_journal
        from sim.frame_record import FrameRecorder, OperatorMarks
        from sim.recognize_train import RecognizeTrainSim, _run_gui
        from sim.train_monitor import TrainMonitor

        tmp = Path(tempfile.mkdtemp())
        jpath = tmp / "journal.log"
        orig_path = trainer_journal.journal_path
        trainer_journal.journal_path = lambda: jpath
        pygame.init()
        seen = {}
        try:
            session = RecognizeTrainSim(n_agents=2, seed=1, state_path=tmp / "s.npz", learner="mb")
            session.teacher = None
            mon = TrainMonitor("journal test")
            real_pump = mon.pump
            frame = {"n": 0}

            def pump():
                inp = real_pump()
                n = frame["n"] = frame["n"] + 1
                if n == 3:
                    inp.estop = True
                inp.focused = n not in (5, 6, 7)
                if n == 9:
                    inp.punish = True
                if n == 40:
                    seen["mid"] = jpath.read_text(encoding="utf-8") if jpath.exists() else ""
                if n >= 41:
                    inp.quit = True
                return inp

            mon.pump = pump
            rec = FrameRecorder(tmp / "frames", fps=2.0)
            try:
                _run_gui(session, mon, None, 0.0, rec, OperatorMarks())
            finally:
                rec.close()
        finally:
            trainer_journal.journal_path = orig_path
        mid = seen.get("mid", "")
        self.assertIn("тренер запущен (сим)", mid)
        self.assertIn("E-STOP", mid)
        self.assertIn("фокус потерян", mid)
        self.assertIn("фокус вернулся", mid)
        self.assertIn("ошибка", mid.lower())
        self.assertIn("УЗНАЮ", mid)
        self.assertIn("· ", mid)  # session._log mirrored
        self.assertRegex(mid, r"\d\d:\d\d:\d\d\.\d{3}")


if __name__ == "__main__":
    unittest.main()
