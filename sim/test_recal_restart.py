"""«↻ перекалибровать» and «ПЕРЕЗАПУСК МОЗГА», sim, monitor, and onboard loop.

Run: python -m unittest sim.test_recal_restart
Screenshots (light and dark, 1024 wide) go to MyApps/testlogs/.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "testlogs"


def weights_hash(mb) -> str:
    h = hashlib.sha256()
    for arr in (mb.brain.kc_mbon_w, mb.brain.kc_fam, mb.brain_r.kc_mbon_w, mb.brain_r.kc_fam):
        h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).tobytes())
    return h.hexdigest()


def counters(mb) -> tuple:
    p = mb.progress
    return (
        int(mb.n_pam),
        int(mb.n_ppl1),
        int(mb.n_novelty),
        int(p.base_pam),
        int(p.base_ppl1),
        int(mb.teacher_pam_l),
        int(mb.teacher_pam_r),
        int(mb.teacher_ppl1_l),
        int(mb.teacher_ppl1_r),
    )


class CalibratorTests(unittest.TestCase):
    def test_progress_and_reset(self):
        from sim.mb_confidence import WINDOW, ConfidenceCalibrator

        cal = ConfidenceCalibrator()
        self.assertEqual(cal.progress(), 0)
        self.assertTrue(cal.calibrating)
        for i in range(WINDOW + 30):
            cal.update(float(i % 7), float(i % 5))
        self.assertEqual(cal.progress(), WINDOW)
        self.assertFalse(cal.calibrating)
        cal.reset()
        self.assertEqual(cal.progress(), 0)
        self.assertTrue(cal.calibrating)
        self.assertFalse(cal.recognized)


class SimRecalRestartTests(unittest.TestCase):
    def setUp(self):
        from sim.recognize_train import RecognizeTrainSim

        self._tmp = tempfile.TemporaryDirectory()
        self.state = Path(self._tmp.name) / "mb_state.npz"
        self.s = RecognizeTrainSim(state_path=self.state, seed=3)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, n, learn=False, teach_every=0):
        for i in range(n):
            teach = "pam" if teach_every and i % teach_every == 0 else None
            self.s.step(1.0, 0.0, learn=learn, teach=teach)

    def test_recalibrate_keeps_weights_and_counters(self):
        self.s.learn = True
        self._run(240, learn=True, teach_every=15)
        mb = self.s.mb
        self.assertEqual(mb.calib_progress(), (200, 200))
        self.assertGreater(mb.n_pam, 0)
        before_w = weights_hash(mb)
        before_c = counters(mb)
        self.s.learn = False
        self.s.recalibrate()
        self.assertEqual(mb.calib_progress(), (0, 200))
        self.assertEqual(weights_hash(mb), before_w)
        self.assertEqual(counters(mb), before_c)
        self.assertFalse(self.s.learn)
        view = self.s.view(True, "", "", "")
        self.assertEqual((view.calib_n, view.calib_total), (0, 200))
        self._run(120)
        self.assertEqual(mb.calib_progress()[0], 120)
        self.assertLess(self.s.view(True, "", "", "").calib_n, 200)
        self._run(100)
        self.assertEqual(mb.calib_progress(), (200, 200))
        self.assertEqual(weights_hash(mb), before_w)
        self.assertEqual(counters(mb)[:3], before_c[:3])

    def test_restart_turns_learning_and_autonomy_off(self):
        self.s.learn = True
        self._run(60, learn=True, teach_every=10)
        old = self.s.mb
        w = weights_hash(old)
        total_pam = old.progress.base_pam + old.n_pam
        self.assertTrue(self.s.pilot.start_auto(float(self.s.world.t)))
        self.assertTrue(self.s.pilot.autonomy)
        self.assertTrue(self.s.restart_brain())
        self.assertIsNot(self.s.mb, old)
        self.assertFalse(self.s.learn)
        self.assertFalse(self.s.pilot.autonomy)
        self.assertEqual(weights_hash(self.s.mb), w)
        self.assertEqual(self.s.mb.progress.base_pam + self.s.mb.n_pam, total_pam)
        self.assertTrue(self.state.is_file())
        self.assertFalse(Path(str(self.state) + ".tmp").exists())

    def test_second_request_while_restarting_is_ignored(self):
        old = self.s.mb
        self.s._restarting = True
        self.assertFalse(self.s.restart_brain())
        self.assertIs(self.s.mb, old)
        self.s._restarting = False


class MonitorButtonTests(unittest.TestCase):
    def setUp(self):
        fd, self._cfg = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.environ["RECOGNIZER_UI_CONFIG"] = self._cfg

    def tearDown(self):
        os.environ.pop("RECOGNIZER_UI_CONFIG", None)
        from sim.ui_theme import apply_theme

        apply_theme("light")
        try:
            os.remove(self._cfg)
        except OSError:
            pass

    def _click(self, mon, view, rect):
        import pygame

        from sim.train_monitor import window_from_logical

        size = mon.screen.get_size() if mon.screen is not None else (1920, 1080)
        wx, wy = window_from_logical(rect.center, 1920, 1080)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(wx), int(wy)), button=1))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(int(wx), int(wy)), button=1))
        inp = mon.pump()
        mon.draw(view)
        return inp

    def test_buttons_confirm_and_screens(self):
        import pygame

        from sim.train_monitor import MonitorView, TrainMonitor

        pygame.init()
        try:
            mon = TrainMonitor("recal")
            mon._on_resize(1920, 1080)
            view = MonitorView(
                learner="mb",
                camera=np.full((90, 160, 3), 90, dtype=np.uint8),
                lidar=np.zeros((64, 64, 3), dtype=np.uint8),
                log_lines=["перекалибровка «УЗНАЮ»: 0/200, веса и счётчики не тронуты"],
                calib_n=37,
                calib_total=200,
                eye_l_ready=True,
                eye_r_ready=True,
                eye_l_recognized=True,
            )
            mon.draw(view)
            self.assertTrue(mon._calibrating(view))
            self.assertGreater(mon.brain_restart_rect.w, 8)
            self.assertGreater(mon.recal_rect.w, 4)
            self.assertGreater(mon.brain_restart_rect.x, mon.lidar_reset_rect.right)

            inp = self._click(mon, view, mon.recal_rect)
            self.assertTrue(inp.recalibrate)
            self.assertFalse(inp.brain_restart)

            first = self._click(mon, view, mon.brain_restart_rect)
            self.assertFalse(first.brain_restart)
            self.assertTrue(mon.restart_armed())
            SHOTS.mkdir(parents=True, exist_ok=True)
            pygame.image.save(pygame.transform.smoothscale(mon.screen, (1024, 576)), str(SHOTS / "recal_restart_confirm_light_1024.png"))
            second = self._click(mon, view, mon.brain_restart_rect)
            self.assertTrue(second.brain_restart)
            self.assertFalse(mon.restart_armed())
            third = self._click(mon, view, mon.brain_restart_rect)
            self.assertFalse(third.brain_restart)
            mon._restart_armed_at = -10.0
            mon.draw(view)
            pygame.image.save(pygame.transform.smoothscale(mon.screen, (1024, 576)), str(SHOTS / "recal_restart_light_1024.png"))
            mon._set_theme("dark")
            mon.draw(view)
            pygame.image.save(pygame.transform.smoothscale(mon.screen, (1024, 576)), str(SHOTS / "recal_restart_dark_1024.png"))
            mon._restart_armed_at = __import__("time").monotonic()
            mon.draw(view)
            pygame.image.save(pygame.transform.smoothscale(mon.screen, (1024, 576)), str(SHOTS / "recal_restart_confirm_dark_1024.png"))
            view.calib_n = 200
            mon.draw(view)
            self.assertFalse(mon._calibrating(view))
        finally:
            pygame.quit()


class OnboardLoopTests(unittest.TestCase):
    def setUp(self):
        here = ROOT / "robot" / "flybrain_onboard"
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        from app import make_brain
        from drive import SportDrive

        self._tmp = tempfile.TemporaryDirectory()
        self.state = Path(self._tmp.name) / "mb_train_state.npz"
        self.loop = make_brain(SportDrive(None), str(self.state))

    def tearDown(self):
        self._tmp.cleanup()

    def test_recalibrate_command(self):
        import time

        mb = self.loop.mb
        for i in range(30):
            mb.cal.update(float(i), float(i % 3))
        w = weights_hash(mb)
        c = counters(mb)
        self.loop.handle({"op": "recalibrate"}, time.monotonic())
        st = self.loop.status()
        self.assertEqual((st["calib_n"], st["calib_total"]), (0, 200))
        # Applied on the next frame by ingest, here directly as that step does.
        self.assertTrue(self.loop._recal_pending)
        self.loop._recal_pending = False
        mb.recalibrate()
        self.assertEqual(mb.calib_progress(), (0, 200))
        self.assertEqual(weights_hash(mb), w)
        self.assertEqual(counters(mb), c)
        self.assertFalse(st["learning"])

    def test_restart_is_idempotent_and_exits_nonzero(self):
        import time

        from app import RESTART_EXIT_CODE

        now = time.monotonic()
        self.loop.handle({"op": "learn_on"}, now)
        self.loop.handle({"op": "restart"}, now)
        self.loop.handle({"op": "restart"}, now)
        st = self.loop.status()
        self.assertTrue(st["restarting"])
        self.assertFalse(st["learning"])
        self.assertFalse(st["autonomy"])
        skipped = [line for line in self.loop.log if "повтор пропущен" in str(line)]
        self.assertEqual(len(skipped), 1)
        cmd = self.loop.tick(now)
        self.assertTrue(cmd.stop)
        self.assertEqual((cmd.x, cmd.z), (0.0, 0.0))
        code = self.loop.finish_restart()
        self.assertEqual(code, RESTART_EXIT_CODE)
        self.assertNotEqual(code, 0)
        self.assertTrue(self.state.is_file())
        self.assertFalse(Path(str(self.state) + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
