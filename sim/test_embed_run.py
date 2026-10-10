"""Embedded 3D stays out of the trainer process, and a crashed run stays off."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class LatchTests(unittest.TestCase):
    def test_unclean_exit_keeps_embed_off(self):
        from sim.ui_settings import begin_gui_session, default_settings, end_gui_session, load_settings, save_settings
        from sim.ui_theme import apply_theme

        root = Path("/tmp/recog_latch_test")
        root.mkdir(parents=True, exist_ok=True)
        previous = os.environ.get("RECOGNIZER_UI_CONFIG")
        os.environ["RECOGNIZER_UI_CONFIG"] = str(root / "window.json")
        try:
            clean = default_settings()
            clean["unclean_exit"] = False
            clean["embed_3d"] = True
            save_settings(clean)
            begin_gui_session()
            armed = load_settings()
            self.assertTrue(armed["embed_3d"])
            self.assertTrue(armed["unclean_exit"])
            end_gui_session()
            self.assertFalse(load_settings()["unclean_exit"])
            self.assertTrue(load_settings()["embed_3d"])

            crashed = default_settings()
            crashed["unclean_exit"] = True
            crashed["embed_3d"] = True
            save_settings(crashed)
            begin_gui_session()
            forced = load_settings()
            self.assertFalse(forced["embed_3d"])
            self.assertTrue(forced["unclean_exit"])
            end_gui_session()
            done = load_settings()
            self.assertFalse(done["unclean_exit"])
            self.assertFalse(done["embed_3d"])
        finally:
            if previous is None:
                os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            else:
                os.environ["RECOGNIZER_UI_CONFIG"] = previous
            apply_theme("light")

    def test_view_switch_stops_embed_immediately(self):
        import pygame

        from sim.train_monitor import TrainMonitor
        from sim.ui_theme import apply_theme

        pygame.init()
        try:
            mon = TrainMonitor("embed-switch")
            mon.use_embed_process = True
            self.assertTrue(mon.embed_3d)
            mon._run_menu(("embed",))
            self.assertFalse(mon.embed_3d)
            self.assertIsNone(mon._embed)
            mon._run_menu(("embed",))
            self.assertTrue(mon.embed_3d)
        finally:
            apply_theme("light")
            pygame.quit()


class CopyTests(unittest.TestCase):
    def test_seqlock_hands_back_an_owned_copy(self):
        from multiprocessing import shared_memory

        import numpy as np

        from sim.embed_brain import SHM_SIZE, _publish, _read_frame

        shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE)
        try:
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[10, 20] = (9, 8, 7)
            seq = _publish(shm, image, 0)
            self.assertEqual(seq % 2, 0)
            got = _read_frame(shm)
            self.assertIsNotNone(got)
            _seq, frame = got
            self.assertEqual(tuple(int(v) for v in frame[10, 20]), (9, 8, 7))
            frame[10, 20] = 0
            again = _read_frame(shm)
            self.assertEqual(tuple(int(v) for v in again[1][10, 20]), (9, 8, 7))
        finally:
            shm.close()
            shm.unlink()

    def test_compact_ids_does_not_keep_the_source(self):
        import numpy as np

        from sim.mb_view3d import compact_ids

        src = np.array([0.0, 2.0, 0.0, 1.0], dtype=np.float32)
        out = compact_ids(src)
        src[:] = 0
        self.assertEqual(out, [1, 3])


class LongGuiTests(unittest.TestCase):
    def test_sim_gui_survives_resize_and_zoom(self):
        import pygame

        import sim.embed_brain as embed
        import sim.mb_flywire as fly
        from sim.recognize_train import RecognizeTrainSim, run_gui
        from sim.ui_theme import apply_theme

        previous = os.environ.get("GUI_STRESS")
        os.environ["GUI_STRESS"] = "1"
        state = Path("/tmp/recog_embed_gui.npz")
        if state.exists():
            state.unlink()
        before = None if fly._GL is None else (id(fly._GL), int(fly._GL.ibo_writes))
        paint_before = fly._LAST_PAINT
        starts = int(embed.STARTS)
        pygame.init()
        try:
            session = RecognizeTrainSim(n_agents=2, seed=1, state_path=state, learner="mb")
            summary = run_gui(session, seconds=12.0)
            self.assertIsInstance(summary, dict)
            self.assertGreaterEqual(float(session.world.t), 12.0)
            self.assertGreaterEqual(int(getattr(session, "_stress_n", 0)), 3)
            self.assertGreater(int(embed.STARTS), starts)
            after = None if fly._GL is None else (id(fly._GL), int(fly._GL.ibo_writes))
            self.assertEqual(after, before)
            self.assertEqual(fly._LAST_PAINT, paint_before)
        finally:
            if previous is None:
                os.environ.pop("GUI_STRESS", None)
            else:
                os.environ["GUI_STRESS"] = previous
            apply_theme("light")
            pygame.quit()
            if state.exists():
                state.unlink()


if __name__ == "__main__":
    unittest.main()
