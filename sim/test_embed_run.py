"""The trainer dock has no 3D block. The J window is a separate process."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class LayoutPruneTests(unittest.TestCase):
    def test_old_brain3d_leaf_gives_its_space_to_the_journal(self):
        from sim.dock_layout import Leaf, Split, find_leaf, layout_dock, tree_from_dict

        old = {
            "t": "split",
            "o": "v",
            "r": 0.9,
            "locked": False,
            "a": {
                "t": "split",
                "o": "h",
                "r": 0.4,
                "locked": True,
                "a": {
                    "t": "split",
                    "o": "v",
                    "r": 0.72,
                    "a": {"t": "leaf", "id": "camera", "visible": True, "collapsed": False},
                    "b": {"t": "leaf", "id": "lidar", "visible": True, "collapsed": False},
                },
                "b": {
                    "t": "split",
                    "o": "v",
                    "r": 0.52,
                    "locked": False,
                    "a": {"t": "leaf", "id": "brain", "visible": True, "collapsed": False},
                    "b": {
                        "t": "split",
                        "o": "v",
                        "r": 0.55,
                        "a": {"t": "leaf", "id": "journal", "visible": True, "collapsed": False},
                        "b": {"t": "leaf", "id": "brain3d", "visible": True, "collapsed": False},
                    },
                },
            },
            "b": {"t": "leaf", "id": "controls", "visible": True, "collapsed": False},
        }
        tree = tree_from_dict(old)
        self.assertIsNone(find_leaf(tree, "brain3d"))
        self.assertIsNotNone(find_leaf(tree, "journal"))
        self.assertTrue(tree.a.locked)
        self.assertAlmostEqual(tree.a.ratio, 0.4, places=3)
        column = tree.a.b
        self.assertIsInstance(column.b, Leaf)
        self.assertEqual(column.b.block, "journal")
        self.assertAlmostEqual(column.ratio, 0.52, places=3)

        combined = Split("v", 0.52, Leaf("brain"), Split("v", 0.55, Leaf("journal"), Leaf("brain3d")))
        before = layout_dock(combined, 0, 0, 400, 800)["blocks"]
        after = layout_dock(column, 0, 0, 400, 800)["blocks"]
        old_span = before["journal"][3] + before["brain3d"][3]
        self.assertGreater(after["journal"][3], before["journal"][3] + 40)
        self.assertGreaterEqual(after["journal"][3], old_span)

    def test_brain_window_roundtrip_stays_beside_window_json(self):
        from sim.ui_settings import brain_window_path, load_brain_window, save_brain_window
        from sim.ui_theme import apply_theme

        root = Path("/tmp/recog_brain_window")
        root.mkdir(parents=True, exist_ok=True)
        previous = os.environ.get("RECOGNIZER_UI_CONFIG")
        os.environ["RECOGNIZER_UI_CONFIG"] = str(root / "window.json")
        try:
            save_brain_window({"x": 12, "y": 24, "w": 640, "h": 400, "embed_3d": True})
            loaded = load_brain_window()
            self.assertEqual(loaded, {"x": 12, "y": 24, "w": 640, "h": 400})
            from sim.mb_view3d import _saved_viewer_size

            os.environ.pop("SDL_VIDEO_WINDOW_POS", None)
            self.assertEqual(_saved_viewer_size(), (640, 400))
            self.assertEqual(os.environ.get("SDL_VIDEO_WINDOW_POS"), "12,24")
            save_brain_window({"x": None, "y": None, "w": 100, "h": 10})
            self.assertEqual(_saved_viewer_size(), (320, 240))
            path = brain_window_path()
            self.assertEqual(path.name, "brain_window.json")
            self.assertNotIn("_internal", path.parts)
            self.assertNotEqual(path.suffix, ".npz")
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("embed_3d", text)
            self.assertNotIn("brain3d", text)
        finally:
            if previous is None:
                os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            else:
                os.environ["RECOGNIZER_UI_CONFIG"] = previous
            apply_theme("light")


class CopyTests(unittest.TestCase):
    def test_compact_ids_does_not_keep_the_source(self):
        import numpy as np

        from sim.mb_view3d import compact_ids

        src = np.array([0.0, 2.0, 0.0, 1.0], dtype=np.float32)
        out = compact_ids(src)
        src[:] = 0
        self.assertEqual(out, [1, 3])


class SeparateWindowTests(unittest.TestCase):
    def test_trainer_does_not_open_the_3d_window(self):
        import pygame

        from sim.recognize_train import RecognizeTrainSim, run_gui
        from sim.ui_theme import apply_theme

        previous = os.environ.get("GUI_STRESS")
        os.environ.pop("GUI_STRESS", None)
        state = Path("/tmp/recog_no_j.npz")
        if state.exists():
            state.unlink()
        pygame.init()
        try:
            session = RecognizeTrainSim(n_agents=2, seed=1, state_path=state, learner="mb")
            summary = run_gui(session, seconds=0.4)
            self.assertIsInstance(summary, dict)
            link = getattr(session, "_brain_view", None)
            self.assertTrue(link is None or link.proc is None)
        finally:
            if previous is None:
                os.environ.pop("GUI_STRESS", None)
            else:
                os.environ["GUI_STRESS"] = previous
            apply_theme("light")
            pygame.quit()
            if state.exists():
                state.unlink()

    def test_stress_resizes_the_dock_and_opens_j_without_gl_in_the_trainer(self):
        import pygame

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
        pygame.init()
        try:
            session = RecognizeTrainSim(n_agents=2, seed=1, state_path=state, learner="mb")
            summary = run_gui(session, seconds=12.0)
            self.assertIsInstance(summary, dict)
            self.assertGreaterEqual(float(session.world.t), 12.0)
            self.assertGreaterEqual(int(getattr(session, "_stress_n", 0)), 3)
            link = getattr(session, "_brain_view", None)
            self.assertIsNotNone(link)
            self.assertTrue(getattr(session, "_stress_brain", False))
            self.assertIsInstance(getattr(session, "_stress_brain_pid", None), int)
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
