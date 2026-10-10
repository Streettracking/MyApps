"""Dock, theme, and the View menu. The 3D brain is not a dock block.

Run: SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -m unittest sim.test_dock_ui
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class DockGeometryTests(unittest.TestCase):
    def test_camera_stays_put_while_the_column_grows(self):
        from sim.dock_layout import default_tree, find_leaf, layout_dock, swap_blocks
        from sim.train_monitor import FOOTER_H, MENU_H, WIN_W

        def place(height):
            area_h = height - MENU_H - FOOTER_H
            return layout_dock(default_tree(), 12, MENU_H, WIN_W - 24, area_h)["blocks"]

        short = place(1080)
        tall = place(2133)
        self.assertEqual(short["camera"][2:], tall["camera"][2:])
        self.assertGreater(tall["lidar"][3], short["lidar"][3] + 80)
        self.assertGreater(tall["brain"][3], 900)
        self.assertGreater(tall["journal"][3], short["journal"][3] + 40)
        cam_w, cam_h = short["camera"][2], short["camera"][3]
        self.assertAlmostEqual((cam_h - 26) / float(cam_w), 9 / 16, delta=0.04)

        tree = default_tree()
        find_leaf(tree, "journal").collapsed = True
        folded = layout_dock(tree, 12, MENU_H, WIN_W - 24, 1080 - MENU_H - FOOTER_H)["blocks"]
        self.assertLess(folded["journal"][3], 40)
        find_leaf(tree, "lidar").visible = False
        hidden = layout_dock(tree, 12, MENU_H, WIN_W - 24, 1080 - MENU_H - FOOTER_H)["blocks"]
        self.assertNotIn("lidar", hidden)
        self.assertIn("camera", hidden)
        self.assertTrue(swap_blocks(tree, "camera", "brain"))
        self.assertEqual(find_leaf(tree, "camera").block, "camera")

    def test_settings_roundtrip_stays_outside_the_npz(self):
        from sim.dock_layout import default_tree, tree_from_dict
        from sim.ui_settings import load_settings, save_settings, settings_path

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.environ["RECOGNIZER_UI_CONFIG"] = path
        try:
            tree = default_tree()
            tree.a.ratio = 0.4
            tree.a.locked = True
            save_settings(
                {
                    "theme": "dark",
                    "maximized": False,
                    "fullscreen": False,
                    "x": 12,
                    "y": 24,
                    "w": 1600,
                    "h": 900,
                    "layout": tree.to_dict(),
                }
            )
            loaded = load_settings()
            self.assertEqual(loaded["theme"], "dark")
            self.assertEqual(loaded["w"], 1600)
            restored = tree_from_dict(loaded["layout"])
            self.assertTrue(restored.a.locked)
            self.assertAlmostEqual(restored.a.ratio, 0.4, places=3)
            text = settings_path().read_text(encoding="utf-8")
            self.assertNotIn("_internal", str(settings_path()))
            self.assertNotIn("npz", str(settings_path()))
            self.assertIn("dark", text)
        finally:
            os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            os.remove(path)


class TrainerDockTests(unittest.TestCase):
    def setUp(self):
        self._fd, self._path = tempfile.mkstemp(suffix=".json")
        os.close(self._fd)
        os.environ["RECOGNIZER_UI_CONFIG"] = self._path

    def tearDown(self):
        os.environ.pop("RECOGNIZER_UI_CONFIG", None)
        from sim.ui_theme import apply_theme

        apply_theme("light")
        try:
            os.remove(self._path)
        except OSError:
            pass

    def test_menu_theme_collapse(self):
        import pygame
        import numpy as np

        from sim.dock_layout import BLOCKS, BLOCK_TITLES, find_leaf, swap_blocks
        from sim.train_monitor import SCROLL_BAR, MonitorView, TrainMonitor
        from sim.ui_theme import theme_name

        self.assertEqual(SCROLL_BAR, 8)
        self.assertNotIn("brain3d", BLOCKS)
        self.assertNotIn("мозг 3D", BLOCK_TITLES.values())
        pygame.init()
        try:
            mon = TrainMonitor("dock")
            mon._on_resize(1920, 1080)
            view = MonitorView(
                title="recognition training",
                learner="mb",
                learning_on=True,
                camera=np.zeros((90, 160, 3), dtype=np.uint8),
                lidar=np.zeros((64, 64, 3), dtype=np.uint8),
                log_lines=["строка журнала"],
                keys_hint="A авто  J мозг",
                eye_l_recognized=True,
            )
            self.assertIsNone(find_leaf(mon.dock, "brain3d"))
            self.assertIsNotNone(find_leaf(mon.dock, "journal"))
            mon.draw(view)
            self.assertIsNone(mon._content_of("brain3d"))
            self.assertIsNotNone(mon._content_of("journal"))
            self.assertGreater(mon.journal_rect.h, 80)
            out = __import__("pathlib").Path("/opt/cursor/artifacts")
            out.mkdir(parents=True, exist_ok=True)
            pygame.image.save(mon.present_into((1920, 1080)), str(out / "trainer_dock_journal.png"))

            find_leaf(mon.dock, "journal").collapsed = True
            mon._apply_layout(mon.screen.get_height())
            mon.menu_open = True
            mon.draw(view)
            self.assertLess(mon.journal_rect.h, 40)
            self.assertTrue(mon.menu_open)
            self.assertGreater(len(mon._menu_hits), 6)
            labels = []
            for _rect, action in mon._menu_hits:
                self.assertNotEqual(action, ("embed",))
                if action[0] == "toggle":
                    labels.append(action[1])
            self.assertNotIn("brain3d", labels)
            self.assertIn("journal", labels)
            light = pygame.surfarray.array3d(mon.present_into((1920, 1080)))
            out = __import__("pathlib").Path("/opt/cursor/artifacts")
            out.mkdir(parents=True, exist_ok=True)
            pygame.image.save(mon.present_into((1920, 1080)), str(out / "trainer_light_dock.png"))
            self.assertGreater(int(light[:, :40].max(axis=2).mean()), 180)

            toggle = None
            for rect, action in mon._menu_hits:
                if action[0] == "toggle" and action[1] == "journal":
                    toggle = rect.center
            self.assertIsNotNone(toggle)
            from sim.train_monitor import window_from_logical

            wx, wy = window_from_logical(toggle, 1920, 1080)
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(wx), int(wy)), button=1))
            mon.menu_open = True
            mon.draw(view)
            clicked = mon.pump()
            self.assertFalse(clicked.brain_toggle)
            self.assertFalse(find_leaf(mon.dock, "journal").visible)
            find_leaf(mon.dock, "journal").visible = True

            mon._set_theme("dark")
            self.assertEqual(theme_name(), "dark")
            swap_blocks(mon.dock, "camera", "lidar")
            find_leaf(mon.dock, "journal").collapsed = False
            mon._apply_layout(mon.screen.get_height())
            mon.menu_open = True
            mon.draw(view)
            self.assertLess(mon.lid_rect.y, mon.cam_rect.y)
            dark = pygame.surfarray.array3d(mon.present_into((1920, 1080)))
            pygame.image.save(mon.present_into((1920, 1080)), str(out / "trainer_dark_dock.png"))
            self.assertLess(int(dark[:, :40].max(axis=2).mean()), 80)
            self.assertLess(int(dark[:, :40].mean()), int(light[:, :40].mean()) - 40)

            reset = None
            for rect, action in mon._menu_hits:
                if action[0] == "reset":
                    reset = rect.center
            wx, wy = window_from_logical(reset, 1920, 1080)
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(wx, wy), button=1))
            mon.pump()
            self.assertLess(mon.cam_rect.y, mon.lid_rect.y)
            self.assertEqual(theme_name(), "dark")
            mon._set_theme("light")
            self.assertEqual(theme_name(), "light")
        finally:
            pygame.quit()
