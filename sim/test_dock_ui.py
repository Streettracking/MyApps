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

    def test_saved_dock_gains_a_robot_card_beside_controls(self):
        import pygame

        from sim.dock_layout import tree_from_dict
        from sim.train_monitor import MonitorView, TrainMonitor

        old = {
            "t": "split",
            "o": "v",
            "r": 0.9,
            "a": {
                "t": "split",
                "o": "h",
                "r": 0.4,
                "locked": True,
                "a": {
                    "t": "split",
                    "o": "v",
                    "r": 0.72,
                    "a": {"t": "leaf", "id": "camera", "visible": True},
                    "b": {"t": "leaf", "id": "lidar", "visible": True},
                },
                "b": {
                    "t": "split",
                    "o": "v",
                    "r": 0.52,
                    "a": {"t": "leaf", "id": "brain", "visible": True},
                    "b": {"t": "leaf", "id": "journal", "visible": True},
                },
            },
            "b": {"t": "leaf", "id": "controls", "visible": True},
        }
        tree = tree_from_dict(old)
        self.assertTrue(tree.a.locked)
        self.assertAlmostEqual(tree.a.ratio, 0.4, places=3)
        self.assertEqual(tree.b.a.block, "controls")
        self.assertEqual(tree.b.b.block, "robot")
        pygame.init()
        mon = TrainMonitor("robot-card")
        try:
            mon._on_resize(1920, 1080)
            mon.draw(MonitorView(learner="mb", onboard="192.168.35.213"))
            row = (mon.stand_up_rect, mon.stand_down_rect, mon.recovery_rect, mon.lidar_reset_rect)
            self.assertGreater(min(rect.w for rect in row), 8)
            self.assertEqual({rect.y for rect in row}, {row[0].y})
            content = mon._content_of("robot")
            self.assertIsNotNone(content)
            host = pygame.Rect(content)
            for rect in row:
                self.assertTrue(host.contains(rect), rect)
            self.assertFalse(mon.lidar_column_rect.contains(mon.lidar_reset_rect))
        finally:
            pygame.quit()

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
            self.assertTrue(mon.menu_open)
            themes = [action for _rect, action in mon._menu_hits if action[0] == "theme"]
            self.assertEqual(themes, [("theme",)])
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
            self.assertFalse(mon.menu_open)
            self.assertLess(mon.cam_rect.y, mon.lid_rect.y)
            self.assertEqual(theme_name(), "dark")
            mon._set_theme("light")
            self.assertEqual(theme_name(), "light")
        finally:
            pygame.quit()

    def test_view_menu_checks_stay_open_and_the_theme_toggle_saves(self):
        import json

        import pygame
        import numpy as np

        from sim.dock_layout import find_leaf
        from sim.train_monitor import MonitorView, TrainMonitor, window_from_logical
        from sim.ui_theme import theme_name

        pygame.init()
        try:
            mon = TrainMonitor("menu")
            mon._on_resize(1920, 1080)
            view = MonitorView(
                title="recognition training",
                learner="mb",
                learning_on=True,
                camera=np.zeros((90, 160, 3), dtype=np.uint8),
                lidar=np.zeros((64, 64, 3), dtype=np.uint8),
                log_lines=["строка журнала"],
                keys_hint="A авто  J мозг",
            )
            find_leaf(mon.dock, "journal").visible = False
            mon._apply_layout(mon.screen.get_height())
            mon.menu_open = True
            mon._set_theme("light")
            mon.draw(view)
            self._assert_menu_checks(mon, on_theme="light")
            self._save_menu(mon, "/opt/cursor/artifacts/view_menu_light.png")

            journal = self._click_menu(mon, ("toggle", "journal"))
            self.assertFalse(journal.quit)
            self.assertTrue(mon.menu_open)
            self.assertTrue(find_leaf(mon.dock, "journal").visible)
            mon.draw(view)
            self._assert_menu_checks(mon, on_theme="light")

            themed = self._click_menu(mon, ("theme",))
            self.assertFalse(themed.quit)
            self.assertTrue(mon.menu_open)
            self.assertEqual(theme_name(), "dark")
            saved = json.loads(__import__("pathlib").Path(self._path).read_text(encoding="utf-8"))
            self.assertEqual(saved["theme"], "dark")
            mon.draw(view)
            self._assert_menu_checks(mon, on_theme="dark")
            self._save_menu(mon, "/opt/cursor/artifacts/view_menu_dark.png")

            outside = mon.cam_rect.center
            self.assertIsNone(mon._menu_at(outside))
            self.assertFalse(mon._menu_panel.collidepoint(outside))
            self._click_at(mon, outside)
            self.assertFalse(mon.menu_open)

            mon.menu_open = True
            mon.draw(view)
            self._click_at(mon, mon.view_rect.center)
            self.assertFalse(mon.menu_open)

            mon.menu_open = True
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            escaped = mon.pump()
            self.assertFalse(escaped.quit)
            self.assertFalse(mon.menu_open)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            quit_key = mon.pump()
            self.assertTrue(quit_key.quit)

            mon.menu_open = True
            mon.draw(view)
            self._click_menu(mon, ("reset",))
            self.assertFalse(mon.menu_open)
            self.assertTrue(find_leaf(mon.dock, "journal").visible)
            self.assertLess(mon.cam_rect.y, mon.lid_rect.y)
        finally:
            pygame.quit()

    def _click_at(self, mon, logical) -> None:
        import pygame

        from sim.train_monitor import window_from_logical

        wx, wy = window_from_logical(logical, 1920, 1080)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(wx), int(wy)), button=1))
        return mon.pump()

    def _click_menu(self, mon, action):
        rect = next(hit for hit, item in mon._menu_hits if item == action)
        return self._click_at(mon, rect.center)

    def _menu_box(self, rect):
        import pygame

        return pygame.Rect(rect.x + 8, rect.centery - 7, 14, 14)

    def _assert_menu_checks(self, mon, on_theme: str) -> None:
        import pygame

        from sim.dock_layout import find_leaf
        from sim.ui_theme import theme_name

        self.assertEqual(theme_name(), on_theme)
        frame = pygame.surfarray.array3d(mon.screen)
        box_x = None
        toggles = [(rect, action) for rect, action in mon._menu_hits if action[0] in ("toggle", "theme")]
        self.assertEqual(len([action for _r, action in toggles if action[0] == "theme"]), 1)
        self.assertNotIn(("theme", "light"), [action for _r, action in mon._menu_hits])
        self.assertNotIn(("theme", "dark"), [action for _r, action in mon._menu_hits])
        for rect, action in toggles:
            box = self._menu_box(rect)
            if box_x is None:
                box_x = box.x
            self.assertEqual(box.x, box_x)
            interior = frame[box.x + 3 : box.x + 11, box.y + 3 : box.y + 11]
            if action[0] == "theme":
                on = on_theme == "dark"
            else:
                on = bool(find_leaf(mon.dock, action[1]).visible)
            if on:
                self.assertGreater(int(interior.max()) - int(interior.min()), 80)
            else:
                self.assertLess(int(interior.max()) - int(interior.min()), 8)
        reset = next(rect for rect, action in mon._menu_hits if action[0] == "reset")
        dog = next(rect for rect, action in mon._menu_hits if action[0] == "doggibot")
        stroke = frame[box_x, toggles[0][0].centery]
        if on_theme == "light":
            self.assertLess(int(stroke.max()), 90)
            for row in (reset, dog):
                self.assertGreater(int(frame[box_x, row.centery].min()), 180)
        else:
            self.assertGreater(int(stroke.min()), 200)
            for row in (reset, dog):
                self.assertLess(int(frame[box_x, row.centery].max()), 120)

    def _save_menu(self, mon, path: str) -> None:
        import pygame

        panel = mon._menu_panel
        left = min(panel.x, mon.view_rect.x) - 6
        title_end = mon.doggibot_rect.right + 12 + mon.font.size("recognition training")[0]
        right = max(panel.right, title_end) + 16
        crop = pygame.Rect(left, mon.view_rect.y - 4, right - left, panel.bottom - mon.view_rect.y + 18)
        crop = crop.clamp(mon.screen.get_rect())
        shot = pygame.Surface((crop.w, crop.h))
        shot.blit(mon.screen, (0, 0), area=crop)
        pygame.image.save(pygame.transform.scale(shot, (crop.w * 2, crop.h * 2)), path)


class HelpPopoverTests(unittest.TestCase):
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

    def test_pages_name_the_readouts(self):
        from sim.ui_help import PAGES, page

        self.assertEqual(PAGES, ("camera", "lidar", "brain", "journal", "controls", "robot", "bar"))
        chunks = []
        for name in PAGES:
            title, blocks = page(name)
            chunks.append(title)
            for kind, text in blocks:
                self.assertIn(kind, ("h", "p"))
                chunks.append(text)
        text = "\n".join(chunks)
        for needle in (
            "R_L",
            "R_R",
            "сырой MBON",
            "дрейф KC→MBON",
            "D−N",
            "активность KC",
            "PAM_L",
            "PAM_R",
            "PPL1_L",
            "PPL1_R",
            "ложных узнаваний без наказания",
            "перекрытие",
            "Л+П",
            "УЗНАЮ",
            "уч PAM",
            "YOLO",
            "дальн",
            "вперёд",
            "1.5",
            "билатерально",
            "секторы",
            "вектор",
            "собака в кадре",
            "нет собаки",
            "Шкала DAN",
            "РУЧНОЕ",
            "АВТОНОМИЯ",
            "ведёт",
            "борт",
            "Esc",
        ):
            self.assertIn(needle, text)

    def test_one_popover_toggles_scrolls_and_stays_inside(self):
        import pygame
        import numpy as np

        from sim.dock_layout import find_leaf
        from sim.train_monitor import MonitorView, TrainMonitor, window_from_logical
        from sim.ui_theme import apply_theme

        pygame.init()
        point = [8, 8]
        real_pos = pygame.mouse.get_pos
        pygame.mouse.get_pos = lambda: (int(point[0]), int(point[1]))
        try:
            mon = TrainMonitor("help")
            mon._on_resize(1920, 1080)
            view = MonitorView(
                title="recognition training",
                learner="mb",
                learning_on=True,
                camera=np.zeros((90, 160, 3), dtype=np.uint8),
                lidar=np.zeros((64, 64, 3), dtype=np.uint8),
                log_lines=["строка журнала"],
                keys_hint="A авто  J мозг",
            )
            mon.draw(view)
            infos = {arg: rect for rect, kind, arg in mon._chrome if kind == "info"}
            self.assertEqual(set(infos), {"camera", "lidar", "brain", "journal", "controls", "robot"})
            for name, rect in infos.items():
                collapse = next(hit for hit, kind, arg in mon._chrome if kind == "collapse" and arg == name)
                close = next(hit for hit, kind, arg in mon._chrome if kind == "close" and arg == name)
                self.assertEqual((rect.w, rect.h), (20, 20))
                self.assertLess(rect.right, collapse.x)
                self.assertLess(collapse.right, close.x)
            self.assertEqual((mon.help_bar_rect.w, mon.help_bar_rect.h), (28, 24))
            self.assertLess(mon.help_bar_rect.right, mon.brain_rect.x)

            self._click(mon, infos["camera"].center, point)
            self.assertEqual(mon.help_id, "camera")
            mon.draw(view)
            self.assertIsNotNone(mon._help_rect)
            self._click(mon, infos["camera"].center, point)
            self.assertIsNone(mon.help_id)

            self._click(mon, infos["camera"].center, point)
            self._click(mon, infos["brain"].center, point)
            self.assertEqual(mon.help_id, "brain")
            mon.draw(view)
            self.assertIsNotNone(mon._help_body)
            inside = mon._help_body.center
            self.assertTrue(mon._help_rect.collidepoint(inside))
            self.assertFalse(infos["brain"].collidepoint(inside))
            self._click(mon, inside, point)
            self.assertEqual(mon.help_id, "brain")

            outside = (24, mon.layout["footer_y"] + 8)
            self.assertFalse(mon._help_rect.collidepoint(outside))
            clicked = self._click(mon, outside, point)
            self.assertIsNone(mon.help_id)
            self.assertFalse(clicked.quit)

            self._click(mon, mon.help_bar_rect.center, point)
            self.assertEqual(mon.help_id, "bar")
            mon.menu_open = True
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            escaped = mon.pump()
            self.assertFalse(escaped.quit)
            self.assertIsNone(mon.help_id)
            self.assertTrue(mon.menu_open)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            menu_off = mon.pump()
            self.assertFalse(menu_off.quit)
            self.assertFalse(mon.menu_open)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            quit_key = mon.pump()
            self.assertTrue(quit_key.quit)

            mon._on_resize(1920, 800)
            mon.draw(view)
            brain = next(rect for rect, kind, arg in mon._chrome if kind == "info" and arg == "brain")
            self._click(mon, brain.center, point)
            mon.draw(view)
            self.assertGreater(len(mon._help_lines), 8)
            held = mon.scroll_y
            self._aim(mon, mon._help_body.center, point)
            pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
            mon.pump()
            self.assertGreater(mon.help_scroll, 0)
            self.assertEqual(mon.scroll_y, held)
            mon.draw(view)
            self.assertGreater(mon.help_scroll, 0)
            frozen = mon.help_scroll
            self._aim(mon, (8, 8), point)
            self.assertFalse(mon._help_rect.collidepoint(mon._mouse()))
            pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
            mon.pump()
            self.assertEqual(mon.help_scroll, frozen)
            self.assertGreater(mon.scroll_y, held)

            find_leaf(mon.dock, "journal").visible = False
            mon._apply_layout(mon.screen.get_height())
            mon.help_id = "journal"
            mon.draw(view)
            self.assertIsNone(mon.help_id)

            find_leaf(mon.dock, "journal").visible = True
            mon._apply_layout(mon.screen.get_height())
            out = __import__("pathlib").Path("/opt/cursor/artifacts")
            out.mkdir(parents=True, exist_ok=True)
            for theme, block in (("dark", "camera"), ("dark", "brain"), ("light", "lidar")):
                apply_theme(theme)
                mon._on_resize(1920, 1080)
                mon.draw(view)
                info = next(rect for rect, kind, arg in mon._chrome if kind == "info" and arg == block)
                mon.help_id = block
                mon.help_scroll = 0
                mon.draw(view)
                self._assert_help_ink(mon)
                pygame.image.save(mon.present_into((1920, 1080)), str(out / ("help_%s_%s.png" % (block, theme))))
                crop = mon.screen.subsurface(mon._help_rect).copy()
                pygame.image.save(crop, str(out / ("help_%s_%s_panel.png" % (block, theme))))
                self.assertFalse(info.collidepoint(mon._help_body.center))
        finally:
            pygame.mouse.get_pos = real_pos
            pygame.quit()

    def _aim(self, mon, logical, point) -> None:
        from sim.train_monitor import FOOTER_H, window_from_logical

        ww, wh = mon.window.get_size()
        canvas_h = mon.screen.get_height() if mon.screen is not None else None
        wx, wy = window_from_logical(
            logical,
            ww,
            wh,
            (mon.scroll_x, mon.scroll_y),
            footer_h=int(getattr(mon, "_footer_h", FOOTER_H)),
            canvas_h=canvas_h,
        )
        point[0], point[1] = int(wx), int(wy)

    def _click(self, mon, logical, point):
        import pygame

        self._aim(mon, logical, point)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(point[0]), int(point[1])), button=1))
        return mon.pump()

    def _assert_help_ink(self, mon) -> None:
        body = mon._help_body
        self.assertTrue(mon._help_rect.contains(body))
        self.assertGreater(len(mon._help_lines), 4)
        for index, rect in enumerate(mon._help_lines):
            self.assertTrue(body.contains(rect), rect)
            for other in mon._help_lines[index + 1 :]:
                self.assertFalse(rect.colliderect(other), (rect, other))


class ResizeGuardTests(unittest.TestCase):
    def test_trainer_does_not_call_the_suspect_apis(self):
        import re
        from pathlib import Path

        from sim.ui_guard import no_caption, no_cursor, no_present, no_snap

        root = Path(__file__).resolve().parent
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in root.glob("*.py")
            if path.name != "ui_guard.py" and not path.name.startswith("test_")
        )
        found = re.findall(
            r"pygame\.(?:mouse\.set_cursor|scrap\b|time\.set_timer|event\.set_grab|event\.set_blocked|display\.set_icon)",
            text,
        )
        self.assertEqual(found, [])
        self.assertFalse(no_cursor())
        os.environ["RECOGNIZER_NO_CURSOR"] = "1"
        try:
            self.assertTrue(no_cursor())
        finally:
            os.environ.pop("RECOGNIZER_NO_CURSOR", None)
        self.assertFalse(no_caption())
        self.assertFalse(no_snap())
        self.assertFalse(no_present())

    def test_resize_does_not_build_a_window_wrapper(self):
        import pygame

        import sim.train_monitor as monitor
        from sim.train_monitor import TrainMonitor

        handle = tempfile.NamedTemporaryFile(prefix="recognizer-ui-", suffix=".json", delete=False)
        handle.close()
        os.environ["RECOGNIZER_UI_CONFIG"] = handle.name
        os.environ.pop("RECOGNIZER_SDL_WINDOW_ON_RESIZE", None)
        os.environ.pop("RECOGNIZER_NO_SDL_WINDOW", None)
        calls = []
        real_borrow = monitor.borrowed_window
        real_throw = monitor.throwaway_window

        def spy_borrow():
            calls.append("borrow")
            return None

        def spy_throw():
            calls.append("throw")
            return None

        monitor.borrowed_window = spy_borrow
        monitor.throwaway_window = spy_throw
        pygame.init()
        try:
            mon = TrainMonitor("guard")
            mon._on_resize(1100, 700)
            mon._on_resize(1200, 720)
            mon._on_resize(1200, 720)
            self.assertEqual(calls, [])
            self.assertIs(mon.window, pygame.display.get_surface())
            os.environ["RECOGNIZER_NO_PRESENT"] = "1"
            plain = mon.present_into((400, 300))
            self.assertEqual(plain.get_size(), (400, 300))
            os.environ.pop("RECOGNIZER_NO_PRESENT", None)
            os.environ["RECOGNIZER_SDL_WINDOW_ON_RESIZE"] = "1"
            mon._save_at = -10.0
            mon._on_resize(1300, 740)
            self.assertEqual(calls, ["throw"])
        finally:
            monitor.borrowed_window = real_borrow
            monitor.throwaway_window = real_throw
            os.environ.pop("RECOGNIZER_SDL_WINDOW_ON_RESIZE", None)
            os.environ.pop("RECOGNIZER_NO_PRESENT", None)
            os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            try:
                os.remove(handle.name)
            except OSError:
                pass
            pygame.quit()


class CursorTests(unittest.TestCase):
    def test_splitters_keep_one_cursor_and_resize_does_not_set_it(self):
        import threading

        import pygame

        import sim.ui_guard as guard
        from sim.dock_layout import header_controls
        from sim.train_monitor import TrainMonitor, scrollbar_geom, window_from_logical

        os.environ.pop("RECOGNIZER_NO_CURSOR", None)
        os.environ.pop("RECOGNIZER_UI_CONFIG", None)
        os.environ.pop("RECOGNIZER_FORCE_SETMODE", None)
        calls = []
        point = [8, 8]
        real_set = pygame.mouse.set_cursor
        real_pos = pygame.mouse.get_pos
        saved_kind = guard._CURSOR_KIND

        def spy(cursor):
            calls.append(cursor)

        pygame.mouse.set_cursor = spy
        pygame.mouse.get_pos = lambda: (int(point[0]), int(point[1]))
        pygame.init()
        try:
            mon = TrainMonitor("cursor")
            self.assertEqual(calls, [])
            ids = {name: id(cursor) for name, cursor in guard._CURSORS.items()}
            self.assertEqual(set(ids), {"arrow", "we", "ns", "hand"})
            self.assertEqual(guard._CURSORS["we"].data, (pygame.SYSTEM_CURSOR_SIZEWE,))
            self.assertEqual(guard._CURSORS["ns"].data, (pygame.SYSTEM_CURSOR_SIZENS,))
            self.assertEqual(guard._CURSORS["hand"].data, (pygame.SYSTEM_CURSOR_HAND,))
            self.assertEqual(guard._CURSORS["arrow"].data, (pygame.SYSTEM_CURSOR_ARROW,))

            during = []
            real_resize = mon._on_resize

            def resize(w, h):
                before = len(calls)
                real_resize(w, h)
                during.append(len(calls) - before)

            mon._on_resize = resize
            mon._on_resize(1920, 1080)
            guard.install_cursors()
            self.assertEqual(during, [0])
            self.assertEqual({name: id(cursor) for name, cursor in guard._CURSORS.items()}, ids)
            self.assertEqual(mon.window.get_size(), (1920, 1080))

            def place(xy):
                ww, wh = mon.window.get_size()
                landed = window_from_logical(xy, ww, wh, (mon.scroll_x, mon.scroll_y))
                point[0], point[1] = landed

            def expect(kind):
                before = len(calls)
                guard._CURSOR_KIND = None
                mon._sync_cursor()
                self.assertIs(calls[-1], guard._CURSORS[kind])
                again = len(calls)
                mon._sync_cursor()
                self.assertEqual(len(calls), again)
                self.assertGreater(again, before)

            guard._CURSOR_KIND = None
            mon._sync_cursor()
            self.assertIs(calls[-1], guard._CURSORS["arrow"])
            frozen = len(calls)
            mon._sync_cursor()
            self.assertEqual(len(calls), frozen)

            vertical = next(item for item in mon._dock_splitters if item["axis"] == "x")
            grip = vertical["rect"]
            cam = mon._dock_blocks["camera"]
            place((grip[0] + grip[2] // 2, cam[1] + cam[3] // 2))
            expect("we")
            mon._split_drag = (vertical["node"], "x", vertical["bounds"])
            point[0], point[1] = 8, 8
            frozen = len(calls)
            mon._sync_cursor()
            self.assertEqual(len(calls), frozen)
            self.assertEqual(mon._cursor_kind(), "we")

            horizontal = next(item for item in mon._dock_splitters if item["axis"] == "y")
            mon._split_drag = (horizontal["node"], "y", horizontal["bounds"])
            expect("ns")
            mon._split_drag = None
            point[0], point[1] = 8, 8
            expect("arrow")

            grip = horizontal["rect"]
            place((grip[0] + max(8, grip[2] // 2), grip[1] + grip[3] // 2))
            expect("ns")

            bar = header_controls(mon._dock_blocks["journal"])["bar"]
            place((bar[0] + 20, bar[1] + bar[3] // 2))
            expect("hand")
            cam = mon._dock_blocks["camera"]
            place((cam[0] + cam[2] // 2, cam[1] + cam[3] // 2))
            expect("arrow")

            mon._scroll_drag = ("y", 0)
            expect("hand")
            mon._scroll_drag = None
            mon._block_drag = "journal"
            frozen = len(calls)
            mon._sync_cursor()
            self.assertEqual(len(calls), frozen)
            mon._block_drag = None

            mon._on_resize(1920, 700)
            self.assertEqual(during, [0, 0])
            track = scrollbar_geom(1920, 700)["track_y"]
            self.assertIsNotNone(track)
            point[0], point[1] = track.center
            expect("hand")

            os.environ["RECOGNIZER_NO_CURSOR"] = "1"
            frozen = len(calls)
            place((grip[0] + 8, grip[1] + grip[3] // 2))
            guard._CURSOR_KIND = None
            mon._sync_cursor()
            self.assertEqual(len(calls), frozen)
            os.environ.pop("RECOGNIZER_NO_CURSOR", None)

            pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=1920, h=1080, size=(1920, 1080)))
            mon.pump()
            self.assertEqual(during, [0, 0, 0])

            guard._CURSOR_KIND = None
            box = []

            def worker():
                try:
                    guard.apply_cursor("we")
                except AssertionError as exc:
                    box.append(str(exc))

            worker_thread = threading.Thread(target=worker, name="cursor-worker")
            worker_thread.start()
            worker_thread.join()
            self.assertTrue(box)
            self.assertIsNone(guard._CURSOR_KIND)
        finally:
            pygame.mouse.set_cursor = real_set
            pygame.mouse.get_pos = real_pos
            guard._CURSOR_KIND = saved_kind
            os.environ.pop("RECOGNIZER_NO_CURSOR", None)
            pygame.quit()

    def test_a_rejected_cursor_is_not_retried_every_frame(self):
        import pygame

        import sim.ui_guard as guard

        os.environ.pop("RECOGNIZER_NO_CURSOR", None)
        if not guard._CURSORS:
            pygame.init()
            pygame.display.set_mode((320, 240))
            guard.install_cursors()
        calls = []
        real_set = pygame.mouse.set_cursor
        saved_kind = guard._CURSOR_KIND

        def boom(cursor):
            calls.append(cursor)
            raise pygame.error("Error while creating system cursor")

        pygame.mouse.set_cursor = boom
        guard._CURSOR_KIND = None
        try:
            guard.apply_cursor("ns")
            guard.apply_cursor("ns")
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0], guard._CURSORS["ns"])
            self.assertEqual(guard._CURSOR_KIND, "ns")
        finally:
            pygame.mouse.set_cursor = real_set
            guard._CURSOR_KIND = saved_kind

    def test_cursors_are_born_once_after_set_mode_and_the_flag_skips_them(self):
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import os, sys
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            flag = sys.argv[1] == "off"
            if flag:
                os.environ["RECOGNIZER_NO_CURSOR"] = "1"
            else:
                os.environ.pop("RECOGNIZER_NO_CURSOR", None)
            import pygame
            pygame.init()
            order = []
            calls = []
            real_mode = pygame.display.set_mode
            def mode(*args, **kwargs):
                order.append("set_mode")
                return real_mode(*args, **kwargs)
            pygame.display.set_mode = mode
            pygame.mouse.set_cursor = lambda *args, **kwargs: calls.append(args[0])
            import sim.train_monitor as tm
            import sim.ui_guard as guard
            real_install = tm.install_cursors
            def install():
                order.append("install")
                assert pygame.display.get_surface() is not None
                if not flag:
                    assert not guard._CURSORS
                real_install()
            tm.install_cursors = install
            mon = tm.TrainMonitor("life")
            assert order[0] == "set_mode"
            assert order.count("install") == 1
            assert order[-1] == "install"
            if flag:
                assert guard._CURSORS == {}
                mon._sync_cursor()
                assert calls == []
            else:
                assert set(guard._CURSORS) == {"arrow", "we", "ns", "hand"}
                ids = {name: id(cursor) for name, cursor in guard._CURSORS.items()}
                assert calls == []
                grown = []
                real_resize = mon._on_resize
                def resize(w, h):
                    before = len(calls)
                    real_resize(w, h)
                    grown.append(len(calls) - before)
                mon._on_resize = resize
                mon._on_resize(1920, 1080)
                real_install()
                assert grown == [0]
                assert {name: id(cursor) for name, cursor in guard._CURSORS.items()} == ids
                assert order.count("install") == 1
            print("ok")
            """
        )
        for mode in ("on", "off"):
            proc = subprocess.run(
                [sys.executable, "-X", "dev", "-c", script, mode],
                cwd="/workspace",
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
            self.assertIn("ok", proc.stdout)


class StatusColorTests(unittest.TestCase):
    def test_warnings_are_amber_and_peach_is_only_on(self):
        from sim.train_monitor import AMBER, GREEN, LAMP_OFF, PEACH, RED, MonitorView, TrainMonitor

        self.assertNotEqual(AMBER, PEACH)
        self.assertNotEqual(AMBER, RED)
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        pygame.init()
        mon = TrainMonitor("colors")
        try:
            self.assertEqual(mon._eye_lamp_color(True, True, 90), GREEN)
            self.assertEqual(mon._eye_lamp_color(False, True, 40), AMBER)
            self.assertEqual(mon._eye_lamp_color(False, False, 0), LAMP_OFF)
            self.assertEqual(mon._link_color(MonitorView(udp_status="udp 127.0.0.1:5451 ok")), PEACH)
            self.assertEqual(mon._link_color(MonitorView(udp_status="нет связи")), AMBER)
            self.assertEqual(mon._link_color(MonitorView(udp_status="link error")), RED)
            self.assertEqual(mon._link_color(MonitorView(onboard="192.168.35.213", udp_status="")), AMBER)
        finally:
            pygame.quit()


class PaletteTests(unittest.TestCase):
    """Card shadow and lamp rims come from ui_theme. Drawing keeps almost no raw RGB."""

    def test_drawing_has_no_literal_rgb_outside_the_palette(self):
        import re
        from pathlib import Path

        from sim.ui_theme import DARK, LIGHT

        themed = (
            "shadow",
            "peach_soft",
            "amber_soft",
            "teacher_soft",
            "lamp_green_hi",
            "lamp_red_hi",
            "lamp_estop_hi",
            "lamp_wash",
            "ring",
            "estop_label",
            "estop_edge",
            "key_badge",
            "key_badge_ink",
        )
        for key in themed:
            self.assertEqual(len(LIGHT[key]), 3)
            self.assertEqual(len(DARK[key]), 3)
        self.assertNotEqual(LIGHT["shadow"], DARK["shadow"])
        self.assertNotEqual(LIGHT["lamp_green_hi"], DARK["lamp_green_hi"])

        # Fixed status colours. Same in both themes, so they stay plain tuples.
        allowed_names = {"PEACH", "GREEN", "RED", "TEACHER", "ESTOP", "AMBER"}
        # The lamp disc is punched with an opaque white mask. It is not a paint colour.
        exceptions = {(255, 255, 255, 255)}
        text = Path("/workspace/sim/train_monitor.py").read_text(encoding="utf-8")
        text = re.sub(r"(?m)#.*$", "", text)
        kept = []
        for line in text.splitlines():
            named = re.match(r"\s*([A-Z][A-Z0-9_]*)\s*=\s*\(", line)
            if named and named.group(1) in allowed_names:
                continue
            kept.append(line)
        found = []
        pat = re.compile(r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+))?\)")
        for match in pat.finditer("\n".join(kept)):
            nums = tuple(int(group) for group in match.groups() if group is not None)
            if min(nums[:3]) < 40:
                continue
            if nums in exceptions:
                continue
            found.append(nums)
        self.assertEqual(found, [])


class FontFloorTests(unittest.TestCase):
    def test_scaled_font_stays_at_least_13px(self):
        import pygame

        from sim.train_monitor import MIN_FONT_PX, MIN_SCALE, MonitorView, TrainMonitor, font_px

        self.assertGreaterEqual(font_px(11, 1.0), MIN_FONT_PX)
        self.assertGreaterEqual(font_px(11, MIN_SCALE) * MIN_SCALE, MIN_FONT_PX)
        self.assertGreaterEqual(font_px(14, MIN_SCALE) * MIN_SCALE, MIN_FONT_PX)
        self.assertGreater(font_px(11, MIN_SCALE), 11)
        pygame.init()
        mon = TrainMonitor("fontfloor")
        try:
            mon._on_resize(1920, 1080)
            mon.draw(MonitorView(learner="mb", title="recognition training"))
            full = mon.font_tiny.get_height()
            self.assertGreaterEqual(mon._font_need, MIN_FONT_PX)
            mon._on_resize(1000, 700)
            mon.draw(MonitorView(learner="mb", title="recognition training"))
            self.assertGreaterEqual(mon._font_need * MIN_SCALE, MIN_FONT_PX)
            self.assertGreater(mon.font_tiny.get_height(), full)
        finally:
            pygame.quit()


class BoardLampTests(unittest.TestCase):
    def test_board_lamp_stays_when_the_word_does_not_fit(self):
        import pygame

        from sim.train_monitor import MonitorView, TrainMonitor

        pygame.init()
        mon = TrainMonitor("board")
        try:
            mon._on_resize(1920, 1080)
            view = MonitorView(udp_status="udp 127.0.0.1:5451 ok")
            surf = pygame.Surface((800, 40))
            surf.fill((10, 10, 10))
            mon.brain_rect = pygame.Rect(700, 0, 160, 24)
            mon.help_bar_rect = pygame.Rect(430, 4, 28, 24)
            mon._link_lamp(surf, view, 400)
            from sim.train_monitor import MENU_H

            self.assertGreater(max(surf.get_at((400, MENU_H // 2))[:3]), 40)
            band = pygame.surfarray.array3d(surf)[414:470, 6:30]
            self.assertTrue((band.max(axis=2) <= 12).all())
            wide = pygame.Surface((800, 40))
            wide.fill((10, 10, 10))
            mon.help_bar_rect = pygame.Rect(700, 4, 28, 24)
            mon._link_lamp(wide, view, 400)
            word = pygame.surfarray.array3d(wide)[414:470, 6:30]
            self.assertGreater(int((word.max(axis=2) > 40).sum()), 8)
        finally:
            pygame.quit()


class FooterClearanceTests(unittest.TestCase):
    def test_buttons_clear_the_footer_at_laptop_sizes(self):
        """Client 1920×1017, 1440×900, and scale 0.75 keep every button above the footer."""
        import pygame

        from sim.recognize_train import SIM_KEYS
        from sim.train_monitor import ESTOP, MonitorView, TrainMonitor, fit_window, window_from_logical

        pygame.init()
        mon = TrainMonitor("footer-fit")
        view = MonitorView(
            title="тренировка узнавания   go2_1",
            learner="mb",
            learning_on=True,
            yolo_state="смотрит",
            udp_status="симулятор. В автономии ведёт мозг, иначе стрелки или сценарий.",
            last_command="сценарий",
            keys_hint=SIM_KEYS,
            focused=True,
        )
        names = (
            "learn_rect",
            "auto_rect",
            "take_rect",
            "steer_rect",
            "treat_rect",
            "estop_rect",
            "stand_up_rect",
            "stand_down_rect",
            "recovery_rect",
            "lidar_reset_rect",
            "lidar_fresh_rect",
            "view_rect",
            "go2_rect",
            "brain_rect",
            "teacher_rect",
            "record_rect",
            "pin_rect",
            "theme_switch_rect",
        )
        try:
            for ww, wh in ((1920, 1017), (1440, 900)):
                mon._on_resize(ww, wh)
                mon.scroll_x = 0
                mon.scroll_y = 0
                mon.draw(view)
                self.assertEqual(mon.window.get_size(), (ww, wh))
                scale = fit_window(ww, wh)[0]
                if ww == 1440:
                    self.assertAlmostEqual(scale, 0.75, places=2)
                canvas_h = mon.screen.get_height()
                footer_h = int(mon._footer_h)
                footer_top = window_from_logical(
                    (16, mon.layout["footer_y"]),
                    ww,
                    wh,
                    (0, 0),
                    footer_h=footer_h,
                    canvas_h=canvas_h,
                )[1]
                self.assertGreater(footer_h, 0)
                self.assertLess(footer_top, wh)
                buttons = []
                for name in names:
                    rect = getattr(mon, name)
                    if rect.w < 8 or rect.h < 8:
                        continue
                    box = self._window_box(mon, rect, ww, wh)
                    buttons.append((name, box, rect))
                    self.assertGreaterEqual(box.x, 0, (name, ww, wh, box))
                    self.assertGreaterEqual(box.y, 0, (name, ww, wh, box))
                    self.assertLessEqual(box.right, ww, (name, ww, wh, box))
                    self.assertLessEqual(box.bottom, wh, (name, ww, wh, box))
                    self.assertLessEqual(box.bottom, footer_top, (name, ww, wh, box, footer_top))
                self.assertTrue(any(name == "estop_rect" for name, _box, _rect in buttons))
                estop = next(box for name, box, _rect in buttons if name == "estop_rect")
                self.assertGreaterEqual(estop.h, 24, (ww, wh, estop))
                frame = mon.present_into((ww, wh))
                sample = frame.get_at((estop.centerx, estop.centery))
                self.assertLess(abs(sample.r - ESTOP[0]), 30, (ww, wh, sample))
                self.assertLess(abs(sample.g - ESTOP[1]), 40, (ww, wh, sample))
                switch = self._window_box(mon, mon.theme_switch_rect, ww, wh)
                pin = self._window_box(mon, mon.pin_rect, ww, wh)
                self.assertLessEqual(switch.right, ww - 4, (ww, wh, switch))
                self.assertGreaterEqual(pin.x, 0)
                self.assertLess(pin.right, switch.x)
                if mon.theme_label_rect.w > 0:
                    label = self._window_box(mon, mon.theme_label_rect, ww, wh)
                    self.assertLessEqual(label.right, switch.x, (ww, wh, label, switch))
                    self.assertGreaterEqual(label.x, pin.right)

            ww, wh = 1920, 280
            mon._on_resize(ww, wh)
            mon.scroll_x = 0
            mon.scroll_y = 0
            mon.draw(view)
            geom = mon.scroll_geometry((ww, wh))
            self.assertIsNotNone(geom["thumb_y"])
            self.assertGreater(geom["max_y"], 0)
            mon.scroll_y = geom["max_y"]
            mon.draw(view)
            canvas_h = mon.screen.get_height()
            footer_h = int(mon._footer_h)
            footer_top = window_from_logical(
                (16, mon.layout["footer_y"]),
                ww,
                wh,
                (0, mon.scroll_y),
                footer_h=footer_h,
                canvas_h=canvas_h,
            )[1]
            client = pygame.Rect(0, 0, ww, wh)
            for name in names:
                rect = getattr(mon, name)
                if rect.w < 8 or rect.h < 8:
                    continue
                box = self._window_box(mon, rect, ww, wh, (0, mon.scroll_y))
                if not client.colliderect(box):
                    continue
                self.assertTrue(client.contains(box), (name, box))
                self.assertLessEqual(box.bottom, footer_top, (name, box, footer_top))
            estop = self._window_box(mon, mon.estop_rect, ww, wh, (0, mon.scroll_y))
            self.assertGreaterEqual(estop.h, 24)
            frame = mon.present_into((ww, wh))
            sample = frame.get_at((estop.centerx, estop.centery))
            self.assertLess(abs(sample.r - ESTOP[0]), 30, sample)
            self.assertLess(abs(sample.g - ESTOP[1]), 40, sample)
        finally:
            pygame.quit()

    def _window_box(self, mon, rect, ww, wh, scroll=(0, 0)):
        import pygame

        from sim.train_monitor import window_from_logical

        canvas_h = mon.screen.get_height()
        footer_h = int(mon._footer_h)
        x0, y0 = window_from_logical((rect.x, rect.y), ww, wh, scroll, footer_h=footer_h, canvas_h=canvas_h)
        x1, y1 = window_from_logical((rect.right, rect.bottom), ww, wh, scroll, footer_h=footer_h, canvas_h=canvas_h)
        return pygame.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
