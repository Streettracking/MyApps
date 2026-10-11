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
            find_leaf(mon.dock, "brain").visible = False
            mon._apply_layout(mon.screen.get_height())
            mon.menu_open = True
            mon._set_theme("light")
            mon.draw(view)
            self._assert_menu_checks(mon, on_theme="light")
            self._save_menu(mon, "/opt/cursor/artifacts/view_menu_light.png")
            self._assert_row_hover(
                mon, ("toggle", "brain"), shot="/opt/cursor/artifacts/view_menu_c_light.png"
            )
            self._assert_row_hover(mon, ("locked", "controls"), skipped=True)
            mon.draw(view)

            brain = self._click_menu(mon, ("toggle", "brain"), at="right")
            self.assertFalse(brain.quit)
            self.assertTrue(mon.menu_open)
            self.assertTrue(find_leaf(mon.dock, "brain").visible)
            mon.draw(view)
            brain_off = self._click_menu(mon, ("toggle", "brain"), at="left")
            self.assertFalse(brain_off.quit)
            self.assertTrue(mon.menu_open)
            self.assertFalse(find_leaf(mon.dock, "brain").visible)
            self._click_menu(mon, ("toggle", "brain"))
            self.assertTrue(find_leaf(mon.dock, "brain").visible)
            locked = self._click_menu(mon, ("locked", "controls"), at="right")
            self.assertFalse(locked.quit)
            self.assertTrue(mon.menu_open)
            self.assertTrue(find_leaf(mon.dock, "controls").visible)
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
            find_leaf(mon.dock, "brain").visible = False
            mon._apply_layout(mon.screen.get_height())
            mon.draw(view)
            self._assert_row_hover(
                mon, ("toggle", "brain"), shot="/opt/cursor/artifacts/view_menu_c_dark.png"
            )
            find_leaf(mon.dock, "brain").visible = True
            mon._apply_layout(mon.screen.get_height())
            mon.draw(view)

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
            self.assertTrue(find_leaf(mon.dock, "brain").visible)
            self.assertLess(mon.cam_rect.y, mon.lid_rect.y)
        finally:
            pygame.quit()

    def _assert_row_hover(self, mon, action, skipped: bool = False, shot: str = "") -> None:
        import pygame

        row = next(rect for rect, item in mon._menu_hits if item == action)
        real = pygame.mouse.get_pos
        pygame.mouse.get_pos = lambda: (int(row.centerx), int(row.centery))
        try:
            mon.draw(mon._last_view)
            if shot:
                self._save_menu_panel(mon, shot)
            spots = ((row.x + 4, row.y + 2), (row.right - 4, row.y + 3))
            samples = [mon.screen.get_at(spot) for spot in spots]
        finally:
            pygame.mouse.get_pos = real
        quiet = next(rect for rect, item in mon._menu_hits if item == ("toggle", "camera"))
        bases = (
            mon.screen.get_at((quiet.x + 4, quiet.y + 2)),
            mon.screen.get_at((quiet.right - 4, quiet.y + 3)),
        )
        for sample, base in zip(samples, bases):
            if skipped:
                self.assertLess(abs(int(sample.r) - int(base.r)), 8)
            else:
                self.assertGreater(abs(int(sample.r) - int(base.r)), 8)

    def _click_at(self, mon, logical) -> None:
        import pygame

        from sim.train_monitor import window_from_logical

        wx, wy = window_from_logical(logical, 1920, 1080)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(wx), int(wy)), button=1))
        return mon.pump()

    def _click_menu(self, mon, action, at: str = "center"):
        rect = next(hit for hit, item in mon._menu_hits if item == action)
        if at == "left":
            point = (rect.x + 2, rect.centery)
        elif at == "right":
            point = (rect.right - 2, rect.centery)
        else:
            point = rect.center
        return self._click_at(mon, point)

    def _slot_patch(self, frame, slot):
        return frame[slot.x + 1 : slot.right - 1, slot.y + 1 : slot.bottom - 1]

    def _assert_empty_slot(self, frame, slot) -> None:
        patch = self._slot_patch(frame, slot)
        rs = patch[:, :, 0].astype(int)
        bs = patch[:, :, 2].astype(int)
        self.assertLess(int(rs.max()) - int(rs.min()), 24, (int(rs.min()), int(rs.max())))
        self.assertFalse(bool(((rs > bs + 30) & (rs > 150)).any()))

    def _assert_peach_mark(self, frame, slot) -> None:
        import numpy as np

        patch = self._slot_patch(frame, slot)
        rs = patch[:, :, 0].astype(int)
        gs = patch[:, :, 1].astype(int)
        bs = patch[:, :, 2].astype(int)
        mask = (rs > bs + 18) & (rs > gs)
        self.assertGreater(int(mask.sum()), 6, int(mask.sum()))
        dist = np.abs(rs - 201) + np.abs(gs - 120) + np.abs(bs - 91)
        self.assertLess(int(dist[mask].min()), 55, int(dist[mask].min()))

    def _assert_gray_mark(self, frame, slot) -> None:
        import numpy as np

        patch = self._slot_patch(frame, slot)
        rs = patch[:, :, 0].astype(int)
        gs = patch[:, :, 1].astype(int)
        bs = patch[:, :, 2].astype(int)
        lum = (rs + gs + bs) / 3.0
        mask = np.abs(lum - float(np.median(lum))) > 24
        self.assertGreater(int(mask.sum()), 6, int(mask.sum()))
        self.assertLess(float(np.median(np.abs(rs[mask] - gs[mask]))), 16)
        self.assertLess(float(np.median(np.abs(rs[mask] - bs[mask]))), 16)
        dist = np.abs(rs - 201) + np.abs(gs - 120) + np.abs(bs - 91)
        self.assertGreater(int(dist[mask].min()), 70, int(dist[mask].min()))

    def _assert_menu_switch(self, frame, rect, on: bool) -> None:
        radius = rect.h // 2
        knob_x = rect.x + (rect.w - radius if on else radius)
        knob_y = rect.centery
        knob = frame[int(knob_x), int(knob_y)]
        self.assertGreater(int(knob.min()), 220)
        if on:
            track = frame[rect.x + 4, rect.centery]
            self.assertGreater(int(track[0]), int(track[2]) + 40)
            self.assertGreater(int(track[0]), 160)
        else:
            track = frame[rect.right - 4, rect.centery]
            self.assertLess(abs(int(track[0]) - int(track[1])), 30)
            self.assertGreater(int(track[0]), 140)
            self.assertLess(int(track[0]), 230)

    def _assert_menu_checks(self, mon, on_theme: str) -> None:
        import pygame

        from sim.dock_layout import find_leaf
        from sim.ui_theme import theme_name

        self.assertEqual(theme_name(), on_theme)
        frame = pygame.surfarray.array3d(mon.screen)
        order = [action for _rect, action in mon._menu_hits]
        self.assertEqual(
            order,
            [
                ("toggle", "camera"),
                ("toggle", "lidar"),
                ("toggle", "brain"),
                ("toggle", "journal"),
                ("toggle", "robot"),
                ("locked", "controls"),
                ("reset",),
                ("theme",),
                ("doggibot",),
                ("go2",),
            ],
        )
        self.assertEqual(
            [mon._menu_captions[action] for action in order[:8]],
            [
                "Камера",
                "Лидар",
                "Панель мозга",
                "Журнал",
                "Робот",
                "Кнопки (E-STOP)",
                "Сброс раскладки",
                "Тёмная тема",
            ],
        )
        self.assertEqual(len([action for action in order if action[0] == "theme"]), 1)
        self.assertNotIn(("theme", "light"), order)
        self.assertNotIn(("theme", "dark"), order)
        self.assertEqual((mon._menu_switch_rect.w, mon._menu_switch_rect.h), (36, 20))
        slot_x = None
        for _rect, action in mon._menu_hits:
            slot = mon._menu_marks[action]
            if slot_x is None:
                slot_x = slot.x
            self.assertEqual(slot.x, slot_x)
            self.assertEqual(slot.w, 16)
            self.assertEqual(slot.x + 24, mon._menu_label_x)
            row = next(rect for rect, item in mon._menu_hits if item == action)
            self.assertEqual(row.x, mon._menu_panel.x)
            self.assertEqual(row.w, mon._menu_panel.w)
            if action[0] == "toggle":
                if find_leaf(mon.dock, action[1]).visible:
                    self._assert_peach_mark(frame, slot)
                else:
                    self._assert_empty_slot(frame, slot)
            elif action[0] == "locked":
                self.assertTrue(find_leaf(mon.dock, action[1]).visible)
                self._assert_gray_mark(frame, slot)
            else:
                self._assert_empty_slot(frame, slot)
        self._assert_menu_switch(frame, mon._menu_switch_rect, on_theme == "dark")
        controls = next(rect for rect, action in mon._menu_hits if action == ("locked", "controls"))
        reset = next(rect for rect, action in mon._menu_hits if action[0] == "reset")
        rule = mon._menu_rule
        self.assertGreater(rule.centery, controls.bottom)
        self.assertLess(rule.centery, reset.top)
        self.assertLess(rule.y - controls.bottom, 16)
        self.assertLess(reset.top - rule.bottom, 16)
        above = frame[rule.centerx, rule.y - 4]
        line = frame[rule.centerx, rule.y]
        self.assertGreaterEqual(abs(int(line[0]) - int(above[0])), 8)

    def _save_menu_panel(self, mon, path: str) -> None:
        import pygame

        panel = mon._menu_panel
        crop = pygame.Rect(panel.x - 20, panel.y - 16, panel.w + 40, panel.h + 40)
        crop = crop.clamp(mon.screen.get_rect())
        shot = pygame.Surface((crop.w, crop.h))
        shot.blit(mon.screen, (0, 0), area=crop)
        pygame.image.save(pygame.transform.smoothscale(shot, (crop.w * 2, crop.h * 2)), path)

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

    def test_view_check_restores_every_other_block_and_controls_stays_on(self):
        import pygame
        import numpy as np

        from sim.dock_layout import BLOCKS, find_leaf
        from sim.train_monitor import MonitorView, TrainMonitor

        pygame.init()
        try:
            mon = TrainMonitor("controls-pin")
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
            original = {name: tuple(rect) for name, rect in mon._dock_blocks.items()}
            saved = mon.dock.to_dict()
            mon.menu_open = True
            mon.draw(view)
            actions = [action for _rect, action in mon._menu_hits]
            self.assertIn(("locked", "controls"), actions)
            self.assertNotIn(("toggle", "controls"), actions)

            frame = pygame.surfarray.array3d(mon.screen)
            self._assert_peach_mark(frame, mon._menu_marks[("toggle", "camera")])
            self._assert_gray_mark(frame, mon._menu_marks[("locked", "controls")])
            self.assertEqual(
                {mon._menu_marks[action].x for _rect, action in mon._menu_hits},
                {mon._menu_label_x - 24},
            )

            self._click_menu(mon, ("locked", "controls"))
            self.assertTrue(mon.menu_open)
            self.assertTrue(find_leaf(mon.dock, "controls").visible)
            self.assertFalse(find_leaf(mon.dock, "controls").collapsed)
            self.assertEqual({name: tuple(rect) for name, rect in mon._dock_blocks.items()}, original)

            for name in BLOCKS:
                if name == "controls":
                    continue
                self._click_menu(mon, ("toggle", name))
                self.assertTrue(mon.menu_open)
                self.assertFalse(find_leaf(mon.dock, name).visible)
                self.assertNotIn(name, mon._dock_blocks)
                self._click_menu(mon, ("toggle", name))
                self.assertTrue(mon.menu_open)
                self.assertTrue(find_leaf(mon.dock, name).visible)
                self.assertEqual({key: tuple(rect) for key, rect in mon._dock_blocks.items()}, original)
                self.assertEqual(mon.dock.to_dict(), saved)

            kinds = [kind for _hit, kind, arg in mon._chrome if arg == "controls"]
            self.assertEqual(kinds, ["info"])
            info = next(hit for hit, kind, arg in mon._chrome if kind == "info" and arg == "controls")
            block = mon._dock_blocks["controls"]
            self.assertGreaterEqual(info.right, block[0] + block[2] - 8)
            self._click_at(mon, info.center)
            self.assertEqual(mon.help_id, "controls")
            self.assertTrue(find_leaf(mon.dock, "controls").visible)
            self.assertFalse(find_leaf(mon.dock, "controls").collapsed)
            self.assertEqual({name: tuple(rect) for name, rect in mon._dock_blocks.items()}, original)
        finally:
            pygame.quit()


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
                self.assertEqual((rect.w, rect.h), (20, 20))
                block = mon._dock_blocks[name]
                if name == "controls":
                    kinds = [kind for _hit, kind, arg in mon._chrome if arg == name]
                    self.assertNotIn("collapse", kinds)
                    self.assertNotIn("close", kinds)
                    self.assertGreaterEqual(rect.right, block[0] + block[2] - 8)
                    continue
                collapse = next(hit for hit, kind, arg in mon._chrome if kind == "collapse" and arg == name)
                close = next(hit for hit, kind, arg in mon._chrome if kind == "close" and arg == name)
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


class CardButtonTests(unittest.TestCase):
    def test_buttons_sit_under_the_card_header_and_accept_clicks(self):
        import pygame

        from sim.dock_layout import find_leaf
        from sim.train_monitor import MonitorView, TrainMonitor, window_from_logical

        pygame.init()
        mon = TrainMonitor("card-buttons")
        view = MonitorView(learner="mb", title="recognition training", onboard=True)
        try:
            mon._on_resize(1920, 1017)
            mon.draw(view)
            controls = mon._content_of("controls")
            robot = mon._content_of("robot")
            self.assertIsNotNone(controls)
            self.assertIsNotNone(robot)
            for name in ("learn_rect", "auto_rect", "take_rect", "steer_rect", "treat_rect", "estop_rect"):
                rect = getattr(mon, name)
                self.assertTrue(controls.contains(rect), (name, rect, controls))
                self.assertEqual(rect.y, controls.y, name)
            for name in ("stand_up_rect", "stand_down_rect", "recovery_rect", "lidar_reset_rect"):
                rect = getattr(mon, name)
                self.assertTrue(robot.contains(rect), (name, rect, robot))
                self.assertEqual(rect.y, robot.y, name)
            self.assertLess(mon.estop_bar_rect.w, 8)

            wx, wy = window_from_logical(
                mon.learn_rect.center,
                1920,
                1017,
                (0, 0),
                footer_h=int(mon._footer_h),
                canvas_h=mon.screen.get_height(),
            )
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(wx), int(wy)), button=1))
            hit = mon.pump()
            self.assertTrue(hit.pause_learn)

            leaf = find_leaf(mon.dock, "controls")
            leaf.collapsed = True
            leaf.visible = False
            mon._apply_layout(mon.screen.get_height())
            mon.draw(view)
            self.assertTrue(leaf.visible)
            self.assertFalse(leaf.collapsed)
            self.assertGreaterEqual(mon.estop_rect.w, 8)
            self.assertTrue(mon._content_of("controls").contains(mon.estop_rect))
            self.assertLess(mon.estop_bar_rect.w, 8)
            kinds = [kind for _hit, kind, arg in mon._chrome if arg == "controls"]
            self.assertIn("info", kinds)
            self.assertNotIn("collapse", kinds)
            self.assertNotIn("close", kinds)

            mon._on_resize(1920, 280)
            mon.scroll_x = 0
            mon.scroll_y = 0
            mon.draw(view)
            self.assertLess(mon.estop_bar_rect.w, 8)
            self.assertTrue(find_leaf(mon.dock, "controls").visible)
            self.assertFalse(find_leaf(mon.dock, "controls").collapsed)
        finally:
            pygame.quit()

    def test_saved_1440_layout_keeps_a_readable_lidar_when_maximized(self):
        import pygame

        from sim.dock_layout import HEADER, LIDAR_MIN_H, default_tree, layout_dock, tree_from_dict
        from sim.hemifield import format_fly_line
        from sim.pilot import format_range_line
        from sim.tabnum import load_font
        from sim.train_monitor import FOOTER_H, MENU_H, WIN_W, MonitorView, TrainMonitor

        area = 1080 - MENU_H - FOOTER_H
        blocks = layout_dock(default_tree(), 12, MENU_H, WIN_W - 24, area)["blocks"]
        cam_w, cam_h = blocks["camera"][2], blocks["camera"][3]
        self.assertEqual(cam_h, HEADER + int(round(cam_w * 9 / 16)))
        self.assertGreaterEqual(blocks["lidar"][3], LIDAR_MIN_H)

        pygame.init()
        mon = TrainMonitor("lidar-floor")
        view = MonitorView(
            learner="mb",
            title="тренировка узнавания",
            fly_line=format_fly_line(142, 353, 211, 0.4, "bilateral"),
            range_line=format_range_line(2.07, 1.4, 0, 8, {"window": 8, "votes": 0}),
        )
        try:
            mon._on_resize(1440, 900)
            mon.draw(view)
            saved = mon.dock.to_dict()
            self.assertFalse(saved["a"]["a"]["locked"])
            wide = mon.lid_rect.h
            mon.dock = tree_from_dict(saved)
            mon._on_resize(1920, 1017)
            mon.draw(view)
            self.assertGreaterEqual(mon.lid_rect.h, LIDAR_MIN_H)
            pref = HEADER + int(round(mon.cam_rect.w * 9 / 16))
            self.assertLess(mon.cam_rect.h, pref)
            column = mon.cam_rect.h + 6 + mon.lid_rect.h
            self.assertAlmostEqual(mon.cam_rect.h / float(column), float(saved["a"]["a"]["r"]), delta=0.04)
            self.assertGreater(wide, mon.lid_rect.h)
            step = max(mon.font_sm.get_height(), load_font(14).get_height())
            room = mon.lidar_column_rect.bottom - (mon.lidar_fresh_rect.bottom + 6)
            self.assertGreaterEqual(room // step, 5, (room, step, mon.lid_rect))
            labels = [rect for rect, group in mon._glyph_rects if group == "lidar"]
            self.assertGreaterEqual(len(labels), 5)
        finally:
            pygame.quit()


class DockReflowTests(unittest.TestCase):
    """Closing, collapsing, and reopening must tile the dock and restore shares."""

    def _area(self):
        from sim.train_monitor import FOOTER_H, MENU_H, WIN_W

        return 12, MENU_H, WIN_W - 24, 1080 - MENU_H - FOOTER_H

    def _place(self, tree):
        from sim.dock_layout import layout_dock

        x, y, w, h = self._area()
        return layout_dock(tree, x, y, w, h), (x, y, w, h)

    def _assert_tiled(self, laid, area):
        blocks = laid["blocks"]
        rects = list(blocks.values()) + [item["rect"] for item in laid["splitters"]]
        x, y, w, h = area
        covered = [[0] * w for _ in range(h)]
        for rx, ry, rw, rh in rects:
            for yy in range(max(0, ry - y), min(h, ry - y + rh)):
                row = covered[yy]
                x0 = max(0, rx - x)
                x1 = min(w, rx - x + rw)
                for xx in range(x0, x1):
                    row[xx] += 1
        empty = sum(cell == 0 for row in covered for cell in row)
        self.assertEqual(empty, 0, empty)
        names = list(blocks)
        for i, left in enumerate(names):
            ax, ay, aw, ah = blocks[left]
            for right in names[i + 1 :]:
                bx, by, bw, bh = blocks[right]
                ix = min(ax + aw, bx + bw) - max(ax, bx)
                iy = min(ay + ah, by + bh) - max(ay, by)
                self.assertFalse(ix > 0 and iy > 0, (left, right))

    def test_each_block_closes_collapses_and_reopens_without_holes(self):
        from sim.dock_layout import BLOCKS, HEADER, default_tree, find_leaf

        fresh = default_tree()
        base, area = self._place(fresh)
        self._assert_tiled(base, area)
        saved = fresh.to_dict()

        for name in BLOCKS:
            if name == "controls":
                continue
            tree = default_tree()
            before, _area = self._place(tree)
            self.assertEqual(before["blocks"], base["blocks"])
            leaf = find_leaf(tree, name)
            leaf.visible = False
            hidden, _area = self._place(tree)
            self.assertNotIn(name, hidden["blocks"])
            self._assert_tiled(hidden, area)
            self.assertEqual(tree.to_dict()["a"]["r"], saved["a"]["r"])
            leaf.visible = True
            opened, _area = self._place(tree)
            self.assertEqual(opened["blocks"], base["blocks"])
            self.assertEqual(tree.to_dict(), saved)

            leaf.collapsed = True
            folded, _area = self._place(tree)
            self.assertIn(name, folded["blocks"])
            self.assertEqual(folded["blocks"][name][3], HEADER, name)
            self._assert_tiled(folded, area)
            self.assertNotEqual(folded["blocks"][name], base["blocks"][name])
            leaf.collapsed = False
            restored, _area = self._place(tree)
            self.assertEqual(restored["blocks"], base["blocks"])
            self.assertEqual(tree.to_dict(), saved)

    def test_a_closed_column_gives_its_width_to_the_neighbor(self):
        from sim.dock_layout import HEADER, SPLIT, Leaf, Split, default_tree, find_leaf

        tree = default_tree()
        base, area = self._place(tree)
        find_leaf(tree, "camera").visible = False
        find_leaf(tree, "lidar").visible = False
        laid, _area = self._place(tree)
        self.assertNotIn("camera", laid["blocks"])
        self.assertNotIn("lidar", laid["blocks"])
        self._assert_tiled(laid, area)
        self.assertEqual(laid["blocks"]["brain"][2], area[2])
        self.assertEqual(laid["blocks"]["journal"][2], area[2])
        self.assertEqual(laid["blocks"]["controls"][2:], base["blocks"]["controls"][2:])
        self.assertEqual(laid["blocks"]["robot"][2:], base["blocks"]["robot"][2:])
        find_leaf(tree, "camera").visible = True
        find_leaf(tree, "lidar").visible = True
        self.assertEqual(self._place(tree)[0]["blocks"], base["blocks"])

        # The remaining pair keeps its saved share of the column.
        column = Split(
            "v",
            0.5,
            Leaf("camera"),
            Split("v", 0.4, Leaf("lidar"), Leaf("journal"), locked=True),
            locked=True,
        )
        find_leaf(column, "camera").visible = False
        pair = layout_of(column)
        span = pair["lidar"][3] + SPLIT + pair["journal"][3]
        self.assertAlmostEqual(pair["lidar"][3] / float(span - SPLIT), 0.4, delta=0.02)
        self.assertAlmostEqual(column.b.ratio, 0.4, places=3)

        locked = default_tree()
        locked.a.a.locked = True
        locked.a.a.ratio = 0.5
        open_blocks = self._place(locked)[0]["blocks"]
        find_leaf(locked, "lidar").collapsed = True
        folded = self._place(locked)[0]
        self.assertEqual(folded["blocks"]["lidar"][3], HEADER)
        self.assertGreater(folded["blocks"]["camera"][3], open_blocks["camera"][3])
        self.assertTrue(locked.a.a.locked)
        self.assertAlmostEqual(locked.a.a.ratio, 0.5, places=3)
        find_leaf(locked, "lidar").collapsed = False
        self.assertEqual(self._place(locked)[0]["blocks"], open_blocks)

    def test_old_window_json_cannot_hide_or_fold_the_buttons_card(self):
        from sim.dock_layout import default_tree, find_leaf, tree_from_dict

        saved = default_tree().to_dict()

        def paint(node, name, **flags):
            if node.get("t") == "leaf" and node.get("id") == name:
                node.update(flags)
                return
            for side in ("a", "b"):
                child = node.get(side)
                if isinstance(child, dict):
                    paint(child, name, **flags)

        paint(saved, "controls", visible=False, collapsed=True)
        paint(saved, "journal", visible=False, collapsed=True)
        tree = tree_from_dict(saved)
        controls = find_leaf(tree, "controls")
        journal = find_leaf(tree, "journal")
        self.assertTrue(controls.visible)
        self.assertFalse(controls.collapsed)
        self.assertFalse(journal.visible)
        self.assertTrue(journal.collapsed)

        def leaf_of(node, name):
            if node.get("t") == "leaf":
                return node if node.get("id") == name else None
            for side in ("a", "b"):
                found = leaf_of(node.get(side) or {}, name)
                if found is not None:
                    return found
            return None

        written = tree.to_dict()
        self.assertTrue(leaf_of(written, "controls")["visible"])
        self.assertFalse(leaf_of(written, "controls")["collapsed"])
        self.assertFalse(leaf_of(written, "journal")["visible"])
        self.assertTrue(leaf_of(written, "journal")["collapsed"])

        fresh = default_tree()
        find_leaf(fresh, "journal").visible = False
        find_leaf(fresh, "journal").collapsed = True
        loaded, _area = self._place(tree)
        expected, _area = self._place(fresh)
        self.assertEqual(loaded["blocks"], expected["blocks"])
        self.assertIn("controls", loaded["blocks"])
        self.assertGreater(loaded["blocks"]["controls"][3], 26)
        self.assertNotIn("journal", loaded["blocks"])

        live = default_tree()
        leaf = find_leaf(live, "controls")
        leaf.visible = False
        leaf.collapsed = True
        opened, _area = self._place(live)
        self.assertTrue(leaf.visible)
        self.assertFalse(leaf.collapsed)
        base, _area = self._place(default_tree())
        self.assertEqual(opened["blocks"]["controls"], base["blocks"]["controls"])


def layout_of(tree):
    from sim.dock_layout import layout_dock

    return layout_dock(tree, 0, 0, 400, 800)["blocks"]
