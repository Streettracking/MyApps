"""Theme pill and the paperclip in the trainer menu bar.

Run: SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -m unittest sim.test_header_chrome
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class PinSettingTests(unittest.TestCase):
    def test_pin_flag_roundtrip(self):
        from sim.ui_settings import load_settings, save_settings

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.environ["RECOGNIZER_UI_CONFIG"] = path
        try:
            save_settings({"theme": "dark", "pin_on_top": "true"})
            loaded = load_settings()
            self.assertEqual(loaded["theme"], "dark")
            self.assertIs(loaded["pin_on_top"], True)
            save_settings({"theme": "light", "pin_on_top": False})
            again = load_settings()
            self.assertIs(again["pin_on_top"], False)
            self.assertEqual(again["theme"], "light")
        finally:
            os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            os.remove(path)


class HeaderChromeTests(unittest.TestCase):
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

    def test_switch_pin_persist_and_do_not_overlap(self):
        import pygame

        from sim.train_monitor import MENU_H, MIN_CONTENT_W, MIN_SCALE, WIN_H, WIN_W, MonitorView, TrainMonitor
        from sim.ui_theme import theme_name

        pygame.init()
        point = [8, 8]
        real_pos = pygame.mouse.get_pos
        pygame.mouse.get_pos = lambda: (int(point[0]), int(point[1]))
        try:
            mon = TrainMonitor("chrome")
            mon._on_resize(1920, 1080)
            view = MonitorView(title="recognition training", learner="mb", learning_on=True, keys_hint="A авто")
            self.assertFalse(mon.pin_on_top)
            self.assertEqual(theme_name(), "light")
            mon.draw(view)
            self._assert_gaps(mon)
            self._assert_teacher_in_controls(mon)
            self._assert_switch(mon, on=False)
            self._assert_pin_plate(mon, on=False)
            self._save_header(mon, "/opt/cursor/artifacts/header_theme_light.png")

            size = tuple(mon.window.get_size())
            themed = self._click(mon, mon.theme_label_rect.center, point)
            self.assertFalse(themed.quit)
            self.assertFalse(themed.brain_toggle)
            self.assertFalse(themed.record_toggle)
            self.assertFalse(themed.treat)
            self.assertEqual(theme_name(), "dark")
            saved = json.loads(__import__("pathlib").Path(self._path).read_text(encoding="utf-8"))
            self.assertEqual(saved["theme"], "dark")
            self.assertIs(saved["pin_on_top"], False)

            mon.menu_open = True
            mon.draw(view)
            self._assert_switch(mon, on=True)
            self._assert_menu_theme(mon, on=True)
            back = self._click(mon, mon.theme_switch_rect.center, point)
            self.assertFalse(back.brain_toggle)
            self.assertTrue(mon.menu_open)
            self.assertEqual(theme_name(), "light")
            mon.draw(view)
            self._assert_switch(mon, on=False)
            self._assert_menu_theme(mon, on=False)

            menu = self._click_menu(mon, ("theme",), point)
            self.assertFalse(menu.quit)
            self.assertTrue(mon.menu_open)
            self.assertEqual(theme_name(), "dark")
            mon.draw(view)
            self._assert_switch(mon, on=True)
            self._assert_menu_theme(mon, on=True)
            self._save_header(mon, "/opt/cursor/artifacts/header_theme_dark.png")

            pinned = self._click(mon, mon.pin_rect.center, point)
            self.assertFalse(pinned.brain_toggle)
            self.assertFalse(pinned.go2_control)
            self.assertFalse(pinned.treat)
            self.assertTrue(mon.pin_on_top)
            self.assertEqual(tuple(mon.window.get_size()), size)
            mon.draw(view)
            self._assert_pin_plate(mon, on=True)
            self._aim(mon, mon.pin_rect.center, point)
            mon.draw(view)
            self.assertIsNotNone(mon._pin_tip_rect)
            self.assertFalse(mon._pin_tip_rect.colliderect(mon.pin_rect))
            for rect in self._header_rects(mon):
                self.assertFalse(mon._pin_tip_rect.colliderect(rect), rect)
            self._save_header(mon, "/opt/cursor/artifacts/header_pin_on.png", with_tip=True)
            saved = json.loads(__import__("pathlib").Path(self._path).read_text(encoding="utf-8"))
            self.assertIs(saved["pin_on_top"], True)
            self.assertEqual(saved["theme"], "dark")

            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t, mod=0, unicode="t", scancode=0))
            keyed = mon.pump()
            self.assertTrue(keyed.treat)
            self.assertTrue(mon.pin_on_top)

            self._click(mon, mon.pin_rect.center, point)
            self.assertFalse(mon.pin_on_top)
            self.assertEqual(tuple(mon.window.get_size()), size)
            self._click(mon, mon.pin_rect.center, point)
            self.assertTrue(mon.pin_on_top)

            mon._on_resize(MIN_CONTENT_W, max(64, int(round(WIN_H * MIN_SCALE))))
            mon.draw(view)
            self._assert_gaps(mon)
            self._assert_scaled_gaps(mon, MIN_CONTENT_W / float(WIN_W))
            mon._on_resize(800, 600)
            mon.draw(view)
            self._assert_gaps(mon)

            kept = json.loads(__import__("pathlib").Path(self._path).read_text(encoding="utf-8"))
            self.assertIs(kept["pin_on_top"], True)
            self.assertEqual(kept["theme"], "dark")
            kept["w"] = None
            kept["h"] = None
            kept["x"] = None
            kept["y"] = None
            kept["maximized"] = False
            kept["fullscreen"] = False
            __import__("pathlib").Path(self._path).write_text(json.dumps(kept), encoding="utf-8")

            pygame.quit()
            pygame.init()
            restored = TrainMonitor("chrome-restored")
            restored._on_resize(1920, 1080)
            self.assertTrue(restored.pin_on_top)
            self.assertEqual(theme_name(), "dark")
            restored.draw(view)
            self._assert_switch(restored, on=True)
            self._assert_pin_plate(restored, on=True)
            self._assert_gaps(restored)
        finally:
            pygame.mouse.get_pos = real_pos
            pygame.quit()

    def _header_rects(self, mon):
        return (
            mon.view_rect,
            mon.doggibot_rect,
            mon.help_bar_rect,
            mon.go2_rect,
            mon.brain_rect,
            mon.record_rect,
            mon.pin_rect,
            mon.theme_label_rect,
            mon.theme_switch_rect,
        )

    def _assert_teacher_in_controls(self, mon) -> None:
        from sim.train_monitor import MENU_H

        self.assertGreater(mon.teacher_rect.y, MENU_H)
        self.assertGreater(mon.teacher_rect.w, 48)
        self.assertLessEqual(mon.teacher_rect.x, mon.learn_rect.right + 12)
        self.assertGreaterEqual(mon.teacher_rect.x, mon.learn_rect.right)
        self.assertEqual(mon.teacher_rect.centery, mon.learn_rect.centery)

    def _assert_gaps(self, mon) -> None:
        from sim.train_monitor import MENU_H, WIN_W

        ordered = (
            mon.help_bar_rect,
            mon.go2_rect,
            mon.brain_rect,
            mon.record_rect,
            mon.pin_rect,
            mon.theme_label_rect,
            mon.theme_switch_rect,
        )
        for rect in self._header_rects(mon):
            self.assertGreaterEqual(rect.x, 0)
            self.assertLessEqual(rect.right, WIN_W)
            self.assertGreaterEqual(rect.y, 0)
            self.assertLessEqual(rect.bottom, MENU_H)
        for left, right in zip(ordered, ordered[1:]):
            self.assertLessEqual(left.right + 4, right.left, (left, right))
        rects = self._header_rects(mon)
        for index, left in enumerate(rects):
            for right in rects[index + 1 :]:
                self.assertFalse(left.colliderect(right), (left, right))
        title_w = mon.font.size("recognition training")[0]
        title_x = mon.doggibot_rect.right + 12
        lamp_w = 14 + mon.font_sm.size("борт")[0]
        self.assertLess(title_x + title_w + 18 + lamp_w, mon.help_bar_rect.x)
        self.assertEqual((mon.theme_switch_rect.w, mon.theme_switch_rect.h), (36, 20))
        self.assertEqual((mon.pin_rect.w, mon.pin_rect.h), (30, 26))
        self.assertLessEqual(mon.theme_switch_rect.right, WIN_W - 8)

    def _assert_scaled_gaps(self, mon, scale: float) -> None:
        import pygame

        def scaled(rect):
            left = int(round(rect.x * scale))
            top = int(round(rect.y * scale))
            right = int(round(rect.right * scale))
            bottom = int(round(rect.bottom * scale))
            return pygame.Rect(left, top, max(1, right - left), max(1, bottom - top))

        ordered = [scaled(rect) for rect in (
            mon.help_bar_rect,
            mon.go2_rect,
            mon.brain_rect,
            mon.record_rect,
            mon.pin_rect,
            mon.theme_label_rect,
            mon.theme_switch_rect,
        )]
        for left, right in zip(ordered, ordered[1:]):
            self.assertLess(left.right, right.left, (left, right))
            self.assertFalse(left.colliderect(right))

    def _assert_switch(self, mon, on: bool) -> None:
        rect = mon.theme_switch_rect
        radius = rect.h // 2
        knob_x = rect.x + (rect.w - radius if on else radius)
        knob_y = rect.y + radius
        knob = self._rgb(mon, knob_x, knob_y)
        self.assertGreater(knob[0], 230)
        self.assertGreater(knob[1], 230)
        self.assertGreater(knob[2], 230)
        if on:
            track = self._rgb(mon, rect.x + 4, rect.centery)
            self.assertGreater(track[0], track[2] + 40)
            self.assertGreater(track[0], 160)
            idle = self._rgb(mon, knob_x, knob_y)
            self.assertGreater(idle[0], 230)
        else:
            track = self._rgb(mon, rect.right - 4, rect.centery)
            self.assertLess(abs(track[0] - track[1]), 30)
            self.assertGreater(track[0], 140)
            self.assertLess(track[0], 230)

    def _assert_pin_plate(self, mon, on: bool) -> None:
        rect = mon.pin_rect
        plate = self._rgb(mon, rect.x + 4, rect.y + 4)
        if on:
            self.assertGreater(plate[0], plate[2] + 40)
            self.assertGreater(plate[0], 160)
        else:
            self.assertLess(abs(plate[0] - plate[1]), 24)
            self.assertGreater(plate[0], 180)
        tones = set()
        for x in range(rect.x + 6, rect.right - 4, 2):
            for y in range(rect.y + 4, rect.bottom - 4, 2):
                tones.add(self._rgb(mon, x, y))
        self.assertGreater(len(tones), 3)

    def _assert_menu_theme(self, mon, on: bool) -> None:
        import pygame

        frame = pygame.surfarray.array3d(mon.screen)
        row = next(rect for rect, action in mon._menu_hits if action == ("theme",))
        box = pygame.Rect(row.x + 8, row.centery - 7, 14, 14)
        interior = frame[box.x + 3 : box.x + 11, box.y + 3 : box.y + 11]
        if on:
            self.assertGreater(int(interior.max()) - int(interior.min()), 80)
        else:
            self.assertLess(int(interior.max()) - int(interior.min()), 8)

    def _aim(self, mon, logical, point) -> None:
        from sim.train_monitor import window_from_logical

        wx, wy = window_from_logical(logical, mon.window.get_width(), mon.window.get_height(), (mon.scroll_x, mon.scroll_y))
        point[0], point[1] = int(wx), int(wy)

    def _click(self, mon, logical, point):
        import pygame

        self._aim(mon, logical, point)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(point[0]), int(point[1])), button=1))
        return mon.pump()

    def _click_menu(self, mon, action, point):
        rect = next(hit for hit, item in mon._menu_hits if item == action)
        return self._click(mon, rect.center, point)

    def _rgb(self, mon, x, y):
        color = mon.screen.get_at((int(x), int(y)))
        return (color.r, color.g, color.b)

    def _save_header(self, mon, path: str, with_tip: bool = False) -> None:
        import pygame

        from sim.train_monitor import MENU_H, WIN_W

        out = __import__("pathlib").Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        bottom = MENU_H
        if with_tip and getattr(mon, "_pin_tip_rect", None) is not None:
            bottom = mon._pin_tip_rect.bottom + 6
        left = max(0, mon.help_bar_rect.x - 12)
        crop = pygame.Rect(left, 0, WIN_W - left, bottom)
        shot = mon.screen.subsurface(crop).copy()
        big = pygame.transform.smoothscale(shot, (shot.get_width() * 2, shot.get_height() * 2))
        pygame.image.save(big, str(out))
