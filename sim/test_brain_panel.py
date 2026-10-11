"""Brain-panel rows stay apart, and embedded zoom does not rebuild the mesh.

Run: python3 -m unittest sim.test_brain_panel
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def _populated_view():
    import numpy as np

    from sim.hemifield import format_fly_line
    from sim.pilot import format_eyes_line, format_range_line
    from sim.tabnum import format_counts, format_lidar_caption, format_skips
    from sim.train_monitor import MonitorView

    curve = [(float(i), 0.2 + 0.05 * (i % 5)) for i in range(12)]
    other = [(float(i), 0.1 + 0.02 * (i % 3)) for i in range(12)]
    drift = [(float(i), 0.4 * i) for i in range(8)]
    return MonitorView(
        title="тренировка узнавания",
        learner="mb",
        learning_on=True,
        camera=np.zeros((90, 160, 3), dtype=np.uint8),
        lidar=_lidar_picture(),
        eyes_line=format_eyes_line(12, -4, True, False),
        recog_line="узнавание: только камера",
        pilot_mode="РУЧНОЕ",
        pilot_who="оператор",
        phase_ru="поиск",
        last_seen_side="R",
        likeness=18.5,
        readout_caption="сырой выход: подход - избегание",
        total_pam=128,
        total_ppl1=17,
        lifetime_known=True,
        n_pam=9,
        n_ppl1=2,
        session_time=3661,
        session_sep=1.25,
        session_acc=0.8,
        session_labeled=4,
        total_sep=-0.5,
        total_acc=0.25,
        total_labeled=11,
        peer_curve=curve,
        other_curve=other,
        drift=12.5,
        drift_curve=drift,
        t=48.0,
        dan_events=[(2.0, "PAM"), (10.0, "PPL1"), (40.0, "PAM"), (46.0, "PPL1")],
        kc_on=240,
        kc_n=5137,
        kc_bins=np.linspace(0.1, 1.0, 48),
        teacher_counts=format_counts(9, 4, 2, 1),
        teacher_skips=format_skips(3, "—"),
        lidar_mode=format_lidar_caption(True, 1.5),
        lidar_fresh_on=True,
        fly_line=format_fly_line(-72, 66, -139, -1.0, "bilateral"),
        range_line=format_range_line(2.07, 1.4, 0, 8, {"window": 8, "votes": 0}),
        log_lines=["журнал"],
        keys_hint="A авто  J мозг",
    )


def _lidar_picture():
    import numpy as np

    lidar = np.zeros((64, 64, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:64, :64]
    ring = np.abs(np.hypot(xx - 31.5, yy - 31.5) - 22) < 1.6
    lidar[ring] = (210, 214, 218)
    return lidar


def _assert_packed(test, rows, panel) -> None:
    import pygame

    panel = pygame.Rect(panel)
    rects = []
    for rect, tag in rows:
        test.assertGreaterEqual(rect.x, panel.x, tag)
        test.assertGreaterEqual(rect.y, panel.y, tag)
        test.assertLessEqual(rect.right, panel.right, tag)
        test.assertLessEqual(rect.bottom, panel.bottom, tag)
        test.assertGreater(rect.h, 0, tag)
        test.assertGreater(rect.w, 0, tag)
        rects.append(rect)
    for i, left in enumerate(rects):
        for right in rects[i + 1 :]:
            test.assertFalse(left.colliderect(right), "%s overlaps %s" % (left, right))


def _assert_numbers_whole(test, mon, panel, rows) -> None:
    import pygame

    panel = pygame.Rect(panel)
    for x, y, text, _size, cell, height in mon._fixed_rows:
        if not panel.collidepoint(x + 1, y + 1):
            continue
        test.assertLessEqual(x + len(text) * cell, panel.right)
        test.assertLessEqual(y + height, panel.bottom)
        owners = [rect for rect, _tag in rows if rect.collidepoint(x + 1, y + 1)]
        test.assertTrue(owners, text)
        test.assertGreaterEqual(owners[0].h, height)


class BrainPanelTests(unittest.TestCase):
    def test_rows_do_not_overlap_at_the_reported_sizes(self):
        import pygame

        from sim.train_monitor import TrainMonitor
        from sim.ui_theme import apply_theme

        pygame.init()
        mon = TrainMonitor("panel")
        view = _populated_view()
        out = __import__("pathlib").Path("/opt/cursor/artifacts")
        out.mkdir(parents=True, exist_ok=True)
        try:
            for ww, wh in ((1920, 1017), (1440, 900)):
                mon._on_resize(ww, wh)
                mon.draw(view)
                content = mon._content_of("brain")
                self.assertIsNotNone(content)
                panel = pygame.Rect(content.x + 4, content.y + 4, content.w - 8, content.h - 8)
                rows = list(mon._brain_rows)
                tags = [tag for _rect, tag in rows]
                _assert_packed(self, rows, panel)
                _assert_numbers_whole(self, mon, panel, rows)
                self.assertIn("progress-title", tags)
                self.assertIn("progress-0", tags)
                self.assertIn("progress-1", tags)
                self.assertIn("progress-2", tags)
                self.assertIn("drift", tags)
                self.assertIn("dan-title", tags)
                self.assertIn("dan-span", tags)
                self.assertGreaterEqual(panel.h, 120)

            narrow = pygame.Surface((280, 420))
            box = pygame.Rect(0, 0, 280, 420)
            mon._fixed_rows = []
            mon._brain_rows = []
            mon._draw_mb(narrow, view, box)
            _assert_packed(self, mon._brain_rows, box)
            _assert_numbers_whole(self, mon, box, mon._brain_rows)
            self.assertTrue(mon._brain_rows)

            short = pygame.Surface((640, 150))
            low = pygame.Rect(0, 0, 640, 150)
            mon._fixed_rows = []
            mon._brain_rows = []
            mon._draw_mb(short, view, low)
            _assert_packed(self, mon._brain_rows, low)
            _assert_numbers_whole(self, mon, low, mon._brain_rows)
            self.assertNotIn("dan-span", [tag for _rect, tag in mon._brain_rows])

            mon._on_resize(1920, 1017)
            apply_theme("light")
            mon._bg = None
            mon._win_bg = None
            mon.draw(view)
            pygame.image.save(mon.present_into((1920, 1017)), str(out / "trainer_brain_panel_light_1920x1017.png"))
            panel_rect = mon.panel_rect.clip(mon.screen.get_rect())
            pygame.image.save(mon.screen.subsurface(panel_rect).copy(), str(out / "trainer_brain_panel_light_crop.png"))
            apply_theme("dark")
            mon._bg = None
            mon._win_bg = None
            mon.draw(view)
            pygame.image.save(mon.present_into((1920, 1017)), str(out / "trainer_brain_panel_dark_1920x1017.png"))
            panel_rect = mon.panel_rect.clip(mon.screen.get_rect())
            pygame.image.save(mon.screen.subsurface(panel_rect).copy(), str(out / "trainer_brain_panel_dark_crop.png"))
            rows = list(mon._brain_rows)
            content = mon._content_of("brain")
            panel = pygame.Rect(content.x + 4, content.y + 4, content.w - 8, content.h - 8)
            _assert_packed(self, rows, panel)
        finally:
            apply_theme("light")
            pygame.quit()

    def test_readout_glyphs_stay_inside_their_block(self):
        """Labels and numbers do not cover each other or leave the block."""
        import pygame

        from sim.dock_layout import Leaf, Split
        from sim.train_monitor import TrainMonitor
        from sim.ui_theme import apply_theme

        pygame.init()
        mon = TrainMonitor("glyphs")
        view = _populated_view()
        out = __import__("pathlib").Path("/opt/cursor/artifacts")
        out.mkdir(parents=True, exist_ok=True)

        def blocks():
            found = {}
            for name in ("camera", "lidar", "brain", "journal", "controls"):
                rect = mon._content_of(name)
                if rect is not None:
                    found[name] = pygame.Rect(rect)
            return found

        def assert_glyphs(where: str) -> None:
            owned = blocks()
            rects = []
            for rect, group in mon._glyph_rects:
                self.assertGreater(rect.w, 0, where)
                self.assertGreater(rect.h, 0, where)
                if group not in owned:
                    continue
                host = owned[group]
                self.assertTrue(host.contains(rect), "%s %s %s outside %s" % (where, group, rect, host))
                rects.append((rect, group))
            for i, (left, left_group) in enumerate(rects):
                for right, right_group in rects[i + 1 :]:
                    self.assertFalse(
                        left.colliderect(right),
                        "%s %s %s overlaps %s %s" % (where, left_group, left, right_group, right),
                    )

        def narrow_tree():
            return Split(
                "v",
                0.88,
                Split(
                    "h",
                    0.22,
                    Split("v", 0.38, Leaf("camera"), Leaf("lidar")),
                    Split("h", 0.30, Leaf("brain"), Leaf("journal")),
                ),
                Leaf("controls"),
            )

        try:
            for theme in ("light", "dark"):
                apply_theme(theme)
                mon._bg = None
                mon._win_bg = None
                mon.dock = __import__("sim.dock_layout", fromlist=["default_tree"]).default_tree()
                for ww, wh in ((1920, 1017), (1440, 900)):
                    mon._on_resize(ww, wh)
                    mon._apply_layout(mon.screen.get_height())
                    mon.draw(view)
                    assert_glyphs("%s %dx%d" % (theme, ww, wh))
                    if (ww, wh) == (1920, 1017) and mon.lid_inner is not None:
                        self.assertEqual(mon.lid_inner.w, mon.lid_inner.h)
                        self.assertLessEqual(mon.lid_inner.right, mon.lidar_column_rect.x)
                        labels = []
                        for rect, group in mon._glyph_rects:
                            if group != "lidar" or not mon.lidar_column_rect.collidepoint(rect.x + 1, rect.y + 1):
                                continue
                            self.assertFalse(rect.colliderect(mon.lid_inner), rect)
                            labels.append(rect)
                        self.assertGreaterEqual(len(labels), 8)
                        scr = mon.screen
                        pygame.image.save(
                            scr.subsurface(mon.lid_rect.clip(scr.get_rect())),
                            str(out / ("trainer_lidar_%s_1920x1017.png" % theme)),
                        )
                    shown = mon.present_into((ww, wh))
                    pygame.image.save(shown, str(out / ("trainer_text_%s_%dx%d.png" % (theme, ww, wh))))
                mon.dock = narrow_tree()
                mon._on_resize(1920, 1017)
                mon._apply_layout(mon.screen.get_height())
                mon.draw(view)
                assert_glyphs("%s narrow" % theme)
                lid = mon._content_of("lidar")
                brain = mon._content_of("brain")
                self.assertLess(lid.w, 460, lid)
                self.assertLess(brain.w, 460, brain)
                pygame.image.save(
                    mon.present_into((1920, 1017)),
                    str(out / ("trainer_text_%s_narrow.png" % theme)),
                )
                scr = mon.screen
                pygame.image.save(scr.subsurface(mon.lid_rect.clip(scr.get_rect())), str(out / ("trainer_lidar_%s_narrow.png" % theme)))
                pygame.image.save(scr.subsurface(mon.panel_rect.clip(scr.get_rect())), str(out / ("trainer_brain_%s_narrow.png" % theme)))
        finally:
            apply_theme("light")
            pygame.quit()

    def test_zoom_stress_and_paint_throttle(self):
        from sim.mb_flywire import gl_frame_size_ok, gl_read_length_ok, gl_sort_due, stress_zoom

        due, key = gl_sort_due(None, None, 0.2, 0.1, 10.0)
        self.assertTrue(due)
        again, _same = gl_sort_due(key, 10.0, 0.2, -0.4, 10.01)
        self.assertFalse(again)
        soon, _moved = gl_sort_due(key, 10.0, 0.8, 0.1, 10.05)
        self.assertFalse(soon)
        later, _ready = gl_sort_due(key, 10.0, 0.8, 0.1, 10.2)
        self.assertTrue(later)
        self.assertTrue(gl_read_length_ok(30, 2, 5))
        self.assertFalse(gl_read_length_ok(31, 2, 5))
        self.assertFalse(gl_frame_size_ok(0, 10))
        self.assertFalse(gl_frame_size_ok(8, -1))
        self.assertTrue(gl_frame_size_ok(1, 1))

        stats = stress_zoom()
        self.assertEqual(stats["frames"], 60)
        self.assertEqual(stats["errors"], 0)
        self.assertGreater(stats["rejected"], 0)
        self.assertTrue(stats["resize_kept"])
        self.assertEqual(stats["gl"] + stats["software"], 60 - stats["rejected"])
        self.assertFalse(stats["worker_gl"])
        self.assertIsNot(stats["worker_paint"], True)
        if stats["gl"]:
            self.assertLessEqual(stats["ibo_writes"], 1)
        else:
            self.assertEqual(stats["ibo_writes"], 0)
            self.assertGreater(stats["software"], 0)

    def test_crash_log_is_outside_the_bundle(self):
        import faulthandler
        from pathlib import Path

        from sim.ui_settings import crash_log_path, install_crash_log

        previous = os.environ.get("RECOGNIZER_UI_CONFIG")
        os.environ.pop("RECOGNIZER_UI_CONFIG", None)
        try:
            path = crash_log_path()
            self.assertEqual(path.name, "crash.log")
            self.assertIn("recognize_trainer", path.parts)
            self.assertNotIn("_internal", path.parts)
            self.assertNotEqual(path.suffix, ".npz")
            root = Path("/tmp/recog_crash_test")
            os.environ["RECOGNIZER_UI_CONFIG"] = str(root / "window.json")
            install_crash_log.handle = None
            written = install_crash_log()
            self.assertEqual(written, root / "crash.log")
            self.assertTrue(written.is_file())
            self.assertNotIn("_internal", written.parts)
            self.assertTrue(faulthandler.is_enabled())
            from sim.ui_settings import ensure_stdio, trainer_log_path

            import sys

            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = None
            sys.stderr = None
            try:
                log_path = ensure_stdio()
                self.assertEqual(log_path, trainer_log_path())
                self.assertEqual(log_path, root / "trainer.log")
                print("trainer-line")
                print("err-line", file=sys.stderr)
                sys.stdout.flush()
                logged = (root / "trainer.log").read_text(encoding="utf-8")
                self.assertIn("trainer-line", logged)
                self.assertIn("err-line", logged)
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                leaked = getattr(ensure_stdio, "handle", None)
                if leaked is not None and leaked not in (old_out, old_err):
                    try:
                        leaked.close()
                    except OSError:
                        pass
                    ensure_stdio.handle = None
            bat = Path(__file__).resolve().parents[1].joinpath("rebuild_exe.bat").read_text(encoding="utf-8")
            self.assertIn("--noconsole", bat)
        finally:
            faulthandler.disable()
            handle = install_crash_log.handle
            install_crash_log.handle = None
            if handle is not None:
                handle.close()
            if previous is None:
                os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            else:
                os.environ["RECOGNIZER_UI_CONFIG"] = previous


if __name__ == "__main__":
    unittest.main()
