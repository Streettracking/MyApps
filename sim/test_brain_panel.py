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


_GRAPH_TAGS = {"plot", "drift-plot", "dan-axis"}


def _assert_text_clear_of_graphs(test, rows, require_graphs: bool = False) -> None:
    graphs = [(rect, tag) for rect, tag in rows if tag in _GRAPH_TAGS]
    texts = [(rect, tag) for rect, tag in rows if tag not in _GRAPH_TAGS and tag != "kc-bars"]
    if require_graphs:
        test.assertTrue(graphs, [tag for _rect, tag in rows])
        test.assertIn("plot-label", [tag for _rect, tag in rows])
    for text, text_tag in texts:
        for graph, graph_tag in graphs:
            test.assertFalse(
                text.colliderect(graph),
                "%s %s hits %s %s" % (text_tag, text, graph_tag, graph),
            )
    labels = [rect for rect, tag in rows if tag == "plot-label"]
    plots = [rect for rect, tag in rows if tag == "plot"]
    test.assertEqual(len(labels), len(plots))
    for label, plot in zip(labels, plots):
        test.assertLessEqual(label.bottom, plot.y, (label, plot))


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

    def test_plot_captions_stay_above_the_graphs_at_every_block_size(self):
        """Text rects miss the curves, the drift lines, and the DAN scale."""
        import pygame

        from sim.dock_layout import find_leaf
        from sim.train_monitor import TrainMonitor
        from sim.ui_theme import apply_theme

        pygame.init()
        mon = TrainMonitor("plot-clearance")
        view = _populated_view()
        quiet = _populated_view()
        quiet.dan_events = []
        try:
            apply_theme("dark")
            mon._bg = None
            mon._win_bg = None
            for ww, wh in ((1920, 1017), (1920, 1400), (1440, 900)):
                mon.dock = __import__("sim.dock_layout", fromlist=["default_tree"]).default_tree()
                mon._on_resize(ww, wh)
                mon.draw(view)
                content = mon._content_of("brain")
                panel = pygame.Rect(content.x + 4, content.y + 4, content.w - 8, content.h - 8)
                _assert_packed(self, mon._brain_rows, panel)
                _assert_text_clear_of_graphs(self, mon._brain_rows, require_graphs=False)

                find_leaf(mon.dock, "journal").visible = False
                mon._apply_layout(mon.screen.get_height())
                mon.draw(quiet)
                content = mon._content_of("brain")
                self.assertIsNone(mon._content_of("journal"))
                self.assertGreater(content.h, 400, (ww, wh, content))
                panel = pygame.Rect(content.x + 4, content.y + 4, content.w - 8, content.h - 8)
                tags = [tag for _rect, tag in mon._brain_rows]
                self.assertIn("dan-hint", tags)
                _assert_packed(self, mon._brain_rows, panel)
                _assert_text_clear_of_graphs(self, mon._brain_rows, require_graphs=True)

            sizes = (
                (220, 220),
                (280, 360),
                (360, 480),
                (480, 640),
                (640, 150),
                (720, 900),
                (900, 1200),
                (1400, 1600),
            )
            for width, height in sizes:
                surface = pygame.Surface((width, height))
                box = pygame.Rect(0, 0, width, height)
                for sample in (view, quiet):
                    mon._fixed_rows = []
                    mon._brain_rows = []
                    mon._draw_mb(surface, sample, box)
                    _assert_packed(self, mon._brain_rows, box)
                    _assert_text_clear_of_graphs(self, mon._brain_rows, require_graphs=height >= 480)
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
            self.assertIn("Desktop\\FlyWire-Go2", bat)
            self.assertIn("--onedir", bat)
            self.assertNotIn("--onefile", bat)
            self.assertIn("/XD logs", bat)
            self.assertIn("/XF *.vbs *.lnk *.cmd", bat)
            self.assertIn("launchers\\*.vbs", bat)
            self.assertIn("serve.py", bat)
            self.assertIn("start_teacher.vbs", bat)
            self.assertIn("start_teacher.cmd", bat)
            self.assertIn("tools\\gui_exe_entry.py", bat)
            self.assertIn("tools\\recognize_trainer_entry.py", bat)
            sim_lines = [line for line in bat.splitlines() if "--name FlyWireGo2Sim" in line]
            trainer_lines = [line for line in bat.splitlines() if "--name DogRecognizeTrainer" in line]
            self.assertEqual(len(sim_lines), 1)
            self.assertEqual(len(trainer_lines), 1)
            self.assertIn("--console", sim_lines[0])
            self.assertNotIn("--windowed", sim_lines[0])
            self.assertIn("--windowed", trainer_lines[0])
            self.assertNotIn("--console", trainer_lines[0])
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


    def test_numbers_share_a_baseline_with_their_labels(self):
        import pygame

        from sim.tabnum import baseline_shift, load_font, paint_slots
        from sim.train_monitor import TrainMonitor

        pygame.init()
        mon = TrainMonitor("baseline")
        try:
            mon._on_resize(1920, 1017)
            mon.draw(_populated_view())
            mono = load_font(14)
            ui_dy, mono_dy = baseline_shift(mon.font_sm, mono)
            glyphs = []
            surf = pygame.Surface((480, 48), pygame.SRCALPHA)
            paint_slots(
                surf,
                8,
                6,
                [("ui", "PAM"), ("num", "+142"), ("ui", "R_L"), ("num", "+353")],
                mon.font_sm,
                (255, 255, 255),
                (220, 220, 220),
                glyphs=glyphs,
            )
            self.assertGreaterEqual(len(glyphs), 2)
            label = glyphs[0]
            number = glyphs[1]
            self.assertEqual(label.y, 6 + ui_dy)
            self.assertEqual(number.y, 6 + mono_dy)
            self.assertEqual(label.y + mon.font_sm.get_ascent(), number.y + mono.get_ascent())
        finally:
            pygame.quit()


def _recorded_status():
    """A /status body the way the onboard brain publishes the panel fields."""
    return {
        "dog_in_frame": 3.0,
        "dog_in_frame_series": [[0.0, 1.0], [1.0, 3.0]],
        "no_dog": 0.0,
        "no_dog_series": [[0.5, 0.0]],
        "kc_mbon_drift": 4.25,
        "kc_mbon_drift_series": [[0.0, 1.0], [1.0, 4.25]],
        "treats_session": 3,
        "treats_total": 11,
        "kc_bins": [0.0] * 47 + [1.0],
        "kc_on": 12,
        "kc_n": 5137,
        "pam_l": 2,
        "pam_r": 1,
        "ppl1_l": 0,
        "ppl1_r": 4,
        "n_pam": 99,
        "drift": 0.0,
        "n_ppl1": 4,
        "n_ppl1_total": 9,
        "dan_events": [[1.0, "PAM"], [2.0, "PPL1"]],
    }


class _StatusLink:
    def __init__(self, age, status):
        self._age = age
        self.status = status

    def status_age(self):
        return self._age

    def status_line(self, bridge_open=False):
        return "борт"


class BrainPanelStatusTests(unittest.TestCase):
    def test_recorded_status_fills_the_same_panel_fields_as_sim(self):
        from sim.brain_panel import BRAIN_STATUS_STALE_S, brain_panel_from_status
        from sim.yolo_teacher import counts_line

        self.assertEqual(BRAIN_STATUS_STALE_S, 2.0)
        ok, fields = brain_panel_from_status(_recorded_status(), BRAIN_STATUS_STALE_S)
        self.assertTrue(ok)
        self.assertEqual(fields["drift"], 4.25)
        self.assertEqual(fields["drift_curve"], [(0.0, 1.0), (1.0, 4.25)])
        self.assertEqual(fields["n_pam"], 3)
        self.assertEqual(fields["total_pam"], 11)
        self.assertEqual(fields["n_ppl1"], 4)
        self.assertEqual(fields["total_ppl1"], 9)
        self.assertTrue(fields["lifetime_known"])
        self.assertEqual(fields["kc_on"], 12)
        self.assertEqual(fields["kc_n"], 5137)
        self.assertEqual(len(fields["kc_bins"]), 48)
        self.assertAlmostEqual(float(fields["kc_bins"][-1]), 1.0)
        self.assertEqual(fields["peer_curve"], [(0.0, 1.0 / 3.0), (1.0, 1.0)])
        self.assertEqual(fields["other_curve"], [(0.5, 0.0)])
        self.assertEqual(fields["teacher_counts"], counts_line(2, 1, 0, 4))
        self.assertEqual(fields["dan_events"], [(1.0, "PAM"), (2.0, "PPL1")])

    def test_stale_or_missing_status_drops_the_numbers(self):
        from sim.brain_panel import (
            BRAIN_OFFLINE_TITLE,
            BRAIN_STATUS_STALE_S,
            brain_header,
            brain_panel_from_status,
        )

        recorded = _recorded_status()
        ok, fields = brain_panel_from_status(recorded, BRAIN_STATUS_STALE_S + 0.01)
        self.assertFalse(ok)
        self.assertEqual(fields, {})
        self.assertEqual(brain_header(True), BRAIN_OFFLINE_TITLE)
        self.assertEqual(BRAIN_OFFLINE_TITLE, "Мозг на .213 не запущен")
        for broken in (None, {}, {"drift": 1.0}):
            ok, fields = brain_panel_from_status(broken, 0.1)
            self.assertFalse(ok)
            self.assertEqual(fields, {})
        missing = dict(recorded)
        del missing["kc_bins"]
        ok, fields = brain_panel_from_status(missing, 0.1)
        self.assertFalse(ok)
        self.assertEqual(fields, {})
        ok, _fields = brain_panel_from_status(recorded, None)
        self.assertFalse(ok)

    def test_onboard_view_uses_the_status_and_greys_out_when_stale(self):
        import pygame

        from pathlib import Path

        from sim.brain_panel import BRAIN_STATUS_STALE_S, offline_plate
        from sim.recognize_train_live import LiveSession, _live_view
        from sim.train_monitor import TrainMonitor, _scheme_rgb

        session = LiveSession(Path("/tmp/mb_panel_view.npz"), onboard=True)
        recorded = _recorded_status()
        session.remote = recorded
        live = _live_view(session, None, None, "", None, True, "—", _StatusLink(0.2, recorded))
        self.assertFalse(live.brain_offline)
        self.assertEqual(live.n_pam, 3)
        self.assertEqual(live.total_pam, 11)
        self.assertEqual(live.drift, 4.25)
        self.assertEqual(live.kc_on, 12)
        self.assertEqual(live.peer_curve[-1], (1.0, 1.0))
        self.assertIn("PAM_L", live.teacher_counts)
        self.assertIn("PPL1_R", live.teacher_counts)

        stale = _live_view(
            session,
            None,
            None,
            "",
            None,
            True,
            "—",
            _StatusLink(BRAIN_STATUS_STALE_S + 0.5, recorded),
        )
        self.assertTrue(stale.brain_offline)
        self.assertEqual(stale.n_pam, 0)
        self.assertEqual(stale.peer_curve, [])
        self.assertEqual(stale.drift_curve, [])
        self.assertEqual(stale.teacher_counts, "")
        self.assertIsNone(stale.kc_bins)

        pygame.init()
        try:
            mon = TrainMonitor("offline-brain")
            mon._on_resize(1920, 1017)
            mon.draw(stale)
            content = mon._content_of("brain")
            plate = offline_plate(_scheme_rgb("card"), _scheme_rgb("ink_dim"))
            color = mon.screen.get_at((content.centerx, content.centery))
            self.assertEqual(tuple(color[:3]), plate)
            self.assertNotEqual(plate, _scheme_rgb("card"))
        finally:
            pygame.quit()

    def test_onboard_status_is_json_and_splits_dog_from_empty(self):
        import json

        import numpy as np

        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.brain_panel import BRAIN_PANEL_KEYS, brain_panel_from_status
        from sim.mb_train import MbTrainer, default_npz

        brain = MbTrainer(default_npz(), seed=1)
        loop = BrainLoop(SportDrive(None), brain, "/tmp/mb_panel_status.npz")
        camera = np.zeros((64, 96, 3), dtype=np.uint8)
        lidar = np.zeros_like(camera)
        loop.ingest(camera, lidar, 1.0)
        quiet = json.loads(json.dumps(loop.status()))
        for key in BRAIN_PANEL_KEYS:
            self.assertIn(key, quiet)
        self.assertIsNone(quiet["dog_in_frame"])
        self.assertEqual(quiet["dog_in_frame_series"], [])
        self.assertEqual(len(quiet["no_dog_series"]), 1)
        self.assertEqual(quiet["no_dog"], quiet["no_dog_series"][0][1])
        self.assertEqual(len(quiet["kc_bins"]), 48)
        self.assertIsInstance(quiet["kc_bins"][0], float)
        self.assertIsInstance(quiet["kc_mbon_drift"], float)
        self.assertIsInstance(quiet["treats_session"], int)
        self.assertIsInstance(quiet["treats_total"], int)
        loop.handle(
            {"op": "teacher_aim", "yolo_l": [0.9, 0.2, 0.4, 0.5], "weight": 0.0},
            2.0,
        )
        loop.ingest(camera, lidar, 2.0)
        seen = json.loads(json.dumps(loop.status()))
        self.assertEqual(len(seen["dog_in_frame_series"]), 1)
        self.assertEqual(seen["dog_in_frame"], seen["dog_in_frame_series"][0][1])
        self.assertEqual(len(seen["no_dog_series"]), 1)
        ok, fields = brain_panel_from_status(seen, 0.0)
        self.assertTrue(ok)
        self.assertEqual(fields["n_pam"], seen["treats_session"])
        self.assertEqual(fields["total_pam"], seen["treats_total"])
        self.assertEqual(fields["kc_on"], seen["kc_on"])
        self.assertEqual(fields["kc_n"], seen["kc_n"])
        self.assertEqual(len(fields["peer_curve"]), 1)
        self.assertEqual(len(fields["other_curve"]), 1)


if __name__ == "__main__":
    unittest.main()
