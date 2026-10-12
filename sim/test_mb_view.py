"""Windowed trainer and the separate 3D mushroom-body view.

Run: python3 -m unittest sim.test_mb_view
"""

from __future__ import annotations

import json
import os
import socket
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class WindowTests(unittest.TestCase):
    def test_opens_resizable_window_at_about_eighty_percent(self):
        import pygame

        from sim.train_monitor import WIN_H, WIN_W, TrainMonitor

        pygame.init()
        mon = TrainMonitor("window")
        self.assertFalse(mon.fullscreen)
        self.assertEqual(mon.screen.get_size(), (WIN_W, WIN_H))
        dw, dh = mon._desktop()
        ww, wh = mon.window.get_size()
        self.assertLess(ww, dw)
        self.assertLess(wh, dh)
        self.assertAlmostEqual(ww, min(max(640, int(round(dw * 0.80))), dw - 16), delta=2)
        self.assertAlmostEqual(wh, min(max(480, int(round(dh * 0.80))), dh - 48), delta=2)
        pygame.quit()

    def test_f11_returns_to_the_same_window(self):
        import pygame

        from sim.train_monitor import WIN_H, WIN_W, TrainMonitor

        pygame.init()
        mon = TrainMonitor("window")
        size = mon.window.get_size()
        self.assertTrue(mon.toggle_fullscreen())
        self.assertTrue(mon.fullscreen)
        self.assertEqual(mon.screen.get_size(), (WIN_W, WIN_H))
        self.assertFalse(mon.toggle_fullscreen())
        self.assertFalse(mon.fullscreen)
        self.assertEqual(mon.window.get_size(), size)
        self.assertEqual(mon.screen.get_size(), (WIN_W, WIN_H))
        pygame.quit()

    def test_resize_and_minimize_do_not_quit_or_drop_the_poll(self):
        import pygame

        from sim.train_monitor import MonitorView, TrainMonitor

        pygame.init()
        mon = TrainMonitor("window")
        polls = []

        def status():
            polls.append(1)
            return {"kc_l": [1], "r_l": 1.0, "r_r": 0.0}

        mon._on_resize(700, 500)
        self.assertEqual(mon.window.get_size(), (700, 500))
        self.assertEqual(mon.screen.get_size(), (1920, 1080))
        self.assertFalse(mon.fullscreen)
        pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=0, h=0, size=(0, 0)))
        inp = mon.pump()
        self.assertFalse(inp.quit)
        self.assertTrue(mon._minimized)
        status()
        mon.draw(MonitorView(learner="mb", onboard=True))
        restored = getattr(pygame, "WINDOWRESTORED", None)
        if restored is not None:
            pygame.event.post(pygame.event.Event(restored))
            inp = mon.pump()
            self.assertFalse(inp.quit)
            self.assertFalse(mon._minimized)
        minimized = getattr(pygame, "WINDOWMINIMIZED", None)
        if minimized is not None:
            pygame.event.post(pygame.event.Event(minimized))
            inp = mon.pump()
            self.assertFalse(inp.quit)
            self.assertTrue(mon._minimized)
            status()
            mon.draw(MonitorView(learner="mb", onboard=True))
        self.assertGreaterEqual(len(polls), 1)
        pygame.quit()

    def test_mouse_maps_into_the_logical_canvas_and_j_is_the_brain(self):
        import pygame

        from sim.recognize_train import SIM_KEYS
        from sim.recognize_train_live import LIVE_KEYS
        from sim.train_monitor import TrainMonitor, window_from_logical

        self.assertIn("J мозг", SIM_KEYS)
        self.assertIn("J мозг", LIVE_KEYS)
        self.assertIn(" B", SIM_KEYS)
        pygame.init()
        mon = TrainMonitor("window")
        mon._on_resize(1920, 1080)
        ww, wh = mon.window.get_size()
        logical = mon._logical_pos((ww // 2, wh // 2))
        self.assertAlmostEqual(logical[0], 960, delta=2)
        self.assertAlmostEqual(logical[1], 540, delta=2)
        wx, wy = window_from_logical(mon.brain_rect.center, ww, wh)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(wx, wy), button=1))
        clicked = mon.pump()
        self.assertTrue(clicked.brain_toggle)
        self.assertFalse(clicked.beep_toggle)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0, unicode="j", scancode=0))
        keyed = mon.pump()
        self.assertTrue(keyed.brain_toggle)
        self.assertFalse(keyed.beep_toggle)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0, unicode="b", scancode=0))
        beep = mon.pump()
        self.assertTrue(beep.beep_toggle)
        self.assertFalse(beep.brain_toggle)
        pygame.quit()

    def test_width_sets_scale_and_short_windows_scroll(self):
        import pygame
        import numpy as np

        from sim.train_monitor import (
            MIN_CONTENT_W,
            MIN_SCALE,
            SCROLL_THUMB,
            WIN_H,
            WIN_W,
            MonitorView,
            TrainMonitor,
            fit_window,
            logical_from_window,
            scroll_limits,
            window_from_logical,
        )

        scale, lw, lh, cw, ch = fit_window(WIN_W, WIN_H)
        self.assertEqual((scale, lw, lh, cw, ch), (1.0, WIN_W, WIN_H, WIN_W, WIN_H))
        self.assertEqual(scroll_limits(WIN_W, WIN_H), (0, 0))

        scale, _lw, lh, cw, ch = fit_window(1800, 1400)
        self.assertAlmostEqual(scale, 1800 / float(WIN_W), places=6)
        self.assertEqual(cw, 1800)
        self.assertEqual(ch, 1400)
        self.assertGreater(lh, WIN_H)
        self.assertEqual(scroll_limits(1800, 1400), (0, 0))

        scale, _lw, lh, cw, ch = fit_window(1000, 700)
        self.assertEqual(scale, MIN_SCALE)
        self.assertEqual(cw, MIN_CONTENT_W)
        self.assertEqual(lh, WIN_H)
        self.assertGreater(cw, 1000)
        self.assertGreater(ch, 700)
        max_x, max_y = scroll_limits(1000, 700)
        self.assertGreater(max_x, 0)
        self.assertGreater(max_y, 0)
        shifted = logical_from_window((20, 30), 1000, 700, (300, 80))
        self.assertAlmostEqual(shifted[0], round((20 + 300) / MIN_SCALE), delta=1)
        self.assertAlmostEqual(shifted[1], round((30 + 80) / MIN_SCALE), delta=1)
        self.assertEqual(logical_from_window((1000 - 4, 40), 1000, 700, (0, 0)), (-1, -1))

        pygame.init()
        mon = TrainMonitor("fit")
        camera = np.zeros((90, 160, 3), dtype=np.uint8)
        camera[:] = (20, 24, 32)
        lidar = np.zeros((80, 80, 3), dtype=np.uint8)
        lidar[:] = (12, 14, 18)
        yy, xx = np.ogrid[:80, :80]
        lidar[np.abs(np.hypot(xx - 39.5, yy - 39.5) - 28) < 1.6] = (0, 40, 255)
        logs = ["  %d.0s  строка журнала %d" % (i, i) for i in range(1, 28)]
        view = MonitorView(
            camera=camera,
            lidar=lidar,
            learner="mb",
            lidar_mode="карта",
            log_lines=logs,
            learning_on=True,
            eye_l_recognized=True,
            teacher_flash_l="ppl1",
            teacher_flash_r="pam",
            title="recognition training",
        )
        mon._on_resize(1920, 1080)
        mon.draw(view)
        base_cam = (mon.cam_rect.w, mon.cam_rect.h)
        base_lid = mon.lid_rect.h
        base_journal = mon.journal_rect.h
        base_rows = mon.log_rows
        self.assertIsNotNone(mon.cam_inner)
        self.assertAlmostEqual(mon.cam_inner.w / float(mon.cam_inner.h), 160 / 90.0, delta=0.03)
        self.assertAlmostEqual(mon.lid_inner.w, mon.lid_inner.h, delta=1)

        mon._on_resize(1440, 1600)
        mon.draw(view)
        self.assertEqual((mon.cam_rect.w, mon.cam_rect.h), base_cam)
        self.assertGreater(mon.lid_rect.h, base_lid + 80)
        self.assertGreater(mon.panel_rect.h, 900)
        self.assertGreater(mon.journal_rect.h, base_journal + 40)
        self.assertGreater(mon.log_rows, base_rows)
        self.assertEqual(mon.screen.get_size()[0], WIN_W)
        self.assertGreater(mon.screen.get_size()[1], WIN_H)
        tall = pygame.surfarray.array3d(mon.present_into((1440, 1600)))
        self.assertGreater(int((tall[:, :40].max(axis=2) <= 90).sum()), 30)
        self.assertAlmostEqual(mon.lid_inner.w, mon.lid_inner.h, delta=1)
        blue = (tall[:, :, 2] > 180) & (tall[:, :, 0] < 50) & (tall[:, :, 1] < 90)
        xs, ys = np.where(blue)
        self.assertGreater(len(xs), 20)
        self.assertAlmostEqual(int(xs.max() - xs.min()), int(ys.max() - ys.min()), delta=4)
        out = __import__("pathlib").Path("/opt/cursor/artifacts")
        out.mkdir(parents=True, exist_ok=True)
        pygame.image.save(mon.present_into((1440, 1600)), str(out / "trainer_window_tall.png"))

        mon._on_resize(1800, 1100)
        mon.draw(view)
        pygame.image.save(mon.present_into((1800, 1100)), str(out / "trainer_window_wide.png"))
        wide_top = pygame.surfarray.array3d(mon.present_into((1800, 1100)))
        self.assertGreater(int((wide_top[:, :40].max(axis=2) <= 90).sum()), 30)
        self.assertEqual(scroll_limits(1800, 1100), (0, 0))

        mon._on_resize(800, 600)
        mon.scroll_x = 0
        mon.scroll_y = 0
        mon.draw(view)
        wx, wy = window_from_logical(mon.brain_rect.center, 800, 600, (0, 0))
        self.assertGreaterEqual(wx, 0)
        self.assertLess(wx, 800)
        self.assertGreater(scroll_limits(800, 600)[0], 0)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(48, 18), button=1))
        missed = mon.pump()
        self.assertFalse(missed.brain_toggle)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(wx, wy), button=1))
        hit = mon.pump()
        self.assertTrue(hit.brain_toggle)

        mon._on_resize(1000, 700)
        mon.scroll_x = 0
        mon.scroll_y = 0
        mon.draw(view)
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1, precise_x=0.0, precise_y=-1.0))
        mon.pump()
        self.assertGreater(mon.scroll_y, 0)
        pygame.key.set_mods(pygame.KMOD_SHIFT)
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1, precise_x=0.0, precise_y=-1.0))
        mon.pump()
        pygame.key.set_mods(0)
        self.assertGreater(mon.scroll_x, 0)

        mon.scroll_x = 0
        mon.scroll_y = 0
        thumb = mon.scroll_geometry((1000, 700))["thumb_y"]
        self.assertIsNotNone(thumb)
        origin = thumb.center
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=origin, button=1))
        mon.pump()
        pygame.event.post(
            pygame.event.Event(pygame.MOUSEMOTION, pos=(origin[0], origin[1] + 70), rel=(0, 70), buttons=(1, 0, 0))
        )
        mon.pump()
        self.assertGreater(mon.scroll_y, 0)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(origin[0], origin[1] + 70), button=1))
        mon.pump()

        mon.scroll_x = 0
        mon.scroll_y = 0
        scroll_frame = mon.present_into((1000, 700))
        shown = pygame.surfarray.array3d(scroll_frame)
        geom = mon.scroll_geometry((1000, 700))
        thumb = geom["thumb_y"]
        pix = shown[thumb.centerx, thumb.centery]
        self.assertLess(int(np.abs(pix.astype(int) - np.array(SCROLL_THUMB)).max()), 8)
        thumb_x = geom["thumb_x"]
        self.assertIsNotNone(thumb_x)
        pix_x = shown[thumb_x.centerx, thumb_x.centery]
        self.assertLess(int(np.abs(pix_x.astype(int) - np.array(SCROLL_THUMB)).max()), 8)
        pygame.image.save(scroll_frame, str(out / "trainer_window_scroll.png"))
        pygame.quit()

    def test_short_client_pins_the_menu_and_the_keys(self):
        """A maximized laptop client keeps the top bar and the keys row on screen."""
        import pygame
        import numpy as np

        from sim.train_monitor import (
            FOOTER_H,
            MENU,
            MonitorView,
            TrainMonitor,
            scroll_limits,
            scrollbar_geom,
            window_from_logical,
        )

        pygame.init()
        mon = TrainMonitor("client")
        view = MonitorView(
            title="recognition training",
            udp_status="симулятор  команда сценарий",
            keys_hint="T/X учить A авто",
            learner="mb",
        )
        out = __import__("pathlib").Path("/opt/cursor/artifacts")
        out.mkdir(parents=True, exist_ok=True)
        menu = np.array(tuple(MENU), dtype=int)
        try:
            for ww, wh in ((1920, 1017), (1536, 801)):
                mon._on_resize(ww, wh)
                self.assertEqual(mon.window.get_size(), (ww, wh))
                max_x, max_y = scroll_limits(ww, wh)
                self.assertGreater(max_y, 0, (ww, wh))
                self.assertIsNotNone(scrollbar_geom(ww, wh, 0, 0)["thumb_y"])
                for scroll_y in (0, max_y):
                    mon.scroll_x = 0
                    mon.scroll_y = scroll_y
                    mon.draw(view)
                    fy = int(mon.layout["footer_y"])
                    self.assertEqual(window_from_logical((0, 0), ww, wh, (0, scroll_y))[1], 0)
                    for ly in (fy + 6, fy + 26, fy + 40, fy + FOOTER_H - 1):
                        _wx, wy = window_from_logical((16, ly), ww, wh, (0, scroll_y))
                        self.assertGreaterEqual(wy, 0, (ww, wh, scroll_y, ly))
                        self.assertLess(wy, wh, (ww, wh, scroll_y, ly))
                    frame = mon.present_into((ww, wh))
                    self.assertEqual(frame.get_size(), (ww, wh))
                    bar_x = window_from_logical((480, 0), ww, wh, (0, scroll_y))[0]
                    top = np.array(frame.get_at((bar_x, 0))[:3], dtype=int)
                    self.assertLess(int(np.abs(top - menu).max()), 5, (ww, wh, scroll_y))
                    row0 = pygame.surfarray.array3d(frame)[:, 0].max(axis=1)
                    self.assertGreater(int((row0 > 200).sum()), ww // 2, (ww, wh, scroll_y))
                    shown = pygame.surfarray.array3d(frame)
                    footer_px = wh - window_from_logical((16, fy), ww, wh, (0, scroll_y))[1]
                    band = shown[16:420, wh - max(1, footer_px) : wh]
                    ink = (band[:, :, 0] < 80) & (band[:, :, 1] < 80) & (band[:, :, 2] < 80)
                    self.assertGreater(int(ink.sum()), 12, (ww, wh, scroll_y))
                pygame.image.save(frame, str(out / ("trainer_client_%dx%d.png" % (ww, wh))))
        finally:
            pygame.quit()

    def test_maximize_trusts_the_live_surface(self):
        import pygame

        from sim.train_monitor import TrainMonitor

        pygame.init()
        mon = TrainMonitor("max")
        mon._on_resize(1280, 720)
        live = mon.window.get_size()
        calls = []
        real = pygame.display.set_mode

        def spy(size, flags=0):
            calls.append(tuple(size))
            return real(size, flags)

        pygame.display.set_mode = spy
        try:
            pygame.event.clear()
            pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=1920, h=1080, size=(1920, 1080)))
            pygame.event.post(pygame.event.Event(pygame.WINDOWMAXIMIZED))
            mon.pump()
            self.assertEqual(calls, [])
            self.assertEqual(mon.window.get_size(), live)
            self.assertTrue(mon._maximized)
            self.assertIs(mon.window, pygame.display.get_surface())
        finally:
            pygame.display.set_mode = real
            pygame.quit()


class ClientChromeTests(unittest.TestCase):
    def test_saved_geometry_is_clipped_and_dpi_is_set_before_the_window(self):
        from pathlib import Path

        from sim.win_dpi import clip_window_box, enable_dpi_awareness, work_area

        self.assertIsNone(work_area())
        enable_dpi_awareness()
        area = (0, 0, 1920, 1040)
        self.assertEqual(clip_window_box(10, 20, 800, 600, area), (10, 20, 800, 600))
        self.assertEqual(clip_window_box(-40, -10, 800, 600, area), (0, 0, 800, 600))
        self.assertEqual(clip_window_box(1500, 800, 800, 600, area), (1120, 440, 800, 600))
        self.assertEqual(clip_window_box(100, 100, 2000, 1200, area), (0, 0, 1920, 1040))
        self.assertEqual(clip_window_box(None, None, 4000, 3000, area), (None, None, 1920, 1040))
        self.assertEqual(clip_window_box(10, 20, 100, 50, None), (10, 20, 320, 240))

        dpi = Path("sim/win_dpi.py").read_text(encoding="utf-8")
        self.assertIn("SetProcessDpiAwareness", dpi)
        self.assertIn("SetProcessDPIAware", dpi)
        self.assertIn("SPI_GETWORKAREA", dpi)
        self.assertNotIn("set_mode", dpi)
        self.assertNotIn("from_display_module", dpi)

        monitor = Path("sim/train_monitor.py").read_text(encoding="utf-8")
        self.assertIn("enable_dpi_awareness()", monitor)
        self.assertLess(monitor.index("enable_dpi_awareness()"), monitor.index("pygame.init()"))
        self.assertIn("WINDOWSIZECHANGED", monitor)

    def test_dock_span_fills_the_visible_width_at_windows_dpi(self):
        from sim.train_monitor import MIN_CONTENT_W, WIN_W, dock_span, fit_window, scroll_limits
        from sim.win_dpi import dpi_ratio

        self.assertEqual(dpi_ratio(1024, 1024), 1.0)
        self.assertEqual(dpi_ratio(800, 800), 1.0)
        self.assertAlmostEqual(dpi_ratio(1024, int(round(1024 * 1.25))), 1.25)
        self.assertAlmostEqual(dpi_ratio(1024, int(round(1024 * 1.5))), 1.5)
        self.assertEqual(dpi_ratio(1024, 1100), 1.0)
        self.assertEqual(dpi_ratio(1536, 1024), 1.0)

        # An 800-wide window keeps the 1920 canvas and still scrolls.
        self.assertEqual(dock_span(800), WIN_W - 24)
        self.assertEqual(dock_span(800, 1.5), WIN_W - 24)
        self.assertGreater(scroll_limits(800, 600)[0], 0)
        self.assertEqual(dock_span(1920), WIN_W - 24)
        self.assertEqual(dock_span(MIN_CONTENT_W), WIN_W - 24)

        for dpi, ww in ((1.0, 1024), (1.25, 1024), (1.5, 1024), (1.0, 1200)):
            pixels = int(round(ww * dpi))
            span = dock_span(ww, dpi)
            _scale, _lw, _lh, content_w, _ch = fit_window(pixels, 768)
            present = content_w / float(WIN_W)
            right = (12 + span) * present
            self.assertGreaterEqual(span, 640)
            self.assertLessEqual(span, WIN_W - 12)
            self.assertAlmostEqual(right, pixels, delta=1.5, msg=(dpi, ww, span, right, pixels))
        self.assertIn("WINDOWMAXIMIZED", monitor)
        self.assertIn("get_surface", monitor)
        self.assertNotIn("from_display_module", monitor)


class BrainTests(unittest.TestCase):
    def test_soma_cloud_and_one_frame(self):
        import pygame

        from sim.mb_view3d import VIEW_PORT, BrainCloud, connectome_npz, demo_packet, render_frame

        self.assertNotIn(VIEW_PORT, (8088, 8090, 8091, 5451))
        path = connectome_npz()
        before = (path.stat().st_mtime_ns, path.stat().st_size)
        cloud = BrainCloud()
        self.assertEqual((path.stat().st_mtime_ns, path.stat().st_size), before)
        self.assertTrue(str(cloud.source).startswith("flywire-soma"))
        self.assertEqual(cloud.n_schematic, 0)
        self.assertGreater(cloud.n_kc, 1000)
        self.assertGreater(cloud.span_nm[0], cloud.span_nm[2] * 8)
        self.assertEqual(cloud.z_aniso, 10.0)
        left = cloud.kc_pos[cloud.kc_side == 1]
        right = cloud.kc_pos[cloud.kc_side == 0]
        self.assertGreater(len(left), 100)
        self.assertGreater(len(right), 100)
        self.assertLess(float(left[:, 0].mean()), float(right[:, 0].mean()))
        self.assertGreater(float(left[:, 1].std()), 0.15)
        import time

        pygame.init()
        started = time.perf_counter()
        frame = render_frame(cloud, demo_packet(cloud), size=(640, 480))
        self.assertLess(time.perf_counter() - started, 2.5)
        image = pygame.surfarray.array3d(frame)
        corner = image[2, 2].astype(int)
        self.assertLess(int(corner.mean()), 40)
        # The shell is hollow, so one centre pixel can be black. The middle of the frame is not.
        x0, x1 = int(image.shape[0] * 0.30), int(image.shape[0] * 0.70)
        y0, y1 = int(image.shape[1] * 0.30), int(image.shape[1] * 0.70)
        core = image[x0:x1, y0:y1]
        self.assertGreater(float(core.mean()), float(corner.mean()) + 2.0)
        warm = (image[:, :, 0] > 180) & (image[:, :, 0] > image[:, :, 1] + 15) & (image[:, :, 0] > image[:, :, 2])
        green = (image[:, :, 1] > 150) & (image[:, :, 1] > image[:, :, 0] + 12)
        self.assertGreater(int(warm.sum()), 8)
        self.assertGreater(int(green.sum()), 8)
        pygame.quit()

    def test_flywire_shell_matches_the_plaques(self):
        import pygame

        from sim.mb_flywire import (
            KIND_KC_AB,
            KIND_KC_G,
            KIND_MBON,
            KIND_PAM,
            KIND_PPL1,
            forward_arrow,
            load_scene,
            lobe_anchors,
        )
        from sim.mb_view3d import BrainCloud, demo_packet, render_frame

        scene = load_scene()
        self.assertIsNotNone(scene)
        self.assertGreater(scene.n_neurons, 1000)
        self.assertGreater(len(scene.faces), 500)
        self.assertGreater(len(scene.sk_edges), 1000)
        left = scene.vertices[scene.optic == 1]
        right = scene.vertices[scene.optic == 2]
        self.assertGreater(len(left), 50)
        self.assertGreater(len(right), 50)
        self.assertLess(float(left[:, 0].mean()), 0.0)
        self.assertGreater(float(right[:, 0].mean()), 0.0)
        anchors = {item["side"]: item for item in lobe_anchors(scene)}
        self.assertEqual(anchors[1]["label"], "зрит. доля Л")
        self.assertEqual(anchors[0]["label"], "зрит. доля П")
        self.assertLess(float(anchors[1]["pos"][0]), float(left[:, 0].mean()))
        self.assertGreater(float(anchors[0]["pos"][0]), float(right[:, 0].mean()))
        self.assertLess(float(anchors[1]["pos"][0]), float(anchors[0]["pos"][0]))
        start, tip = forward_arrow(scene)
        self.assertGreater(float(tip[2]), float(start[2]))
        kinds = set(int(k) for k in scene.neuron_kind.tolist())
        self.assertTrue({KIND_KC_G, KIND_KC_AB, KIND_MBON, KIND_PAM, KIND_PPL1} <= kinds)
        pygame.init()
        try:
            cloud = BrainCloud()
            lit = render_frame(cloud, demo_packet(cloud), size=(480, 360))
            dark = demo_packet(cloud)
            dark["rec_l"] = False
            dark["rec_r"] = False
            dark["kc_l"] = []
            dark["kc_r"] = []
            dark["flash_l"] = ""
            dark["flash_r"] = ""
            dim = render_frame(cloud, dark, size=(480, 360))
            self.assertNotEqual(int(pygame.surfarray.array3d(lit).sum()), int(pygame.surfarray.array3d(dim).sum()))
        finally:
            pygame.quit()

    def test_idle_orbit_does_not_spin(self):
        from sim.mb_view3d import Orbit

        orbit = Orbit()
        self.assertFalse(orbit.auto)
        start = orbit.yaw
        orbit.tick(2.0, False)
        self.assertEqual(orbit.yaw, start)
        orbit.drag(10, 0)
        self.assertGreater(orbit.yaw, start)
        held = orbit.yaw
        orbit.tick(1.0, False)
        self.assertEqual(orbit.yaw, held)

    def test_pitch_pan_and_reset(self):
        import math

        import numpy as np

        from sim.mb_view3d import DIST0, PAN_KEY, PITCH0, PITCH_LIM, YAW0, Orbit, _keys, _project

        orbit = Orbit()
        orbit.drag(0, 500)
        self.assertAlmostEqual(orbit.pitch, PITCH_LIM, places=4)
        self.assertGreater(PITCH_LIM, math.radians(80))
        orbit.drag(0, -1000)
        self.assertAlmostEqual(orbit.pitch, -PITCH_LIM, places=4)
        orbit.pan_pixels(80, -40)
        self.assertLess(orbit.pan_x, 0.0)
        self.assertLess(orbit.pan_y, 0.0)
        orbit.nudge(PAN_KEY, PAN_KEY)
        moved_x, moved_y = orbit.pan_x, orbit.pan_y
        orbit.reset()
        self.assertEqual(orbit.yaw, YAW0)
        self.assertEqual(orbit.pitch, PITCH0)
        self.assertEqual(orbit.dist, DIST0)
        self.assertEqual(orbit.pan_x, 0.0)
        self.assertEqual(orbit.pan_y, 0.0)
        self.assertNotEqual((moved_x, moved_y), (0.0, 0.0))

        class _Key:
            def __init__(self, key, repeat=False):
                self.key = key
                self.repeat = repeat

        import pygame

        pygame.init()
        try:
            held = Orbit()
            before = (held.pan_x, held.pan_y, held.yaw, held.pitch)
            for key in (
                pygame.K_a,
                pygame.K_d,
                pygame.K_w,
                pygame.K_s,
                pygame.K_LEFT,
                pygame.K_RIGHT,
                pygame.K_UP,
                pygame.K_DOWN,
            ):
                _keys(held, _Key(key))
                _keys(held, _Key(key, repeat=True))
            self.assertEqual((held.pan_x, held.pan_y, held.yaw, held.pitch), before)
            self.assertFalse(held.auto)
            _keys(held, _Key(pygame.K_PAGEUP))
            self.assertAlmostEqual(held.pan_y, PAN_KEY)
            _keys(held, _Key(pygame.K_PAGEDOWN))
            self.assertAlmostEqual(held.pan_y, 0.0)
            self.assertEqual(held.yaw, YAW0)
            held.nudge(PAN_KEY, PAN_KEY)
            _keys(held, _Key(pygame.K_HOME))
            self.assertEqual(held.yaw, YAW0)
            self.assertEqual(held.pan_x, 0.0)
            self.assertEqual(held.pan_y, 0.0)
        finally:
            pygame.quit()

        origin = np.zeros((1, 3), dtype=np.float64)
        x0, y0, _z0 = _project(origin, 0.0, 0.0, 3.0, 100.0, 100.0, 200.0, 0.0, 0.0)
        x1, y1, _z1 = _project(origin, 0.0, 0.0, 3.0, 100.0, 100.0, 200.0, 0.4, 0.3)
        self.assertLess(float(x1[0]), float(x0[0]))
        self.assertGreater(float(y1[0]), float(y0[0]))

    def test_trainer_arrows_do_not_steer(self):
        import inspect

        from sim.train_monitor import TrainMonitor

        src = inspect.getsource(TrainMonitor.pump)
        self.assertNotIn("steer_x", src)
        self.assertNotIn("steer_z", src)
        self.assertNotIn("K_UP", src)
        self.assertNotIn("K_LEFT", src)

    def test_eyes_follow_the_plaques_and_frames_start_off(self):
        import pygame

        from sim.mb_view3d import BrainCloud, _hemi_recognized, _keys, demo_packet, eye_layout, packet_from_session, render_frame

        cloud = BrainCloud()
        eyes, arrow = eye_layout(cloud)
        by_side = {item["side"]: item for item in eyes}
        self.assertEqual(by_side[1]["label"], "зрит. доля Л")
        self.assertEqual(by_side[0]["label"], "зрит. доля П")
        left_kc = cloud.kc_pos[cloud.kc_side == 1]
        right_kc = cloud.kc_pos[cloud.kc_side == 0]
        self.assertLess(float(by_side[1]["pos"][0]), float(left_kc[:, 0].mean()))
        self.assertGreater(float(by_side[0]["pos"][0]), float(right_kc[:, 0].mean()))
        self.assertGreater(float(by_side[1]["pos"][2]), float(by_side[1]["calyx"][2]))
        self.assertGreater(float(by_side[0]["pos"][2]), float(by_side[0]["calyx"][2]))
        self.assertGreater(float(arrow[1][2]), float(arrow[0][2]))
        self.assertEqual(demo_packet(cloud)["rec_l"], True)
        self.assertEqual(demo_packet(cloud)["rec_r"], False)

        class MB:
            last_fwd = None
            last_fwd_r = None
            r_l = 0.0
            r_r = 0.0
            brain = None
            brain_r = None
            flash = None
            flash_r = None

            def eye_recognized(self):
                return True, False

        class Sim:
            onboard = False
            mb = MB()

        packet = packet_from_session(Sim(), 1.0)
        self.assertTrue(packet["rec_l"])
        self.assertFalse(packet["rec_r"])
        self.assertIn(packet["theme"], ("light", "dark"))

        class Live:
            onboard = True
            eye_l_recognized = False
            eye_r_recognized = True
            mb = None
            remote = {"recognized_L": True, "recognized_R": False, "r_l": 1.0, "r_r": 2.0}

        packet = packet_from_session(Live(), 1.0)
        self.assertFalse(packet["rec_l"])
        self.assertTrue(packet["rec_r"])
        self.assertFalse(_hemi_recognized(object(), "l"))

        pygame.init()
        try:
            plain = render_frame(cloud, demo_packet(cloud), size=(640, 480))
            boxed = render_frame(cloud, demo_packet(cloud), size=(640, 480), frames=True)
            self.assertFalse(pygame.surfarray.array3d(plain).sum() == pygame.surfarray.array3d(boxed).sum())
        finally:
            pygame.quit()

        class _Key:
            def __init__(self, key):
                self.key = key
                self.repeat = False

        pygame.init()
        try:
            frames = [False]
            _keys(type("O", (), {"toggle": lambda self: None, "nudge": lambda *a: None, "reset": lambda self: None})(), _Key(pygame.K_f), frames)
        finally:
            pygame.quit()
        self.assertTrue(frames[0])

    def test_udp_packet_is_local_and_rate_limited(self):
        from sim.mb_view3d import HOST, VIEW_PORT, BrainView, compact_ids, packet_from_session
        import numpy as np

        vec = np.zeros(900, dtype=np.float32)
        vec[:900] = 1
        ids = compact_ids(vec, cap=400)
        self.assertLessEqual(len(ids), 400)
        self.assertGreater(len(ids), 200)
        self.assertEqual(ids, sorted(ids))

        class Fwd:
            def __init__(self, on):
                self.kc = np.zeros(8, dtype=np.float32)
                self.kc[on] = 1

        class MB:
            last_fwd = Fwd([1, 3])
            last_fwd_r = Fwd([2])
            r_l = 4.0
            r_r = -2.0
            brain = None
            brain_r = None
            flash = None
            flash_r = None

        class Session:
            onboard = False
            learner_kind = "mb"
            mb = MB()

        packet = packet_from_session(Session(), 1.5)
        self.assertEqual(packet["kc_l"], [1, 3])
        self.assertEqual(packet["kc_r"], [2])
        self.assertEqual(packet["r_l"], 4.0)
        self.assertEqual(packet["edges"], [])

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((HOST, 0))
        port = sock.getsockname()[1]
        self.assertNotEqual(port, VIEW_PORT)
        link = BrainView(port)

        class Alive:
            def poll(self):
                return None

        link.proc = Alive()
        link.send({"kc_l": [1]}, now=0.0)
        link.send({"kc_l": [9]}, now=0.01)
        link.proc = None
        sock.settimeout(0.5)
        raw, addr = sock.recvfrom(65535)
        self.assertEqual(addr[0], HOST)
        self.assertEqual(json.loads(raw.decode("utf-8"))["kc_l"], [1])
        sock.settimeout(0.05)
        with self.assertRaises(socket.timeout):
            sock.recvfrom(65535)
        link.close()
        sock.close()

    def test_viewer_command_stays_off_the_robot_ports(self):
        from sim.mb_view3d import viewer_command

        cmd = viewer_command(5473)
        text = " ".join(cmd)
        self.assertIn("5473", text)
        self.assertNotIn("8088", text)
        self.assertNotIn("8090", text)
        self.assertNotIn("8091", text)
        self.assertNotIn("5451", text)
        self.assertTrue(text.endswith("5473") or "--port 5473" in text or "--port" in text)


def _channel(value: float) -> float:
    c = value / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _contrast(fg, bg) -> float:
    def lum(rgb):
        return 0.2126 * _channel(rgb[0]) + 0.7152 * _channel(rgb[1]) + 0.0722 * _channel(rgb[2])

    light, dark = lum(fg), lum(bg)
    if dark > light:
        light, dark = dark, light
    return (light + 0.05) / (dark + 0.05)


class TrainerInkTests(unittest.TestCase):
    def test_text_is_near_black_on_the_light_cards(self):
        import pygame

        from sim.learn_flash import _LABEL, _VALUE
        from sim.train_monitor import CARD, INK_DIM, LABEL, VALUE, MonitorView, TrainMonitor

        self.assertEqual(LABEL, (0x22, 0x22, 0x22))
        self.assertEqual(VALUE, (0x11, 0x11, 0x11))
        self.assertLessEqual(max(INK_DIM), 0x44)
        self.assertEqual(_LABEL, LABEL)
        self.assertEqual(_VALUE, VALUE)
        self.assertGreater(_contrast(LABEL, CARD), 7.0)
        self.assertGreater(_contrast(VALUE, CARD), 7.0)
        self.assertGreater(_contrast(INK_DIM, CARD), 4.5)
        pygame.init()
        try:
            mon = TrainMonitor("ink")
            view = MonitorView(
                learner="mb",
                log_lines=["учитель: PAM Л — собака в поле"],
                keys_hint="A авто  B звук  J мозг  Y учитель",
                learning_on=True,
                focused=True,
                udp_status="симулятор",
                last_command="StopMove",
            )
            mon.draw(view)
            image = pygame.surfarray.array3d(mon.screen)
            footer = int(mon.layout["footer_y"])
            hint = image[:, footer + 36: footer + 58]
            ink = (hint.max(axis=2) <= 40) & (hint.min(axis=2) >= 8)
            self.assertGreater(int(ink.sum()), 40)
            button = mon.auto_rect
            patch = image[button.x + 28: button.right - 6, button.y + 6: button.bottom - 6]
            letters = (patch.max(axis=2) <= 40) & (patch.min(axis=2) >= 8)
            self.assertGreater(int(letters.sum()), 20)
        finally:
            pygame.quit()


def _white_plate(frame, block: int = 28, frac: float = 0.92) -> bool:
    """True when a solid near-white rectangle is sitting on the frame.

    Glyphs of near-white text are thinner than ``block`` and do not count.
    """
    white = frame.min(axis=2) >= 245
    height, width = white.shape[1], white.shape[0]
    step = block // 2
    for y in range(0, max(1, height - block), step):
        row = white[:, y : y + block]
        for x in range(0, max(1, width - block), step):
            if float(row[x : x + block].mean()) >= frac:
                return True
    return False


class ThemeSchemeTests(unittest.TestCase):
    def test_scheme_and_blocks_follow_the_theme(self):
        import pygame
        import numpy as np

        from sim.learn_flash import LearnFlash, MbLayout, _LABEL, _LINE, _PAPER, _VALUE
        from sim.train_monitor import CARD, INK_DIM, INSET, LABEL, LINE, VALUE, MonitorView, TrainMonitor
        from sim.ui_theme import apply_theme

        self.assertGreater(_contrast((248, 248, 248), (40, 42, 46)), 7.0)
        self.assertGreater(_contrast((255, 255, 255), (40, 42, 46)), 7.0)
        self.assertGreater(_contrast((198, 200, 204), (40, 42, 46)), 4.5)
        pygame.init()
        try:
            mon = TrainMonitor("scheme")
            mon._on_resize(1920, 1080)
            layout = MbLayout(
                kc_x=np.array([0.18, 0.42, 0.66], np.float32),
                kc_y=np.array([0.35, 0.55, 0.75], np.float32),
                mbon_x=np.array([0.94, 0.94], np.float32),
                mbon_y=np.array([0.28, 0.72], np.float32),
                mbon_label=["MBON Л", "MBON П"],
                mbon_approach=np.array([True, False]),
                kc_root=np.array([101, 202, 303]),
                lobe_name=("γ", "αβ", "α′β′"),
                lobe_x=(0.18, 0.42, 0.66),
                n_pam=40,
                n_ppl1=12,
            )
            camera = np.zeros((90, 160, 3), dtype=np.uint8)
            camera[:] = (18, 20, 24)
            lidar = np.zeros((64, 64, 3), dtype=np.uint8)
            idle = MonitorView(
                learner="mb",
                learning_on=True,
                focused=True,
                camera=camera,
                lidar=lidar,
                lidar_mode="лидар",
                log_lines=["схема обучения открыта"],
                keys_hint="G вспышка  A авто  J мозг",
                udp_status="симулятор",
                last_command="StopMove",
                mb_layout=layout,
                peer_curve=[(0.0, 0.2), (1.0, 0.7), (2.0, 0.4)],
                other_curve=[(0.0, 0.1), (1.0, 0.3), (2.0, 0.2)],
                drift_curve=[(0.0, 0.0), (1.0, 0.4)],
                dan_events=[(0.4, "PAM"), (1.2, "PPL1")],
                kc_bins=np.linspace(0.1, 1.0, 48),
                kc_on=12,
                kc_n=2000,
                teacher_counts="учитель: PAM_L 1  PAM_R 0  PPL1_L 0  PPL1_R 1",
                n_pam=1,
                n_ppl1=1,
                t=4.0,
            )
            out = __import__("pathlib").Path("/opt/cursor/artifacts")
            out.mkdir(parents=True, exist_ok=True)

            def paper_at():
                body = mon._content_of("brain")
                self.assertIsNotNone(body)
                rect = pygame.Rect(body.x + 8, body.y + 78, body.w - 16, max(80, body.h - 96))
                graph = pygame.Rect(rect.x + 8, rect.y + 58, rect.w - 210, rect.h - 146)
                return int(graph.centerx), int(graph.centery)

            mon.flash_open = True
            mon.menu_open = True
            mon._set_theme("light")
            mon.draw(idle)
            frame = pygame.surfarray.array3d(mon.screen)
            px, py = paper_at()
            light_paper = tuple(int(c) for c in frame[px, py])
            self.assertGreater(min(light_paper), 240)
            pygame.image.save(mon.screen, str(out / "trainer_scheme_light.png"))

            mon._set_theme("dark")
            self.assertGreater(_contrast(tuple(LABEL), tuple(CARD)), 7.0)
            self.assertGreater(_contrast(tuple(VALUE), tuple(CARD)), 7.0)
            self.assertGreater(_contrast(tuple(LABEL), tuple(INSET)), 7.0)
            self.assertGreater(_contrast(tuple(INK_DIM), tuple(CARD)), 4.5)
            self.assertEqual(tuple(_PAPER), tuple(INSET))
            self.assertEqual(tuple(_LINE), tuple(LINE))
            self.assertLess(max(tuple(_PAPER)), 80)
            self.assertGreater(min(tuple(_LABEL)), 240)
            self.assertGreater(min(tuple(_VALUE)), 250)
            mon.draw(idle)
            frame = pygame.surfarray.array3d(mon.screen)
            dark_paper = tuple(int(c) for c in frame[px, py])
            self.assertLess(max(dark_paper), 90)
            self.assertFalse(_white_plate(frame))
            pygame.image.save(mon.screen, str(out / "trainer_scheme_dark.png"))

            mon.flash_open = False
            mon.draw(idle)
            self.assertFalse(_white_plate(pygame.surfarray.array3d(mon.screen)))

            pulse = LearnFlash(
                t=4.0,
                kind="PAM",
                kc_on=np.array([0, 1], np.int32),
                pre=np.array([0], np.int32),
                post=np.array([0], np.int32),
                dw=np.array([0.4], np.float32),
                toward=np.array([True]),
                n_syn=1,
                sum_abs=0.4,
                before=0.1,
                after=0.8,
                mbon_before=np.array([0.1, 0.0], np.float32),
                mbon_after=np.array([1.4, -0.2], np.float32),
                strong_pre=0,
                strong_post=0,
            )
            other = LearnFlash(
                t=4.0,
                kind="PPL1",
                kc_on=np.array([2], np.int32),
                pre=np.array([2], np.int32),
                post=np.array([1], np.int32),
                dw=np.array([-0.3], np.float32),
                toward=np.array([False]),
                n_syn=1,
                sum_abs=0.3,
                before=-0.2,
                after=-0.6,
                mbon_before=np.array([0.0, 0.2], np.float32),
                mbon_after=np.array([0.1, -0.8], np.float32),
                strong_pre=2,
                strong_post=1,
            )
            mon.flash_open = True
            shown = MonitorView(**{**idle.__dict__, "learn_flash": pulse, "learn_flash_r": other})
            mon.draw(shown)
            self.assertFalse(_white_plate(pygame.surfarray.array3d(mon.screen)))
        finally:
            apply_theme("light")
            pygame.quit()


def _ink(surface, rect, peach: bool) -> int:
    count = 0
    for y in range(rect.top, rect.bottom):
        for x in range(rect.left, rect.right):
            red, green, blue, _alpha = surface.get_at((x, y))
            is_peach = red > 150 and red > blue + 40 and green < 190
            is_gray = abs(red - green) < 18 and abs(green - blue) < 18 and red > 90
            if peach and is_peach:
                count += 1
            elif not peach and is_gray and not is_peach:
                count += 1
    return count


class WindowPinTests(unittest.TestCase):
    def test_linux_pin_is_a_noop_and_the_call_stays_off_set_mode(self):
        import os
        from pathlib import Path

        import sim.mb_view3d as view
        import sim.window_pin as pin

        if os.name != "nt":
            self.assertFalse(pin.set_always_on_top(True))
            self.assertFalse(pin.set_always_on_top(False))
        src = Path(pin.__file__).read_text(encoding="utf-8")
        for needle in ("SetWindowPos", "HWND_TOPMOST", "HWND_NOTOPMOST", "get_wm_info"):
            self.assertIn(needle, src)
        self.assertNotIn("set_mode", src)
        self.assertNotIn("from_display_module", src)
        viewer = Path(view.__file__).read_text(encoding="utf-8")
        self.assertIn("set_always_on_top", viewer)
        self.assertNotIn("from_display_module", viewer)
        fly = Path(view.__file__).resolve().parent.joinpath("mb_flywire.py").read_text(encoding="utf-8")
        self.assertIn("VIEW_HINT", fly)
        self.assertEqual(view.VIEW_HINT, "ЛКМ вращать · ПКМ сдвиг · колёсико зум · Home сброс")
        self.assertNotIn("поверх окон", view.VIEW_HINT)
        self.assertNotIn("стрелки", view.VIEW_HINT)
        self.assertNotIn("WASD", view.VIEW_HINT)
        self.assertEqual(pin.PIN_TIP, "поверх окон")
        self.assertIn("draw.arc", src)
        self.assertIn("aaline", src)

    def test_click_and_t_toggle_once_and_t_does_not_pan(self):
        import pygame

        from sim.mb_view3d import PAN_KEY, Orbit, _keys
        from sim.window_pin import consume_pin, pin_button_rect

        size = (960, 700)
        rect = pin_button_rect(size)
        on_clip = (rect[0] + 4, rect[1] + 4)
        handled, nxt = consume_pin(False, button=1, pos=on_clip, size=size)
        self.assertTrue(handled)
        self.assertTrue(nxt)
        handled, nxt = consume_pin(True, button=1, pos=(40, 40), size=size)
        self.assertFalse(handled)
        self.assertTrue(nxt)
        handled, nxt = consume_pin(False, button=3, pos=on_clip, size=size)
        self.assertFalse(handled)
        self.assertFalse(nxt)

        pygame.init()
        try:
            handled, nxt = consume_pin(False, key=pygame.K_t, repeat=False)
            self.assertTrue(handled)
            self.assertTrue(nxt)
            handled, nxt = consume_pin(True, key=pygame.K_t, repeat=True)
            self.assertFalse(handled)
            self.assertTrue(nxt)
            orbit = Orbit()
            before = (orbit.pan_x, orbit.pan_y, orbit.yaw, orbit.pitch)

            class _Key:
                def __init__(self, key, repeat=False):
                    self.key = key
                    self.repeat = repeat

            _keys(orbit, _Key(pygame.K_t))
            _keys(orbit, _Key(pygame.K_t, repeat=True))
            self.assertEqual((orbit.pan_x, orbit.pan_y, orbit.yaw, orbit.pitch), before)
            _keys(orbit, _Key(pygame.K_d))
            _keys(orbit, _Key(pygame.K_LEFT))
            self.assertEqual(orbit.pan_x, 0.0)
        finally:
            pygame.quit()

    def test_brain_window_takes_the_trainer_theme_from_the_packet(self):
        import json
        from pathlib import Path

        import pygame

        from sim.mb_view3d import BrainCloud, render_frame, saved_view_theme, set_view_theme
        from sim.ui_settings import read_saved_theme
        from sim.ui_theme import apply_theme

        root = Path("/tmp/recog_theme_file")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "window.json"
        previous = os.environ.get("RECOGNIZER_UI_CONFIG")
        os.environ["RECOGNIZER_UI_CONFIG"] = str(path)
        pygame.init()
        try:
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"theme": "dark"}).encode("utf-8"))
            self.assertEqual(read_saved_theme(), "dark")
            self.assertEqual(saved_view_theme(), "dark")
            path.write_text('{"theme": "lig', encoding="utf-8")
            self.assertIsNone(read_saved_theme())
            self.assertEqual(saved_view_theme(), "light")
            path.unlink()
            self.assertIsNone(read_saved_theme())
            self.assertEqual(saved_view_theme(), "light")
            path.write_text('{"theme": "light", "w": 1}\n', encoding="utf-8")
            self.assertEqual(read_saved_theme(), "light")
            self.assertEqual(saved_view_theme(), "light")

            cloud = BrainCloud()
            packet = {"r_l": 0.0, "r_r": 0.0, "kc_l": [], "kc_r": [], "flash_l": "", "flash_r": "", "rec_l": False, "rec_r": False, "edges": [], "theme": "light"}
            light = render_frame(cloud, packet, size=(160, 120))
            corner = pygame.surfarray.array3d(light)[2, 2].astype(int)
            self.assertGreater(int(corner.mean()), 180)
            packet["theme"] = "dark"
            dark = render_frame(cloud, packet, size=(160, 120))
            corner = pygame.surfarray.array3d(dark)[2, 2].astype(int)
            self.assertLess(int(corner.mean()), 40)
            out = Path("/opt/cursor/artifacts")
            out.mkdir(parents=True, exist_ok=True)
            pygame.image.save(pygame.transform.scale(light, (320, 240)), str(out / "brain3d_theme_light.png"))
            pygame.image.save(pygame.transform.scale(dark, (320, 240)), str(out / "brain3d_theme_dark.png"))
            apply_theme("dark")
            from sim.mb_view3d import packet_from_session

            class _Session:
                onboard = False
                mb = None

            self.assertEqual(packet_from_session(_Session(), 1.0)["theme"], "dark")
        finally:
            set_view_theme("dark")
            apply_theme("light")
            if previous is None:
                os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            else:
                os.environ["RECOGNIZER_UI_CONFIG"] = previous
            pygame.quit()

    def test_clip_is_peach_when_pinned_and_gray_when_not(self):
        from pathlib import Path

        import pygame

        from sim.window_pin import PIN_OFF, PIN_ON, draw_pin_button

        pygame.init()
        try:
            on = pygame.Surface((220, 80))
            off = pygame.Surface((220, 80))
            on.fill((14, 15, 18))
            off.fill((14, 15, 18))
            hot = draw_pin_button(on, True)
            cold = draw_pin_button(off, False)
            self.assertGreater(_ink(on, hot, peach=True), 40)
            self.assertEqual(_ink(on, hot, peach=False), 0)
            self.assertGreater(_ink(off, cold, peach=False), 40)
            self.assertEqual(_ink(off, cold, peach=True), 0)
            bg = (14, 15, 18)
            for y in range(on.get_height()):
                for x in range(on.get_width()):
                    if hot.collidepoint(x, y):
                        continue
                    self.assertEqual(tuple(on.get_at((x, y))[:3]), bg)
            hover = pygame.Surface((220, 80))
            hover.fill(bg)
            draw_pin_button(hover, True, hover=True)
            outside = 0
            for y in range(hover.get_height()):
                for x in range(hot.left):
                    if tuple(hover.get_at((x, y))[:3]) != bg:
                        outside += 1
            self.assertGreater(outside, 40)
            dest = Path("/opt/cursor/artifacts")
            dest.mkdir(parents=True, exist_ok=True)
            pygame.image.save(on, str(dest / "pin_on.png"))
            pygame.image.save(off, str(dest / "pin_off.png"))
            pygame.image.save(hover, str(dest / "pin_hover.png"))
        finally:
            pygame.quit()

    def test_pinned_roundtrip_and_the_help_names_t(self):
        import json
        from pathlib import Path

        from sim.recognize_train import SIM_KEYS
        from sim.ui_help import page
        from sim.ui_settings import brain_window_path, load_brain_window, save_brain_window

        root = Path("/tmp/recog_pin_window")
        root.mkdir(parents=True, exist_ok=True)
        previous = os.environ.get("RECOGNIZER_UI_CONFIG")
        os.environ["RECOGNIZER_UI_CONFIG"] = str(root / "window.json")
        try:
            path = brain_window_path()
            path.write_text(json.dumps({"x": 3, "y": 4, "w": 800, "h": 600}), encoding="utf-8")
            self.assertFalse(load_brain_window()["pinned"])
            save_brain_window({"x": 3, "y": 4, "w": 800, "h": 600, "pinned": True})
            self.assertTrue(load_brain_window()["pinned"])
            save_brain_window({"x": 3, "y": 4, "w": 800, "h": 600, "pinned": "false"})
            self.assertFalse(load_brain_window()["pinned"])
            text = path.read_text(encoding="utf-8")
            self.assertIn("pinned", text)
            self.assertNotIn("embed_3d", text)
        finally:
            if previous is None:
                os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            else:
                os.environ["RECOGNIZER_UI_CONFIG"] = previous

        _title, blocks = page("bar")
        prose = "\n".join(text for _kind, text in blocks)
        self.assertIn("скрепк", prose)
        self.assertIn("поверх окон", prose)
        self.assertIn("T в том окне", prose)
        self.assertIn("brain_window.json", prose)
        self.assertIn("персиков", prose)
        self.assertIn("серая", prose)
        self.assertIn("лакомство PAM", prose)
        self.assertIn("T/X учить", SIM_KEYS)
        self.assertNotIn("поверх", SIM_KEYS)


class CalloutTests(unittest.TestCase):
    def test_two_levels_leaders_and_the_facing_badge(self):
        import math

        import numpy as np

        from sim.mb_flywire import anatomy_anchors, load_scene
        from sim.mb_view3d import (
            COARSE_LABELS,
            DETAIL_LABELS,
            DIST0,
            PITCH0,
            VIEW_HINT,
            YAW0,
            BrainCloud,
            _outside_box,
            _project,
            consume_details,
            details_hit_rect,
            facing_label,
            hit_details,
            layout_callouts,
            schematic_anchors,
        )
        from sim.window_pin import pin_button_rect

        self.assertEqual(VIEW_HINT, "ЛКМ вращать · ПКМ сдвиг · колёсико зум · Home сброс")
        self.assertEqual(facing_label(YAW0, PITCH0, DIST0), "вид сзади")
        self.assertEqual(facing_label(YAW0 + math.pi, PITCH0, DIST0), "вид спереди")

        size = (960, 700)
        pin = pin_button_rect(size)
        pin_box = (pin[0], pin[1], pin[0] + pin[2], pin[1] + pin[3])
        box = details_hit_rect(size)
        self.assertFalse(
            box[0] < pin_box[2] and pin_box[0] < box[0] + box[2] and box[1] < pin_box[3] and pin_box[1] < box[1] + box[3]
        )
        self.assertTrue(hit_details(size, (box[0] + 4, box[1] + 4)))
        handled, nxt = consume_details(False, button=1, pos=(box[0] + 4, box[1] + 4), size=size)
        self.assertTrue(handled)
        self.assertTrue(nxt)
        handled, nxt = consume_details(False, button=1, pos=(400, 400), size=size)
        self.assertFalse(handled)
        self.assertFalse(nxt)

        cloud = BrainCloud()
        schematic = {(item["label"], item["side"]): item for item in schematic_anchors(cloud)}
        self.assertGreater(float(schematic[("α", 1)]["pos"][1]), float(schematic[("β", 1)]["pos"][1]))
        self.assertLess(abs(float(schematic[("β", 1)]["pos"][0])), abs(float(schematic[("α", 1)]["pos"][0])))
        self.assertLess(abs(float(schematic[("γ", 1)]["pos"][0])), abs(float(schematic[("ножка", 1)]["pos"][0])))
        self.assertLess(float(schematic[("чашечка", 1)]["pos"][2]), float(schematic[("ножка", 1)]["pos"][2]))
        self.assertTrue(schematic[("LAL", -1)]["approx"])
        self.assertTrue(schematic[("DN → поворот", -1)]["approx"])
        self.assertLess(float(schematic[("DN → поворот", -1)]["pos"][1]), float(schematic[("LAL", -1)]["pos"][1]))

        scene = load_scene()
        self.assertIsNotNone(scene)
        anchors = anatomy_anchors(scene)
        placed = {(item["label"], item["side"]): item for item in anchors}
        alpha = placed[("α", 1)]["pos"]
        beta = placed[("β", 1)]["pos"]
        self.assertGreater(float(alpha[1]), float(beta[1]))
        self.assertLess(abs(float(beta[0])), abs(float(alpha[0])))
        self.assertLess(abs(float(placed[("γ", 1)]["pos"][0])), abs(float(placed[("ножка", 1)]["pos"][0])))
        self.assertLess(float(placed[("чашечка", 1)]["pos"][2]), float(placed[("ножка", 1)]["pos"][2]))
        self.assertTrue(placed[("LAL", -1)]["approx"])
        self.assertTrue(placed[("DN → поворот", -1)]["approx"])
        self.assertLess(float(placed[("зрит. доля Л", 1)]["pos"][0]), 0.0)
        self.assertGreater(float(placed[("зрит. доля П", 0)]["pos"][0]), 0.0)
        _sx, sy, _dep = _project(
            np.vstack((alpha, beta)),
            YAW0,
            PITCH0,
            float(scene.fit_dist),
            480.0,
            360.0,
            500.0,
        )
        self.assertLess(float(sy[0]), float(sy[1]))

        shell = scene.vertices[::3]
        coarse, box, shown = layout_callouts(anchors, shell, YAW0, PITCH0, float(scene.fit_dist), size)
        self.assertFalse(shown)
        self.assertEqual({item["label"] for item in coarse}, set(COARSE_LABELS))
        for item in coarse:
            self.assertTrue(_outside_box(item["rect"], box))
            self.assertGreater(item["rect"][2] - item["rect"][0], item["rect"][3] - item["rect"][1])
        detail, box, shown = layout_callouts(
            anchors, shell, YAW0, PITCH0, float(scene.fit_dist), size, details=True
        )
        self.assertTrue(shown)
        self.assertTrue(set(DETAIL_LABELS) <= {item["label"] for item in detail})
        self.assertTrue(set(COARSE_LABELS) <= {item["label"] for item in detail})
        for item in detail:
            self.assertTrue(_outside_box(item["rect"], box))
        hover, _box, shown = layout_callouts(
            anchors, shell, YAW0, PITCH0, float(scene.fit_dist), size, pointer=(480, 360)
        )
        self.assertTrue(shown)
        self.assertTrue(set(DETAIL_LABELS) <= {item["label"] for item in hover})


if __name__ == "__main__":
    unittest.main()
