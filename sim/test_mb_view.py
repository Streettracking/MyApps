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
        from sim.train_monitor import TrainMonitor

        self.assertIn("J мозг", SIM_KEYS)
        self.assertIn("J мозг", LIVE_KEYS)
        self.assertIn(" B", SIM_KEYS)
        pygame.init()
        mon = TrainMonitor("window")
        ww, wh = mon.window.get_size()
        logical = mon._logical_pos((ww // 2, wh // 2))
        self.assertAlmostEqual(logical[0], 960, delta=2)
        self.assertAlmostEqual(logical[1], 540, delta=2)
        center = mon.brain_rect.center
        wx = int(round(center[0] * ww / 1920.0))
        wy = int(round(center[1] * wh / 1080.0))
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


class BrainTests(unittest.TestCase):
    def test_soma_cloud_and_one_frame(self):
        import pygame

        from sim.mb_view3d import VIEW_PORT, BrainCloud, connectome_npz, demo_packet, render_frame

        self.assertNotIn(VIEW_PORT, (8088, 8090, 8091, 5451))
        path = connectome_npz()
        before = (path.stat().st_mtime_ns, path.stat().st_size)
        cloud = BrainCloud()
        self.assertEqual((path.stat().st_mtime_ns, path.stat().st_size), before)
        self.assertEqual(cloud.source, "flywire-soma")
        self.assertEqual(cloud.n_schematic, 0)
        self.assertGreater(cloud.n_kc, 1000)
        left = cloud.kc_pos[cloud.kc_side == 1]
        right = cloud.kc_pos[cloud.kc_side == 0]
        self.assertGreater(len(left), 100)
        self.assertGreater(len(right), 100)
        self.assertLess(float(left[:, 0].mean()), float(right[:, 0].mean()))
        pygame.init()
        frame = render_frame(cloud, demo_packet(cloud), size=(640, 480))
        image = pygame.surfarray.array3d(frame)
        peach = (
            (abs(image[:, :, 0].astype(int) - 201) < 8)
            & (abs(image[:, :, 1].astype(int) - 120) < 8)
            & (abs(image[:, :, 2].astype(int) - 91) < 8)
        )
        green = (
            (abs(image[:, :, 0].astype(int) - 141) < 8)
            & (abs(image[:, :, 1].astype(int) - 181) < 8)
            & (abs(image[:, :, 2].astype(int) - 150) < 8)
        )
        self.assertGreater(int(peach.sum()), 10)
        self.assertGreater(int(green.sum()), 10)
        self.assertLess(int(peach.nonzero()[0].mean()), int(green.nonzero()[0].mean()))
        pygame.quit()

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


if __name__ == "__main__":
    unittest.main()
