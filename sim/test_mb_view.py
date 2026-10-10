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
        self.assertEqual(anchors[1]["label"], "глаз Л")
        self.assertEqual(anchors[0]["label"], "глаз П")
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

    def test_idle_orbit_stops_when_toggled_off(self):
        from sim.mb_view3d import Orbit

        orbit = Orbit()
        start = orbit.yaw
        orbit.tick(0.6, False)
        self.assertGreater(orbit.yaw, start)
        orbit.toggle()
        held = orbit.yaw
        orbit.tick(1.0, False)
        self.assertEqual(orbit.yaw, held)
        orbit.drag(10, 0)
        self.assertGreater(orbit.yaw, held)

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
            _keys(held, _Key(pygame.K_d))
            self.assertAlmostEqual(held.pan_x, PAN_KEY)
            _keys(held, _Key(pygame.K_PAGEUP))
            self.assertAlmostEqual(held.pan_y, PAN_KEY)
            _keys(held, _Key(pygame.K_a, repeat=False))
            self.assertFalse(held.auto)
            self.assertAlmostEqual(held.pan_x, PAN_KEY)
            _keys(held, _Key(pygame.K_a, repeat=True))
            self.assertAlmostEqual(held.pan_x, 0.0)
            _keys(held, _Key(pygame.K_HOME))
            self.assertEqual(held.yaw, YAW0)
            self.assertEqual(held.pan_x, 0.0)
        finally:
            pygame.quit()

        origin = np.zeros((1, 3), dtype=np.float64)
        x0, y0, _z0 = _project(origin, 0.0, 0.0, 3.0, 100.0, 100.0, 200.0, 0.0, 0.0)
        x1, y1, _z1 = _project(origin, 0.0, 0.0, 3.0, 100.0, 100.0, 200.0, 0.4, 0.3)
        self.assertLess(float(x1[0]), float(x0[0]))
        self.assertGreater(float(y1[0]), float(y0[0]))

    def test_eyes_follow_the_plaques_and_frames_start_off(self):
        import pygame

        from sim.mb_view3d import BrainCloud, _hemi_recognized, _keys, demo_packet, eye_layout, packet_from_session, render_frame

        cloud = BrainCloud()
        eyes, arrow = eye_layout(cloud)
        by_side = {item["side"]: item for item in eyes}
        self.assertEqual(by_side[1]["label"], "глаз Л")
        self.assertEqual(by_side[0]["label"], "глаз П")
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


if __name__ == "__main__":
    unittest.main()
