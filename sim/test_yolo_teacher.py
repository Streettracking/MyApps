"""Hemisphere rewards and the onboard teach channel. No torch and no robot.

Run: python3 -m unittest sim.test_yolo_teacher
"""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sim.yolo_teacher import (
    BOXES_KEY,
    FULLSCREEN_KEY,
    STEER_KEY,
    TEACHER_KEY,
    DetBox,
    TeacherRuntime,
    YoloClient,
    dan_for_eyes,
    send_teach,
    should_query,
    yolo_status,
    zone_of,
)


class _Quiet(BaseHTTPRequestHandler):
    boxes = [{"x0": 0.02, "y0": 0.1, "x1": 0.28, "y1": 0.9, "conf": 0.91}]

    def log_message(self, fmt, *args):
        return

    def _send(self, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({"ok": True})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length:
            self.rfile.read(length)
        self._send({"ok": True, "boxes": self.boxes})


class ZoneTests(unittest.TestCase):
    def test_labels_follow_the_hemifields(self):
        overlap = 0.4
        right_edge = DetBox(0.02, 0.2, 0.2, 0.8, 0.9)
        left_edge = DetBox(0.8, 0.2, 0.98, 0.8, 0.9)
        middle = DetBox(0.4, 0.2, 0.6, 0.8, 0.9)
        self.assertEqual(zone_of(right_edge, overlap), "П")
        self.assertEqual(zone_of(left_edge, overlap), "Л")
        self.assertEqual(zone_of(middle, overlap), "Л+П")

    def test_false_recognize_is_ppl1_and_a_quiet_miss_is_nothing(self):
        self.assertEqual(dan_for_eyes(False, True, False, False), (None, "pam"))
        self.assertEqual(dan_for_eyes(False, False, True, False), ("ppl1", None))
        self.assertEqual(dan_for_eyes(False, False, False, False), (None, None))
        self.assertEqual(dan_for_eyes(True, True, True, False), ("pam", "pam"))

    def test_learning_gate(self):
        self.assertEqual(yolo_status(False, True, False), "выкл (нет обучения)")
        self.assertEqual(yolo_status(True, False, False), "смотрит")
        self.assertEqual(yolo_status(True, True, False), "учит")
        self.assertEqual(yolo_status(True, True, True), "учитель не запущен")
        self.assertFalse(should_query(False, True, False, False, 10.0))
        self.assertFalse(should_query(True, False, True, False, 10.0))
        self.assertTrue(should_query(True, True, True, False, 10.0))
        self.assertTrue(should_query(True, False, False, False, 10.0))
        self.assertFalse(should_query(True, False, False, True, 0.2))


class ChannelTests(unittest.TestCase):
    def test_dead_service_returns_none(self):
        client = YoloClient("http://127.0.0.1:9", timeout=0.2)
        self.assertIsNone(client.detect(b"not-a-jpeg"))
        self.assertFalse(client.health())

    def test_onboard_post_is_teach_sides_and_not_move(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Quiet)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            runtime = TeacherRuntime("http://127.0.0.1:%d" % port, conf=0.5, rate=2.0)
            runtime.on = True
            self.assertTrue(runtime.wants_frame(True))
            self.assertFalse(runtime.wants_frame(False))
            runtime.offer_frame(b"\xff\xd8\xff")
            deadline = time.monotonic() + 2.0
            kinds = (None, None)
            while time.monotonic() < deadline:
                kinds = runtime.collect(time.monotonic(), 0.4, False, False, True, False)
                if runtime.boxes:
                    break
                time.sleep(0.02)
            self.assertEqual(zone_of(runtime.boxes[0], 0.4), "П")
            self.assertEqual(kinds, (None, "pam"))
            posted = []

            class Link:
                def post(self, op, **extra):
                    posted.append((op, extra))
                    return {}

            send_teach(Link(), kinds[0], kinds[1])
            self.assertEqual(posted, [("teach_sides", {"left": "", "right": "pam"})])
            self.assertNotIn("move", [item[0] for item in posted])
            runtime.note_learning(False)
            self.assertEqual(runtime.visible_boxes(False), [])
            self.assertEqual(runtime.status_text(False), "выкл (нет обучения)")
            runtime.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_brain_teach_does_not_drive_and_skips_npz_keys(self):
        import tempfile
        from pathlib import Path

        import numpy as np

        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import MbTrainer, default_npz
        from sim.npz_compat import open_npz

        brain = MbTrainer(default_npz(), seed=1, overlap=0.4)
        loop = BrainLoop(SportDrive(None), brain, "/tmp/mb_teacher_side.npz")
        loop.learn = True
        loop.mb.learn = True
        camera = np.zeros((64, 96, 3), dtype=np.uint8)
        camera[:, :48] = (220, 40, 40)
        camera[:, 48:] = (20, 20, 80)
        lidar = np.zeros_like(camera)
        loop.handle({"op": "teach_sides", "left": "pam", "right": ""}, 1.0)
        loop.ingest(camera, lidar, 1.0)
        self.assertEqual(loop.drive.moves, [])
        self.assertEqual(loop.pilot.phase, "stop")
        self.assertEqual(loop.pilot.mode, "manual")
        self.assertEqual(loop.mb.teacher_pam_l, 1)
        self.assertEqual(loop.mb.teacher_pam_r, 0)
        self.assertEqual(loop.mb.reinforce, "teacher")
        loop.learn = False
        loop.mb.learn = False
        before = loop.mb.n_pam
        loop.handle({"op": "teach_sides", "left": "", "right": "ppl1"}, 2.0)
        loop.ingest(camera, lidar, 2.0)
        self.assertEqual(loop.mb.n_pam, before)
        self.assertEqual(loop.mb.teacher_ppl1_r, 0)
        self.assertEqual(loop.drive.moves, [])
        payload = loop.status()
        self.assertIn("last_seen_side", payload)
        self.assertIn("teacher_pam_l", payload)
        self.assertIn("phase_ru", payload)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.npz"
            loop.mb.save(path)
            with open_npz(path) as saved:
                keys = set(saved.files)
        self.assertNotIn("teacher_pam_l", keys)
        self.assertNotIn("overlap", keys)


class KeyTests(unittest.TestCase):
    def test_f11_h_and_y_are_free_of_each_other_and_of_a(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        pygame.init()
        self.assertNotEqual(pygame.K_F11, pygame.K_a)
        self.assertNotEqual(pygame.K_h, pygame.K_a)
        self.assertNotEqual(pygame.K_y, pygame.K_a)
        self.assertNotEqual(pygame.K_y, pygame.K_k)
        self.assertNotEqual(pygame.K_h, pygame.K_y)
        self.assertNotEqual(pygame.K_F11, pygame.K_h)
        self.assertEqual(TEACHER_KEY, "Y")
        self.assertEqual(BOXES_KEY, "H")
        self.assertEqual(FULLSCREEN_KEY, "F11")
        self.assertEqual(STEER_KEY, "K")
        from sim.train_monitor import TrainMonitor
        from sim.yolo_teacher import DetBox

        mon = TrainMonitor("keys")
        for key, attr in (
            (pygame.K_F11, "fullscreen_toggle"),
            (pygame.K_h, "boxes_toggle"),
            (pygame.K_y, "teacher_toggle"),
            (pygame.K_k, "steer_toggle"),
            (pygame.K_a, "autonomy_toggle"),
        ):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0))
            inp = mon.pump()
            self.assertTrue(getattr(inp, attr), key)
            self.assertFalse(inp.teacher_toggle and attr != "teacher_toggle")
        import numpy as np

        from sim.train_monitor import MonitorView

        camera = np.zeros((96, 160, 3), dtype=np.uint8)
        camera[:] = (24, 26, 32)
        original = camera.copy()
        view = MonitorView(
            camera=camera,
            learner="mb",
            learning_on=True,
            yolo_state="смотрит",
            teacher_counts="учитель: PAM_L 0 / PAM_R 1 / PPL1_L 0 / PPL1_R 0",
            teacher_boxes=[DetBox(0.04, 0.15, 0.32, 0.85, 0.91, "П")],
            keys_hint="A авто  K руль  Y учитель  H рамки  F11 экран",
            phase_ru="поиск ←",
            last_seen_side="L",
            overlap=0.4,
        )
        mon.draw(view)
        np.testing.assert_array_equal(camera, original)
        shot = __import__("pathlib").Path("/opt/cursor/artifacts/yolo_overlay.png")
        shot.parent.mkdir(parents=True, exist_ok=True)
        mon.save_screenshot(str(shot))
        frame = pygame.surfarray.array3d(mon.screen)
        orange = (
            (np.abs(frame[:, :, 0].astype(int) - 255) < 40)
            & (np.abs(frame[:, :, 1].astype(int) - 148) < 40)
            & (np.abs(frame[:, :, 2].astype(int) - 40) < 50)
        )
        self.assertGreater(int(orange.sum()), 10)
        pygame.quit()


if __name__ == "__main__":
    unittest.main()
