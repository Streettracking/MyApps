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
    TeachGate,
    TeacherRuntime,
    YoloClient,
    dan_for_eyes,
    hemisphere_hits,
    plaque_recognized,
    send_teach,
    should_query,
    teach_phrase,
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
        amber = (
            (np.abs(frame[:, :, 0].astype(int) - 214) < 28)
            & (np.abs(frame[:, :, 1].astype(int) - 168) < 28)
            & (np.abs(frame[:, :, 2].astype(int) - 64) < 28)
        )
        self.assertGreater(int(amber.sum()), 10)
        from sim.train_monitor import WIN_W, monitor_layout

        self.assertGreaterEqual(monitor_layout(WIN_W, 1080)["cam"][2], WIN_W // 2)
        pygame.quit()


def _ready(boxes=()):
    runtime = TeacherRuntime(conf=0.5, rate=2.0)
    runtime.on = True
    runtime._learning = True
    runtime.answered = True
    runtime.service_down = False
    runtime._teach_boxes = list(boxes)
    runtime.boxes = list(boxes)
    return runtime


class PunishTests(unittest.TestCase):
    def test_empty_answer_punishes_the_plaque_bit(self):
        runtime = _ready([])
        self.assertEqual(runtime.collect(0.0, 0.4, True, False, True, False), ("ppl1", None))
        self.assertEqual(runtime.ppl1_l, 1)
        self.assertEqual(runtime.skip_n, 0)

    def test_no_answer_yet_does_not_invent_ppl1(self):
        runtime = TeacherRuntime(conf=0.5, rate=2.0)
        runtime.on = True
        runtime._learning = True
        self.assertFalse(runtime.answered)
        self.assertEqual(runtime.collect(0.0, 0.4, True, False, True, False), (None, None))
        self.assertEqual(runtime.skip_reason, "нет ответа YOLO")
        self.assertIn("нет ответа YOLO", runtime.skips_line())
        self.assertEqual(runtime.ppl1_l, 0)
        runtime.collect(0.1, 0.4, True, False, True, False)
        self.assertEqual(runtime.skip_n, 1)

    def test_weak_box_and_other_hemisphere_do_not_block_ppl1(self):
        weak = DetBox(0.80, 0.2, 0.96, 0.8, 0.40)
        runtime = _ready([weak])
        self.assertEqual(runtime.collect(0.0, 0.4, True, False, True, False), ("ppl1", None))
        spill = DetBox(0.05, 0.1, 0.45, 0.9, 0.92)
        self.assertEqual(hemisphere_hits([spill], 0.4, 0.5, center_only=False), (True, True))
        self.assertEqual(hemisphere_hits([spill], 0.4, 0.5, center_only=True), (False, True))
        runtime = _ready([spill])
        self.assertEqual(runtime.collect(0.0, 0.4, True, False, True, False), ("ppl1", "pam"))
        middle = DetBox(0.42, 0.2, 0.58, 0.8, 0.88)
        self.assertEqual(hemisphere_hits([middle], 0.4, 0.5, center_only=True), (True, True))
        runtime = _ready([middle])
        self.assertEqual(runtime.collect(0.0, 0.4, True, True, True, False), ("pam", "pam"))
        self.assertEqual(runtime.skip_reason, "бокс в поле")

    def test_pam_and_ppl1_do_not_share_a_slot(self):
        gate = TeachGate(2.0)
        self.assertEqual(gate.filter(0.0, "pam", "pam"), ("pam", "pam"))
        self.assertEqual(gate.filter(0.1, "ppl1", "ppl1"), ("ppl1", "ppl1"))
        self.assertEqual(gate.filter(0.2, "ppl1", None), (None, None))
        self.assertEqual(gate.filter(0.49, "pam", None), (None, None))
        dog = DetBox(0.80, 0.2, 0.96, 0.8, 0.93)
        runtime = _ready([dog])
        self.assertEqual(runtime.collect(0.0, 0.4, True, False, True, False), ("pam", None))
        runtime._teach_boxes = []
        self.assertEqual(runtime.collect(0.1, 0.4, True, False, True, False), ("ppl1", None))
        self.assertEqual(runtime.collect(0.2, 0.4, True, False, True, False), (None, None))
        self.assertEqual(runtime.skip_reason, "лимит")
        self.assertIn("(лимит)", runtime.skips_line())

    def test_phrases_and_onboard_plaque_bits(self):
        self.assertEqual(teach_phrase("L", "ppl1"), "учитель: PPL1 Л — ложное узнавание")
        self.assertEqual(teach_phrase("R", "ppl1"), "учитель: PPL1 П — ложное узнавание")
        self.assertEqual(teach_phrase("L", "pam"), "учитель: PAM Л — собака в поле")
        self.assertEqual(teach_phrase("R", "pam"), "учитель: PAM П — собака в поле")

        class Brain:
            overlap = 0.25

            def eye_recognized(self):
                return False, True

        self.assertEqual(plaque_recognized(Brain(), True, False, 0.4), (False, True, 0.25))
        self.assertEqual(plaque_recognized(None, True, False, 0.4), (True, False, 0.4))

        from sim.recognize_train_live import _drive_teacher

        class Session:
            learner_kind = "mb"
            mb = None
            learn = True
            eye_l_recognized = True
            eye_r_recognized = False
            overlap = 0.4
            log = []

            def now(self):
                return 3.0

            def _log(self, text):
                self.log.append(text)

        session = Session()
        session.teacher = _ready([])
        _drive_teacher(session, None, 1, {}, False, None)
        self.assertEqual(session.log, ["учитель: PPL1 Л — ложное узнавание"])
        self.assertEqual(session.teacher_flash_l_show, "ppl1")
        self.assertEqual(session.teacher_flash_r_show, "")
        self.assertTrue(session.teacher_skips.endswith(": 0"))
        session.teacher.answered = False
        session.eye_l_recognized = True
        _drive_teacher(session, None, 2, {}, False, None)
        self.assertEqual(session.teacher.skip_reason, "нет ответа YOLO")
        self.assertIn("нет ответа YOLO", session.teacher_skips)


class LayoutTests(unittest.TestCase):
    def test_camera_stays_large_on_1080p_and_smaller(self):
        from sim.train_monitor import monitor_layout

        for w, h in ((1920, 1080), (1600, 900), (1280, 720)):
            box = monitor_layout(w, h)
            cam = box["cam"]
            self.assertGreaterEqual(cam[2], w // 2)
            self.assertLess(cam[3], h)
            for name, rect in box.items():
                if name in ("controls_y", "footer_y"):
                    self.assertGreaterEqual(rect, 0)
                    self.assertLess(rect, h)
                    continue
                x, y, rw, rh = rect
                self.assertGreaterEqual(x, 0, name)
                self.assertGreaterEqual(y, 0, name)
                self.assertLessEqual(x + rw, w + 1, name)
                self.assertLessEqual(y + rh, h + 1, name)
            self.assertLess(box["cam"][1] + box["cam"][3], box["lid"][1] + 2)
            self.assertLess(box["estop"][0], w)
            self.assertGreater(box["estop"][0], box["treat"][0])


class SimGuiTests(unittest.TestCase):
    def test_sim_and_robot_counts_do_not_read_onboard(self):
        from sim.recognize_train_live import _teacher_counts
        from sim.yolo_teacher import TeacherRuntime, counts_line

        teacher = TeacherRuntime()
        teacher.pam_l = 4
        teacher.ppl1_r = 2

        class Sim:
            pass

        class Robot:
            remote = {
                "teacher_pam_l": 99,
                "teacher_pam_r": 1,
                "teacher_ppl1_l": 3,
                "teacher_ppl1_r": 0,
            }

        class Board:
            onboard = True
            remote = {
                "teacher_pam_l": 7,
                "teacher_pam_r": 1,
                "teacher_ppl1_l": 3,
                "teacher_ppl1_r": 0,
            }

        local = counts_line(4, 0, 0, 2)
        self.assertEqual(_teacher_counts(Sim(), teacher), local)
        self.assertEqual(_teacher_counts(Robot(), teacher), local)
        self.assertEqual(_teacher_counts(Board(), teacher), counts_line(7, 1, 3, 0))

    def test_sim_gui_teacher_runs_several_frames(self):
        """The --sim window loop, headless. Learning and the teacher are on."""
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import tempfile
        from pathlib import Path

        from sim.recognize_train import RecognizeTrainSim, run_gui
        from sim.yolo_teacher import DetBox, TeacherRuntime, counts_line

        real_init = TeacherRuntime.__init__

        def _init(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            self.on = True
            self.client.health = lambda: True
            self.client.detect = lambda jpeg: [DetBox(0.05, 0.15, 0.28, 0.85, 0.92)]

        TeacherRuntime.__init__ = _init
        folder = tempfile.mkdtemp()
        try:
            session = RecognizeTrainSim(
                n_agents=2,
                seed=1,
                state_path=Path(folder) / "mb_train_state.npz",
                learner="mb",
                dan="teacher",
            )
            self.assertTrue(session.learn)
            self.assertFalse(hasattr(session, "onboard"))
            summary = run_gui(session, seconds=0.5)
            self.assertGreaterEqual(float(summary["t"]), 0.5)
            self.assertTrue(session.teacher.on)
            self.assertEqual(
                session.teacher_counts,
                counts_line(
                    session.teacher.pam_l,
                    session.teacher.pam_r,
                    session.teacher.ppl1_l,
                    session.teacher.ppl1_r,
                ),
            )
            self.assertFalse(session.state_path.is_file())
        finally:
            TeacherRuntime.__init__ = real_init
            import pygame

            if pygame.get_init():
                pygame.quit()


class FlashTests(unittest.TestCase):
    def test_plaques_flash_and_the_skip_line_is_drawn(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame
        import numpy as np

        pygame.init()
        from sim.train_monitor import MonitorView, TrainMonitor

        mon = TrainMonitor("punish")
        camera = np.zeros((96, 160, 3), dtype=np.uint8)
        camera[:] = (24, 26, 32)
        original = camera.copy()
        view = MonitorView(
            camera=camera,
            learner="mb",
            learning_on=True,
            eye_l_recognized=True,
            eye_r_recognized=False,
            teacher_flash_l="ppl1",
            teacher_flash_r="pam",
            teacher_skips="ложных узнаваний без наказания: 3 (нет ответа YOLO)",
            teacher_counts="учитель: PAM_L 1 / PAM_R 0 / PPL1_L 1 / PPL1_R 0",
            yolo_state="учит",
            log_lines=["  3.0s  учитель: PPL1 Л — ложное узнавание"],
            keys_hint="Y учитель",
        )
        mon.draw(view)
        np.testing.assert_array_equal(camera, original)
        shot = __import__("pathlib").Path("/opt/cursor/artifacts/teacher_punish_flash.png")
        shot.parent.mkdir(parents=True, exist_ok=True)
        mon.save_screenshot(str(shot))
        frame = pygame.surfarray.array3d(mon.screen)
        band = frame[:, 60:160, :]
        red = (
            (np.abs(band[:, :, 0].astype(int) - 204) < 28)
            & (np.abs(band[:, :, 1].astype(int) - 62) < 28)
            & (np.abs(band[:, :, 2].astype(int) - 56) < 28)
        )
        green = (
            (np.abs(band[:, :, 0].astype(int) - 64) < 28)
            & (np.abs(band[:, :, 1].astype(int) - 184) < 28)
            & (np.abs(band[:, :, 2].astype(int) - 96) < 28)
        )
        self.assertGreater(int(red.sum()), 20)
        self.assertGreater(int(green.sum()), 20)
        red_x = np.where(red)[0]
        green_x = np.where(green)[0]
        self.assertGreater(float(red_x.mean()), float(green_x.mean()))
        pygame.quit()


if __name__ == "__main__":
    unittest.main()
