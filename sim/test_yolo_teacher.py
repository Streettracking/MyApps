"""Hemisphere rewards and the onboard teach channel. No torch and no robot.

Run: python3 -m unittest sim.test_yolo_teacher
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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
        image_left = DetBox(0.02, 0.2, 0.2, 0.8, 0.9)
        image_right = DetBox(0.8, 0.2, 0.98, 0.8, 0.9)
        middle = DetBox(0.4, 0.2, 0.6, 0.8, 0.9)
        self.assertEqual(zone_of(image_left, overlap), "Л")
        self.assertEqual(zone_of(image_right, overlap), "П")
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
            self.assertEqual(zone_of(runtime.boxes[0], 0.4), "Л")
            self.assertEqual(kinds, ("pam", None))
            posted = []

            class Link:
                def post(self, op, **extra):
                    posted.append((op, extra))
                    return {}

            send_teach(Link(), kinds[0], kinds[1])
            self.assertEqual(posted, [("teach_sides", {"left": "pam", "right": ""})])
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
        self.assertIsInstance(payload["kc_l"], list)
        self.assertIsInstance(payload["kc_r"], list)
        self.assertLessEqual(len(payload["kc_l"]), 400)
        self.assertLessEqual(len(payload["kc_r"]), 400)
        self.assertTrue(all(isinstance(i, int) for i in payload["kc_l"]))
        self.assertTrue(all(isinstance(i, int) for i in payload["kc_r"]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.npz"
            loop.mb.save(path)
            with open_npz(path) as saved:
                keys = set(saved.files)
        self.assertNotIn("teacher_pam_l", keys)
        self.assertNotIn("overlap", keys)
        self.assertNotIn("kc_l", keys)
        self.assertNotIn("kc_r", keys)


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
            teacher_boxes=[DetBox(0.04, 0.15, 0.32, 0.85, 0.91, "Л")],
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
        peach = (
            (np.abs(frame[:, :, 0].astype(int) - 201) < 36)
            & (np.abs(frame[:, :, 1].astype(int) - 120) < 36)
            & (np.abs(frame[:, :, 2].astype(int) - 91) < 36)
        )
        self.assertGreater(int(peach.sum()), 10)
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
        self.assertEqual(hemisphere_hits([spill], 0.4, 0.5, center_only=True), (True, False))
        runtime = _ready([spill])
        self.assertEqual(runtime.collect(0.0, 0.4, True, False, True, False), ("pam", None))
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
        dog = DetBox(0.04, 0.2, 0.22, 0.8, 0.93)
        runtime = _ready([dog])
        self.assertEqual(runtime.collect(0.0, 0.4, True, False, True, False), ("pam", None))
        runtime._teach_boxes = []
        self.assertEqual(runtime.collect(0.1, 0.4, True, False, True, False), ("ppl1", None))
        self.assertEqual(runtime.collect(0.2, 0.4, True, False, True, False), (None, None))
        self.assertEqual(runtime.skip_reason, "лимит")
        self.assertIn("лимит", runtime.skips_line())

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
        from sim.tabnum import skips_are_quiet

        self.assertTrue(skips_are_quiet(session.teacher_skips))
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
            self.assertEqual(box["estop"][0] + box["estop"][2], w - 16)
            self.assertGreater(box["estop"][2], box["treat"][2])
            self.assertGreater(box["estop"][3], box["learn"][3])


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


def _filled(frame, xy, rgb, radius: int = 8, tol: int = 34) -> int:
    x, y = int(xy[0]), int(xy[1])
    patch = frame[max(0, x - radius): x + radius + 1, max(0, y - radius): y + radius + 1]
    dist = abs(patch.astype(int) - __import__("numpy").array(rgb)).max(axis=2)
    return int((dist < tol).sum())


def _ring(frame, xy, radius: int = 8) -> int:
    import numpy as np

    x, y = int(xy[0]), int(xy[1])
    patch = frame[max(0, x - radius): x + radius + 1, max(0, y - radius): y + radius + 1]
    ring = (
        (np.abs(patch[:, :, 0].astype(int) - 176) < 22)
        & (np.abs(patch[:, :, 1].astype(int) - 174) < 22)
        & (np.abs(patch[:, :, 2].astype(int) - 170) < 22)
    )
    return int(ring.sum())


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
        fly_l = mon.eye_lamps["Л"]
        fly_r = mon.eye_lamps["П"]
        self.assertLess(fly_l["teacher"][0], fly_r["teacher"][0])
        self.assertGreater(_filled(frame, fly_l["teacher"], (217, 136, 128)), 12)
        self.assertGreater(_filled(frame, fly_r["teacher"], (141, 181, 150)), 12)
        self.assertLess(_filled(frame, fly_r["recog"], (141, 181, 150)), 8)
        self.assertLess(_filled(frame, fly_r["recog"], (217, 136, 128)), 8)
        self.assertGreater(_filled(frame, fly_l["recog"], (141, 181, 150)), 12)
        self.assertLess(_filled(frame, fly_l["recog"], (217, 136, 128)), 8)
        banner = (mon.panel_rect.x + 10, mon.panel_rect.y + 12)
        self.assertLess(_filled(frame, banner, (217, 136, 128), radius=6), 6)
        self.assertLess(_filled(frame, banner, (141, 181, 150), radius=6), 6)
        pygame.quit()

    def test_teacher_lamp_is_blue_when_yolo_sees_that_eye(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame
        import numpy as np

        pygame.init()
        from sim.train_monitor import MonitorView, TrainMonitor
        from sim.yolo_teacher import DetBox

        mon = TrainMonitor("teacher lamps")
        camera = np.zeros((96, 160, 3), dtype=np.uint8)
        camera[:] = (24, 26, 32)
        lidar = np.zeros((64, 64, 3), dtype=np.uint8)
        yy, xx = np.ogrid[:64, :64]
        lidar[np.abs(np.hypot(xx - 31.5, yy - 31.5) - 22) < 1.4] = (0, 40, 255)
        shared = dict(
            camera=camera,
            lidar=lidar,
            learner="mb",
            overlap=0.4,
            eye_l_recognized=True,
            eye_r_recognized=False,
            learning_on=True,
            yolo_state="учит",
        )
        mon.draw(MonitorView(teacher_boxes=[DetBox(0.78, 0.2, 0.94, 0.8, 0.88)], **shared))
        frame = pygame.surfarray.array3d(mon.screen)
        from sim.train_monitor import TEACHER

        self.assertGreater(_filled(frame, mon.eye_lamps["П"]["teacher"], TEACHER), 8)
        self.assertLess(_filled(frame, mon.eye_lamps["Л"]["teacher"], TEACHER), 8)
        self.assertLess(_filled(frame, mon.eye_lamps["П"]["teacher"], (201, 120, 91)), 8)
        self.assertGreater(_ring(frame, mon.eye_lamps["Л"]["teacher"]), 8)
        mon.draw(MonitorView(teacher_boxes=[DetBox(0.45, 0.2, 0.55, 0.8, 0.4)], **shared))
        frame = pygame.surfarray.array3d(mon.screen)
        self.assertGreater(_ring(frame, mon.eye_lamps["Л"]["teacher"]), 8)
        self.assertGreater(_ring(frame, mon.eye_lamps["П"]["teacher"]), 8)
        mon.draw(
            MonitorView(
                teacher_boxes=[DetBox(0.45, 0.2, 0.55, 0.8, 0.8)],
                teacher_flash_l="ppl1",
                **shared,
            )
        )
        frame = pygame.surfarray.array3d(mon.screen)
        self.assertGreater(_filled(frame, mon.eye_lamps["Л"]["teacher"], (217, 136, 128)), 12)
        self.assertGreater(_filled(frame, mon.eye_lamps["П"]["teacher"], TEACHER), 8)
        self.assertLess(_filled(frame, mon.eye_lamps["П"]["teacher"], (201, 120, 91)), 8)
        self.assertGreater(_filled(frame, mon.eye_lamps["Л"]["recog"], (141, 181, 150)), 12)
        mon.draw(
            MonitorView(
                teacher_flash_r="pam",
                teacher_flash_l="ppl1",
                teacher_boxes=[DetBox(0.78, 0.2, 0.94, 0.8, 0.88)],
                **shared,
            )
        )
        out = __import__("pathlib").Path("/opt/cursor/artifacts")
        out.mkdir(parents=True, exist_ok=True)
        pygame.image.save(mon.present_into((1024, 768)), str(out / "trainer_window_4x3.png"))
        pygame.image.save(mon.present_into((1600, 900)), str(out / "trainer_window_16x9.png"))
        watching = dict(shared)
        watching["yolo_state"] = "смотрит"
        mon.draw(MonitorView(teacher_boxes=[DetBox(0.78, 0.2, 0.94, 0.8, 0.88)], **watching))
        frame = pygame.surfarray.array3d(mon.screen)
        lamp = (mon.teacher_rect.x + 18, mon.teacher_rect.centery)
        self.assertGreater(_filled(frame, lamp, TEACHER), 8)
        self.assertLess(_filled(frame, lamp, (201, 120, 91)), 8)
        self.assertGreater(_filled(frame, mon.eye_lamps["П"]["teacher"], TEACHER), 8)
        mon.draw(MonitorView(**shared))
        frame = pygame.surfarray.array3d(mon.screen)
        self.assertGreater(_filled(frame, lamp, (201, 120, 91)), 8)
        self.assertLess(_filled(frame, lamp, TEACHER), 8)
        from sim.ui_help import page

        for name in ("camera", "bar"):
            prose = "\n".join(line for _kind, line in page(name)[1]).lower()
            self.assertIn("сине-голуб", prose)
        shown = pygame.surfarray.array3d(mon.present_into((1024, 768)))
        blue = (shown[:, :, 2] > 180) & (shown[:, :, 0] < 50) & (shown[:, :, 1] < 90)
        xs, ys = np.where(blue)
        self.assertGreater(len(xs), 10)
        self.assertAlmostEqual(int(xs.max() - xs.min()), int(ys.max() - ys.min()), delta=4)
        pygame.quit()


class FixedColumnTests(unittest.TestCase):
    def test_readout_labels_share_columns_across_values(self):
        from sim.hemifield import format_fly_line
        from sim.pilot import format_eyes_line, format_range_line
        from sim.tabnum import (
            format_drift,
            format_familiar,
            format_kc,
            format_lifetime,
            format_pilot_mode,
            format_pilot_rest,
            format_session_counts,
            label_index,
        )
        from sim.yolo_teacher import counts_line

        eyes = [
            format_eyes_line(8, 1, True, True),
            format_eyes_line(None, None, False, True),
            format_eyes_line(-168, 84, False, False),
        ]
        from sim.tabnum import lidar_metric_rows

        sample = lidar_metric_rows(
            format_fly_line(12, 3, 9, 0.2, "bilateral"),
            format_range_line(1.5, 3.0, 2, 2, {"window": 8, "votes": 3}),
        )
        labels = [row[0] for row in sample]
        self.assertEqual(
            labels,
            ["муха L", "муха R", "Δ", "z", "вектор", "билатерально", "дальн", "вперёд", "сектор"],
        )
        self.assertEqual(sample[0][1], "+12")
        self.assertEqual(sample[5][1], "")
        self.assertEqual(sample[6][2], "м")
        self.assertEqual(sample[8][2], "3/8")
        self.assertTrue(sample[4][1].startswith("+") or sample[4][1].startswith("-"))
        self.assertEqual(sample[4][2], "°")
        sectors = lidar_metric_rows(
            format_fly_line(1, 1, 0, 0.0, "sectors"),
            format_range_line(None, None, None, None, None),
        )
        self.assertEqual(sectors[5][0], "секторы")
        self.assertEqual(sectors[4][1], "-")

        flies = [
            format_fly_line(12, 3, 9, 0.2, "bilateral"),
            format_fly_line(-84, 168, -252, -1.0, "sectors"),
            format_fly_line(None, None, None, None, "bilateral"),
        ]
        ranges = [
            format_range_line(1.5, 3.0, 2, 2, {"window": 8, "votes": 3}),
            format_range_line(12.25, None, None, 7, {"window": 8, "votes": 8, "coast": True}),
        ]
        counts = [counts_line(0, 1, 0, 0), counts_line(922, 3, 1204, 8)]
        groups = (
            (eyes, ("R_L", "R_R")),
            (flies, ("муха", "Δ", " z ")),
            (ranges, ("дальн", "вперёд", "сектор")),
            (counts, ("PAM_L", "PAM_R", "PPL1_L", "PPL1_R")),
        )
        for lines, tokens in groups:
            self.assertEqual(len({len(line) for line in lines}), 1)
            for token in tokens:
                indexes = {label_index(line, token) for line in lines}
                self.assertEqual(len(indexes), 1, token)
        self.assertEqual(label_index(eyes[0], "ОБА"), label_index(eyes[1], "ОДИ"))
        self.assertEqual(label_index(eyes[0], "ОБА"), label_index(eyes[2], "НЕТ"))
        life = [format_lifetime(1, 2, True), format_lifetime(40, 6, True), format_lifetime(0, 0, False)]
        session = [
            format_session_counts(1, 0, 4, "звук выкл"),
            format_session_counts(12, 120, 3661, "звук недоступен"),
        ]
        familiar = [format_familiar("сессия", 4, 1), format_familiar("сессия", 3661, 20)]
        drift = [format_drift(0.0), format_drift(-12.5), format_drift(2048.0)]
        kc = [format_kc(0, 5137), format_kc(514, 5137), format_kc(2000, 8)]
        modes = [format_pilot_mode("РУЧНОЕ"), format_pilot_mode("АВТОНОМИЯ"), format_pilot_mode("РУЧНОЕ · ПЕРЕХВАТ")]
        rests = [
            format_pilot_rest("оператор", "доворот (Л)", "L"),
            format_pilot_rest("мозг", "НЕТ → ПОИСК", ""),
            format_pilot_rest("оператор", "подтверждено — иду", "R"),
        ]
        for lines, tokens in (
            (life, ("PAM", "PPL1", "всего")),
            (session, ("PAM", "PPL1", "звук")),
            (familiar, ("знакомство",)),
            (drift, ("дрейф",)),
            (kc, ("KC", "из")),
            (modes, ()),
            (rests, ("ведёт:",)),
        ):
            self.assertEqual(len({len(line) for line in lines}), 1, lines[0][:24])
            for token in tokens:
                self.assertEqual(len({label_index(line, token) for line in lines}), 1, token)
        self.assertEqual(label_index(rests[0], "видели"), label_index(rests[2], "видели"))
        self.assertTrue(all(line[0] in "+-" or line.strip()[:1] in "+-" or "дрейф" in line for line in drift))

    def test_drawn_label_pixels_ignore_the_numbers(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        pygame.init()
        import numpy as np

        from sim.hemifield import format_fly_line
        from sim.pilot import format_eyes_line, format_range_line
        from sim.tabnum import format_lidar_caption, phrase, signed
        from sim.train_monitor import MonitorView, TrainMonitor
        from sim.yolo_teacher import DetBox, counts_line

        def view(scale: int) -> MonitorView:
            camera = np.zeros((90, 160, 3), dtype=np.uint8)
            camera[:] = (40, 42, 46)
            pam = 2 if scale == 0 else 922
            return MonitorView(
                title="тренировка узнавания",
                camera=camera,
                learner="mb",
                learning_on=True,
                eyes_line=format_eyes_line(8 if scale == 0 else -168, 1 if scale == 0 else 84, scale == 0, False),
                fly_line=format_fly_line(12 if scale == 0 else -40, 3 if scale == 0 else 200, 9, 0.2 if scale == 0 else -0.8, "bilateral"),
                range_line=format_range_line(1.2 if scale == 0 else 18.5, 3.0, 1 if scale == 0 else 7, 2, None),
                teacher_counts=counts_line(pam, 0 if scale == 0 else 4, 1, 0 if scale == 0 else 15),
                teacher_skips="ложных узнаваний без наказания: %s  %s"
                % (
                    signed(3 if scale == 0 else 28),
                    phrase("лимит" if scale == 0 else "нет ответа YOLO", 36),
                ),
                teacher_boxes=[DetBox(0.62, 0.2, 0.84, 0.8, 0.51 if scale == 0 else 0.93, "П")],
                n_pam=pam,
                n_ppl1=1 if scale == 0 else 40,
                total_pam=pam,
                total_ppl1=6,
                session_time=4 if scale == 0 else 3661,
                total_time=10 if scale == 0 else 7200,
                drift=0.4 if scale == 0 else -18.6,
                kc_on=12 if scale == 0 else 2000,
                kc_n=5137,
                likeness=4 if scale == 0 else -120,
                readout_caption="сырой выход: подход – избегание",
                pilot_mode="РУЧНОЕ" if scale == 0 else "АВТОНОМИЯ",
                pilot_who="оператор" if scale == 0 else "мозг",
                phase_ru="доворот (Л)" if scale == 0 else "подтверждено — иду",
                last_seen_side="L" if scale == 0 else "R",
                lidar_mode=format_lidar_caption(True, 1.5 if scale == 0 else 10),
                lidar_fresh_on=True,
                record_on=True,
                record_saved=3 if scale == 0 else 128,
                record_bytes=4000 if scale == 0 else 5_000_000,
                overlap=0.4,
                t=4 if scale == 0 else 48,
                hemi_l=8 if scale == 0 else -30,
                hemi_r=1 if scale == 0 else 12,
                yolo_state="учит",
                recog_line="узнавание: только камера",
                keys_hint="A авто  K руль  Y учитель  H рамки  F11 экран",
            )

        mon = TrainMonitor("columns")
        mon.draw(view(0))
        first = pygame.surfarray.array3d(mon.screen).copy()
        rows_a = list(mon._fixed_rows)
        anchors_a = list(mon._anchors)
        mon.draw(view(1))
        second = pygame.surfarray.array3d(mon.screen).copy()
        rows_b = list(mon._fixed_rows)
        anchors_b = list(mon._anchors)
        self.assertEqual(anchors_a, anchors_b)
        self.assertFalse(any(kind == "fallback" for kind, _x, _y in anchors_a))
        self.assertTrue(any(kind == "status" for kind, _x, _y in anchors_a))
        self.assertTrue(any(kind == "num" for kind, _x, _y in anchors_a))
        self.assertEqual(len(rows_a), len(rows_b))
        self.assertGreaterEqual(len(rows_a), 8)
        for (xa, ya, ta, sa, cell, height), (xb, yb, tb, sb, cell_b, height_b) in zip(rows_a, rows_b):
            self.assertEqual((xa, ya, sa, cell, height), (xb, yb, sb, cell_b, height_b))
            self.assertEqual(len(ta), len(tb))
            for i, (ca, cb) in enumerate(zip(ta, tb)):
                if ca != cb or ca == " ":
                    continue
                x0 = xa + i * cell
                block_a = first[x0 : x0 + cell, ya : ya + height]
                block_b = second[x0 : x0 + cell, ya : ya + height]
                np.testing.assert_array_equal(block_a, block_b)
        pygame.quit()


class QuietLaunchTests(unittest.TestCase):
    def test_owned_teacher_stops_only_its_own_process(self):
        from sim.yolo_teacher import OwnedTeacher

        stopped = []

        class Proc:
            def poll(self):
                return None

            def terminate(self):
                stopped.append("term")

            def wait(self, timeout=1):
                return 0

            def kill(self):
                stopped.append("kill")

        owned = OwnedTeacher()
        owned.proc = Proc()
        owned.owned = True
        owned.close()
        self.assertEqual(stopped, ["term"])
        self.assertFalse(owned.owned)

        foreign = OwnedTeacher()
        foreign.proc = Proc()
        foreign.owned = False
        foreign.close()
        self.assertEqual(stopped, ["term"])

    def test_dummy_driver_does_not_spawn_and_a_missing_venv_is_reported(self):
        from sim.yolo_teacher import OwnedTeacher, teacher_argv

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ.pop("RECOGNIZER_START_TEACHER", None)
        self.assertEqual(OwnedTeacher().start("http://127.0.0.1:8091"), "")
        os.environ["RECOGNIZER_START_TEACHER"] = "1"
        os.environ["RECOGNIZER_YOLO_PYTHON"] = str(Path("/tmp/no-such-yolo-python"))
        os.environ["RECOGNIZER_YOLO_WEIGHTS"] = str(Path("/tmp/no-such-best.pt"))
        os.environ["RECOGNIZER_YOLO_SERVE"] = str(Path("/tmp/no-such-serve.py"))
        try:
            owned = OwnedTeacher()
            note = owned.start("http://127.0.0.1:9")
            self.assertIn("нет файла", note)
            self.assertIsNone(owned.proc)
            self.assertIsNone(teacher_argv("http://127.0.0.1:8091", home=Path("/tmp/no-such-home")))
        finally:
            os.environ.pop("RECOGNIZER_START_TEACHER", None)
            os.environ.pop("RECOGNIZER_YOLO_PYTHON", None)
            os.environ.pop("RECOGNIZER_YOLO_WEIGHTS", None)
            os.environ.pop("RECOGNIZER_YOLO_SERVE", None)

    def test_teacher_argv_prefers_pythonw_beside_the_venv(self):
        from sim.yolo_teacher import teacher_argv

        root = Path(tempfile.mkdtemp())
        desktop = root / "Desktop" / "FlyWire-Go2"
        scripts = desktop / "yolo" / "venv" / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").write_text("", encoding="utf-8")
        (scripts / "pythonw.exe").write_text("", encoding="utf-8")
        weights = desktop / "yolo" / "runs" / "go2_mix_v2" / "weights" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"pt")
        serve = desktop / "tools" / "yolo_teacher" / "serve.py"
        serve.parent.mkdir(parents=True)
        serve.write_text("print('serve')\n", encoding="utf-8")
        argv = teacher_argv("http://127.0.0.1:8091", home=root)
        self.assertIsNotNone(argv)
        self.assertTrue(str(argv[0]).endswith("pythonw.exe"))
        self.assertEqual(argv[-1], "8091")
        self.assertIn("--weights", argv)

    def test_launchers_hide_the_console_and_keep_the_cmd_files(self):
        from pathlib import Path as RepoPath

        root = RepoPath(__file__).resolve().parents[1]
        names = (
            "DogRecognizeTrainer_sim",
            "DogRecognizeTrainer_robot",
            "DogRecognizeTrainer_onboard",
        )
        for name in names:
            cmd = (root / "launchers" / ("%s.cmd" % name)).read_text(encoding="utf-8")
            vbs = (root / "launchers" / ("%s.vbs" % name)).read_text(encoding="utf-8")
            lnk = (root / "launchers" / ("%s.lnk" % name)).read_bytes()
            self.assertIn("DogRecognizeTrainer.exe", cmd)
            self.assertIn("pause", cmd)
            self.assertIn("sh.Run cmd, 0, False", vbs)
            self.assertIn("sh.CurrentDirectory = folder", vbs)
            self.assertIn("wscript", vbs.lower())
            if name == "DogRecognizeTrainer_sim":
                self.assertIn("--sim --state logs\\mb_train_state_sim.npz", vbs)
                self.assertIn("--sim --state logs\\mb_train_state_sim.npz", cmd)
            elif name == "DogRecognizeTrainer_onboard":
                self.assertIn("--onboard 192.168.35.213", vbs)
                self.assertIn("--onboard 192.168.35.213", cmd)
                self.assertNotIn("mb_train_state_sim", vbs)
            else:
                self.assertNotIn("--sim", vbs)
                self.assertNotIn("--onboard", vbs)
                self.assertNotIn("mb_train_state_sim", vbs)
            self.assertIn((name + ".vbs").encode("utf-16-le"), lnk)
            self.assertIn(b"wscript.exe", lnk)
        teacher_cmd = (root / "tools" / "yolo_teacher" / "start_teacher.cmd").read_text(encoding="utf-8")
        teacher_vbs = (root / "tools" / "yolo_teacher" / "start_teacher.vbs").read_text(encoding="utf-8")
        self.assertIn("pause", teacher_cmd)
        self.assertIn("python.exe", teacher_cmd)
        self.assertIn("pythonw.exe", teacher_vbs)
        self.assertIn(", 0, False", teacher_vbs)
        quiet = (root / "sim" / "quiet_proc.py").read_text(encoding="utf-8")
        self.assertIn("CREATE_NO_WINDOW", quiet)
        self.assertNotIn("from_display_module", quiet)
        go2 = (root / "sim" / "go2_control.py").read_text(encoding="utf-8")
        self.assertNotIn(".terminate", go2)
        self.assertNotIn(".kill(", go2)


class _Box:
    def __init__(self, x0, x1, conf=0.9, y0=0.2, y1=0.8):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.conf = conf


class TeacherLeadTests(unittest.TestCase):
    def test_close_comes_from_the_box_and_lidar_is_not_a_source(self):
        from sim.stimuli import camera_stimuli, learning_sources

        self.assertEqual(learning_sources(), ("yolo_bbox",))
        self.assertNotIn("lidar", learning_sources())
        close, _grew, _area = camera_stimuli([_Box(0.4, 0.55)], None)
        self.assertFalse(close)
        close, _grew, _area = camera_stimuli([_Box(0.2, 0.8, y0=0.05, y1=0.95)], None)
        self.assertTrue(close)

    def test_mix_and_plaque(self):
        from sim.teacher_lead import lead_plaque, mix_velocity

        self.assertEqual(mix_velocity(0.4, -0.2, 0.0, 0.4, 0.0), (0.4, -0.2))
        self.assertEqual(mix_velocity(0.4, -0.2, 0.0, 0.4, 1.0), (0.0, 0.4))
        mixed = mix_velocity(0.4, 0.0, 0.0, 0.4, 0.5)
        self.assertAlmostEqual(mixed[0], 0.2)
        self.assertAlmostEqual(mixed[1], 0.2)
        self.assertEqual(lead_plaque(0.0), "ведёт: муха")
        self.assertEqual(lead_plaque(1.0), "ведёт: учитель")
        self.assertEqual(lead_plaque(0.7), "ведёт: смешанно 70/30")

    def test_turn_toward_the_box_and_search_the_last_side(self):
        from sim.pilot import STOP_M, X_MAX
        from sim.teacher_lead import TeacherLead

        lead = TeacherLead()
        speed, yaw = lead.update_from_boxes(0.0, [_Box(0.05, 0.25)], 3.0, 3.0)
        self.assertGreater(yaw, 0.0)
        self.assertEqual(speed, 0.0)
        speed, yaw = lead.update_from_boxes(0.2, [_Box(0.75, 0.95)], 3.0, 3.0)
        self.assertLess(yaw, 0.0)
        speed, yaw = lead.update_from_boxes(0.4, [_Box(0.45, 0.55)], 3.0, 3.0)
        self.assertAlmostEqual(yaw, 0.0)
        self.assertAlmostEqual(speed, X_MAX)
        speed, yaw = lead.update_from_boxes(0.6, [_Box(0.45, 0.55)], STOP_M, 3.0)
        self.assertAlmostEqual(speed, 0.0)
        self.assertAlmostEqual(yaw, 0.0)
        speed, yaw = lead.update_from_boxes(1.0, [], 3.0, 3.0)
        self.assertEqual(speed, 0.0)
        self.assertLess(yaw, 0.0)
        other = TeacherLead()
        other.update_from_boxes(0.0, [_Box(0.05, 0.25)], 3.0, 3.0)
        speed, yaw = other.update_from_boxes(0.2, [], 3.0, 3.0)
        self.assertGreater(yaw, 0.0)

    def test_estop_and_lidar_beat_the_teacher(self):
        from sim.teacher_lead import apply_guards, mix_velocity

        mixed = mix_velocity(0.2, 0.2, 0.4, 0.3, 1.0)
        stopped = apply_guards(mixed[0], mixed[1], True, 3.0)
        self.assertEqual(stopped, (0.0, 0.0, True))
        held = apply_guards(0.4, 0.3, False, 0.8)
        self.assertEqual(held[0], 0.0)
        self.assertAlmostEqual(held[1], 0.3)
        self.assertFalse(held[2])

    def test_conspecific_is_a_pluggable_stimulus(self):
        from sim.stimuli import Stimulus, StimulusSet
        from sim.teacher_lead import TeacherLead, attach_recognition_stimuli, observe_recognition

        board = StimulusSet()
        self.assertEqual(board.status(), [])
        attach_recognition_stimuli(board)
        lead = TeacherLead()
        lead.hit_r = True
        lead.approaching = True
        observe_recognition(board, lead, False)
        self.assertEqual(board.reinforcement(0.0, True), (None, None))
        self.assertEqual(board.status(), [{"name": "conspecific", "active": False, "value": 0.0}])
        observe_recognition(board, lead, True)
        self.assertEqual(board.reinforcement(0.0, False), (None, None))
        self.assertEqual(board.status()[0]["active"], True)
        self.assertEqual(board.status()[0]["value"], 0.0)
        self.assertEqual(board.reinforcement(0.0, True), (None, "pam"))
        self.assertEqual(board.status()[0]["value"], 1.0)
        self.assertEqual(board.reinforcement(0.1, True), (None, None))
        self.assertEqual(board.status()[0]["value"], 0.0)
        self.assertEqual(board.reinforcement(0.6, True), (None, "pam"))
        held = StimulusSet()
        attach_recognition_stimuli(held)
        eye = TeacherLead()
        eye.hit_l = True
        eye.centered = True
        observe_recognition(held, eye, True)
        self.assertEqual(held.reinforcement(1.0, True, "pam", None), (None, None))
        self.assertEqual(held.status()[0]["value"], 0.0)
        self.assertEqual(held.reinforcement(1.0, True), ("pam", None))

        class Marker(Stimulus):
            def offer(self, now, learning, skip_l=None, skip_r=None):
                self.active = True
                self.value = -1.0
                if not learning:
                    return 0.0, 0.0
                return 0.0, -1.0

        both = StimulusSet()
        attach_recognition_stimuli(both)
        both.attach(Marker("marker"))
        other = TeacherLead()
        other.hit_r = True
        other.approaching = True
        observe_recognition(both, other, True)
        self.assertEqual(both.reinforcement(0.0, True), (None, None))
        self.assertEqual([row["name"] for row in both.status()], ["conspecific", "marker"])
        self.assertEqual(both.status()[0]["value"], 1.0)
        self.assertEqual(both.status()[1], {"name": "marker", "active": True, "value": -1.0})

    def test_outcome_rewards_and_a_match_does_not(self):
        from sim.teacher_lead import TeacherLead, apply_guards, commands_match, outcome_kinds

        self.assertEqual(
            outcome_kinds(True, False, True, False, False, False),
            ("pam", None),
        )
        self.assertEqual(
            outcome_kinds(False, True, False, True, False, False),
            (None, "pam"),
        )
        self.assertEqual(
            outcome_kinds(True, True, True, False, False, True),
            ("ppl1", "ppl1"),
        )
        self.assertEqual(
            outcome_kinds(False, False, False, False, True, False, True, False),
            ("ppl1", None),
        )
        self.assertTrue(commands_match(0.4, 0.0, 0.4, 0.02))
        self.assertFalse(commands_match(0.4, 0.0, 0.0, 0.4))

        agreed = TeacherLead()
        agreed.hit_l = True
        self.assertEqual(agreed.action_kinds(0.0, True, True), (None, None))

        centred = TeacherLead()
        centred.update_from_boxes(0.0, [_Box(0.45, 0.55)], 3.0, 3.0)
        centred.set_hits(True, True)
        self.assertTrue(centred.centered)
        self.assertFalse(centred.approaching)
        self.assertEqual(centred.action_kinds(0.0, True, True), ("pam", "pam"))
        self.assertEqual(centred.action_kinds(0.1, True, True), (None, None))
        self.assertEqual(centred.action_kinds(0.0, False, True), (None, None))
        quiet = TeacherLead()
        quiet.update_from_boxes(0.0, [_Box(0.45, 0.55)], 3.0, 3.0)
        quiet.set_hits(True, False)
        self.assertEqual(quiet.action_kinds(1.0, True, False), (None, None))

        closing = TeacherLead()
        closing.update_from_boxes(0.0, [_Box(0.75, 0.85)], 3.0, 3.0)
        closing.set_hits(False, True)
        self.assertEqual(closing.action_kinds(0.0, True, True), (None, None))
        closing.update_from_boxes(0.6, [_Box(0.70, 0.90)], 3.0, 3.0)
        closing.set_hits(False, True)
        self.assertTrue(closing.approaching)
        self.assertFalse(closing.too_close)
        self.assertEqual(closing.action_kinds(0.6, True, True), (None, "pam"))

        lost = TeacherLead()
        lost.update_from_boxes(0.0, [_Box(0.05, 0.25)], 3.0, 3.0)
        lost.set_hits(True, False)
        lost.update_from_boxes(0.6, [], 3.0, 3.0)
        lost.set_hits(False, False)
        self.assertTrue(lost.lost)
        self.assertEqual(lost.action_kinds(0.6, True, True), ("ppl1", None))

        large = _Box(0.25, 0.75, y0=0.05, y1=0.95)
        close = TeacherLead()
        close.update_from_boxes(0.0, [large], 3.0, 3.0)
        close.set_hits(True, True)
        self.assertTrue(close.too_close)
        self.assertEqual(close.action_kinds(0.0, True, True), ("ppl1", "ppl1"))
        close.update_from_boxes(0.6, [large], 0.4, 0.3)
        close.set_hits(True, True)
        self.assertTrue(close.too_close)

        blocked = TeacherLead()
        blocked.update_from_boxes(0.0, [_Box(0.75, 0.85)], 3.0, 3.0)
        blocked.set_hits(False, True)
        self.assertFalse(blocked.too_close)
        blocked.update_from_boxes(0.6, [_Box(0.75, 0.85)], 0.4, 0.8)
        blocked.set_hits(False, True)
        self.assertFalse(blocked.too_close)
        self.assertFalse(blocked.approaching)
        self.assertEqual(blocked.action_kinds(0.6, True, True), (None, None))
        held_reflex = apply_guards(0.4, 0.3, False, 0.8)
        self.assertEqual(held_reflex[0], 0.0)
        self.assertAlmostEqual(held_reflex[1], 0.3)

        other_kind = TeacherLead()
        other_kind.hit_l = True
        other_kind.centered = True
        self.assertEqual(other_kind.action_kinds(0.0, True, True), ("pam", None))
        other_kind.centered = False
        other_kind.too_close = True
        self.assertEqual(other_kind.action_kinds(0.1, True, True), ("ppl1", None))
        same = TeacherLead()
        same.hit_l = True
        same.centered = True
        self.assertEqual(same.action_kinds(0.0, True, True, "pam", None), (None, None))

        agreed.note(0.0, True)
        agreed.note(1.0, False)
        self.assertAlmostEqual(agreed.fraction(1.0), 0.5)

    def test_onboard_mix_estop_and_match_leave_the_npz_alone(self):
        import tempfile
        from pathlib import Path

        import numpy as np

        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import MbTrainer, default_npz
        from sim.npz_compat import open_npz

        def loop_at(now):
            brain = MbTrainer(default_npz(), seed=1, overlap=0.4)
            loop = BrainLoop(SportDrive(None), brain, "/tmp/mb_teacher_lead.npz")
            loop.learn = True
            loop.mb.learn = True
            camera = np.zeros((32, 32, 3), dtype=np.uint8)
            loop.ingest(camera, np.zeros_like(camera), now)
            loop.handle({"op": "autonomy_on"}, now)
            loop.frames_ok = True
            loop.last_frame_t = now
            loop.last_hb = now
            loop.recognized = True
            loop.recognized_l = True
            loop.recognized_r = True
            loop.mb.r_l = 4.0
            loop.mb.r_r = 4.0
            loop.dist_m = 3.0
            loop.forward_m = 3.0
            return loop

        agree = loop_at(5.0)
        before_pam = agree.mb.teacher_pam_l
        before_ppl = agree.mb.teacher_ppl1_l
        agree.handle(
            {"op": "teacher_aim", "x": 0.4, "z": 0.0, "weight": 1.0, "hit_l": True, "hit_r": False},
            5.0,
        )
        cmd = agree.tick(5.0)
        self.assertAlmostEqual(cmd.x, 0.4)
        self.assertAlmostEqual(cmd.z, 0.0)
        self.assertEqual(agree.mb.teacher_pam_l, before_pam)
        self.assertEqual(agree.mb.teacher_ppl1_l, before_ppl)
        self.assertEqual(agree.status()["lead"], "учитель")
        self.assertAlmostEqual(agree.status()["teacher_weight"], 1.0)
        self.assertAlmostEqual(agree.status()["fly_x"], 0.4)
        self.assertAlmostEqual(agree.status()["teacher_x"], 0.4)
        self.assertAlmostEqual(agree.status()["lead_match"], 1.0)

        reward = loop_at(6.0)
        before = reward.mb.teacher_pam_l
        reward.handle(
            {
                "op": "teacher_aim",
                "x": 0.4,
                "z": 0.0,
                "weight": 1.0,
                "hit_l": True,
                "hit_r": False,
                "centered": True,
            },
            6.0,
        )
        reward.tick(6.0)
        self.assertEqual(reward.mb.teacher_pam_l, before + 1)
        reward_stim = reward.status()["stimuli"]
        self.assertEqual(reward_stim, [{"name": "conspecific", "active": True, "value": 1.0}])

        punish = loop_at(7.0)
        before_pam = punish.mb.teacher_pam_l
        before_ppl = punish.mb.teacher_ppl1_l
        punish.handle(
            {
                "op": "teacher_aim",
                "x": 0.0,
                "z": 0.35,
                "weight": 1.0,
                "lost": True,
                "last_l": True,
            },
            7.0,
        )
        punish.tick(7.0)
        self.assertEqual(punish.mb.teacher_ppl1_l, before_ppl + 1)
        self.assertEqual(punish.mb.teacher_pam_l, before_pam)
        self.assertEqual(punish.status()["stimuli"], [{"name": "conspecific", "active": True, "value": -1.0}])

        miss = loop_at(5.0)
        before = miss.mb.teacher_pam_l
        miss.handle(
            {"op": "teacher_aim", "x": 0.0, "z": 0.8, "weight": 0.5, "hit_l": True, "hit_r": True},
            5.0,
        )
        cmd = miss.tick(5.0)
        self.assertAlmostEqual(cmd.x, 0.2)
        self.assertAlmostEqual(cmd.z, 0.4)
        self.assertEqual(miss.mb.teacher_pam_l, before)
        self.assertEqual(miss.status()["stimuli"], [{"name": "conspecific", "active": True, "value": 0.0}])
        self.assertEqual(miss.status()["lead"], "смешанно")
        self.assertAlmostEqual(miss.status()["fly_z"], 0.0)
        self.assertAlmostEqual(miss.status()["teacher_z"], 0.8)

        blocked = loop_at(5.0)
        blocked.forward_m = 0.8
        blocked.handle(
            {"op": "teacher_aim", "x": 0.4, "z": 0.3, "weight": 1.0, "hit_l": False, "hit_r": False},
            5.0,
        )
        cmd = blocked.tick(5.0)
        self.assertEqual(cmd.x, 0.0)
        self.assertAlmostEqual(cmd.z, 0.3)

        halted = loop_at(5.0)
        halted.handle({"op": "teacher_aim", "x": 0.4, "z": 0.3, "weight": 1.0}, 5.0)
        halted.handle({"op": "estop"}, 5.0)
        cmd = halted.tick(5.1)
        self.assertEqual(cmd.x, 0.0)
        self.assertEqual(cmd.z, 0.0)
        self.assertEqual(halted.pilot.mode, "estop")
        self.assertTrue(all(abs(move[0]) < 1e-9 and abs(move[2]) < 1e-9 for move in halted.drive.moves))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.npz"
            agree.mb.save(path)
            with open_npz(path) as saved:
                self.assertNotIn("teacher_weight", saved.files)
                self.assertNotIn("lead_match", saved.files)
            fresh = MbTrainer(default_npz(), seed=1, overlap=0.4)
            fresh.load(path)
            self.assertEqual(fresh.teacher_pam_l, 0)

    def test_pause_blocks_target_loss_until_motion_and_grace(self):
        import json
        import urllib.request

        import numpy as np

        from robot.flybrain_onboard.app import BrainLoop, serve
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import PAUSE_GRACE_S, MbTrainer, default_npz

        self.assertEqual(PAUSE_GRACE_S, 1.0)

        def loop_at(now):
            brain = MbTrainer(default_npz(), seed=1, overlap=0.4)
            loop = BrainLoop(SportDrive(None), brain, "/tmp/mb_pause_hold.npz")
            loop.learn = True
            loop.mb.learn = True
            camera = np.zeros((32, 32, 3), dtype=np.uint8)
            loop.ingest(camera, np.zeros_like(camera), now)
            loop.handle({"op": "autonomy_on"}, now)
            loop.frames_ok = True
            loop.last_frame_t = now
            loop.last_hb = now
            loop.recognized = True
            loop.recognized_l = True
            loop.recognized_r = True
            loop.mb.r_l = 4.0
            loop.mb.r_r = 4.0
            loop.dist_m = 3.0
            loop.forward_m = 3.0
            return loop

        def lose(loop, now):
            loop.frames_ok = True
            loop.last_frame_t = now
            loop.handle(
                {
                    "op": "teacher_aim",
                    "x": 0.0,
                    "z": 0.35,
                    "weight": 1.0,
                    "lost": True,
                    "last_l": True,
                },
                now,
            )

        paused = loop_at(5.0)
        quiet = paused.status()
        self.assertFalse(quiet["paused"])
        self.assertFalse(quiet["learning_frozen"])
        left = np.array(paused.mb.brain.kc_mbon_w, copy=True)
        right = np.array(paused.mb.brain_r.kc_mbon_w, copy=True)
        before = int(paused.mb.n_ppl1)
        paused.handle({"op": "pause"}, 5.0)
        lose(paused, 5.1)
        cmd = paused.tick(5.2)
        self.assertEqual((cmd.x, cmd.z), (0.0, 0.0))
        held = paused.status()
        self.assertTrue(held["paused"])
        self.assertTrue(held["learning_frozen"])
        self.assertFalse(held["ppl1_event"])
        self.assertEqual(paused.mb.n_ppl1, before)
        self.assertEqual(paused.mb.teacher_ppl1_l, 0)
        np.testing.assert_array_equal(paused.mb.brain.kc_mbon_w, left)
        np.testing.assert_array_equal(paused.mb.brain_r.kc_mbon_w, right)

        paused.handle({"op": "punish"}, 5.3)
        paused.ingest(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8), 5.3)
        self.assertFalse(paused.status()["ppl1_event"])
        self.assertEqual(paused.mb.n_ppl1, before)
        np.testing.assert_array_equal(paused.mb.brain.kc_mbon_w, left)
        np.testing.assert_array_equal(paused.mb.brain_r.kc_mbon_w, right)

        paused.handle({"op": "resume"}, 8.0)
        lose(paused, 8.0)
        paused.tick(8.0)
        self.assertFalse(paused.status()["paused"])
        self.assertTrue(paused.status()["learning_frozen"])
        self.assertEqual(paused.mb.n_ppl1, before)
        np.testing.assert_array_equal(paused.mb.brain.kc_mbon_w, left)

        lose(paused, 8.0 + PAUSE_GRACE_S)
        paused.tick(8.0 + PAUSE_GRACE_S)
        self.assertFalse(paused.status()["learning_frozen"])
        self.assertEqual(paused.mb.n_ppl1, before + 1)
        self.assertTrue(paused.status()["ppl1_event"])
        self.assertEqual(paused.mb.teacher_ppl1_l, 1)

        stopped = loop_at(5.0)
        stopped.handle({"op": "estop"}, 5.0)
        lose(stopped, 5.1)
        cmd = stopped.tick(5.2)
        self.assertEqual((cmd.x, cmd.z), (0.0, 0.0))
        self.assertFalse(stopped.status()["paused"])
        self.assertTrue(stopped.status()["learning_frozen"])
        self.assertEqual(stopped.mb.n_ppl1, 0)
        self.assertFalse(stopped.status()["ppl1_event"])

        external = loop_at(5.0)
        external.handle({"op": "external_stop"}, 5.0)
        lose(external, 5.1)
        cmd = external.tick(5.2)
        self.assertEqual((cmd.x, cmd.z), (0.0, 0.0))
        self.assertTrue(external.status()["learning_frozen"])
        self.assertEqual(external.mb.n_ppl1, 0)

        served = loop_at(1.0)
        server = serve(served, 0)
        port = int(server.server_address[1])
        try:
            def post(path, payload):
                req = urllib.request.Request(
                    "http://127.0.0.1:%d%s" % (port, path),
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            body = post("/pause", {})
            self.assertTrue(body["paused"])
            self.assertTrue(body["learning_frozen"])
            body = post("/cmd", {"op": "resume"})
            self.assertFalse(body["paused"])
            self.assertTrue(body["learning_frozen"])
        finally:
            server.shutdown()
            server.server_close()

    def test_sim_estop_drops_a_punish(self):
        import numpy as np

        from sim.recognize_train import RecognizeTrainSim

        session = RecognizeTrainSim(n_agents=1, seed=1, learner="mb", dan="teacher", auto_teach=False)
        session.pilot.estop()
        left = np.array(session.mb.brain.kc_mbon_w, copy=True)
        before = int(session.mb.n_ppl1)
        session.step(0.4, 0.2, learn=True, teach="ppl1")
        self.assertEqual(session.mb.n_ppl1, before)
        self.assertTrue(session.mb.learning_hold)
        np.testing.assert_array_equal(session.mb.brain.kc_mbon_w, left)


class YoloConfSliderTests(unittest.TestCase):
    def test_default_conf_drops_a_room_object(self):
        from sim.yolo_teacher import DEFAULT_CONF, DetBox, clamp_conf, hemisphere_hits

        self.assertGreaterEqual(DEFAULT_CONF, 0.75)
        self.assertEqual(clamp_conf(1.4), 1.0)
        self.assertEqual(clamp_conf(-0.2), 0.0)
        weak = DetBox(0.30, 0.2, 0.55, 0.8, 0.55)
        strong = DetBox(0.30, 0.2, 0.55, 0.8, 0.92)
        self.assertEqual(hemisphere_hits([weak], 0.4), (False, False))
        hit_l, hit_r = hemisphere_hits([strong], 0.4)
        self.assertTrue(hit_l or hit_r)

    def test_slider_sets_the_teacher_and_stays_grey(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        from sim.recognize_train_live import take_teacher_conf
        from sim.train_monitor import PEACH, MonitorInput, MonitorView, TrainMonitor
        from sim.yolo_teacher import DEFAULT_CONF, TeacherRuntime

        class Session:
            teacher_conf = None
            teacher = TeacherRuntime(conf=DEFAULT_CONF)

        session = Session()
        take_teacher_conf(session, MonitorInput())
        self.assertIsNone(session.teacher_conf)
        self.assertAlmostEqual(session.teacher.conf, DEFAULT_CONF)
        take_teacher_conf(session, MonitorInput(teacher_conf=0.91))
        self.assertAlmostEqual(session.teacher_conf, 0.91)
        self.assertAlmostEqual(session.teacher.conf, 0.91)
        take_teacher_conf(session, MonitorInput(teacher_conf=4.0))
        self.assertAlmostEqual(session.teacher.conf, 1.0)

        pygame.init()
        try:
            mon = TrainMonitor("yolo-conf")
            mon._on_resize(1920, 1080)
            mon.draw(MonitorView(learner="mb", learning_on=True, teacher_conf=DEFAULT_CONF, yolo_state="учит"))
            track = mon.conf_slider_rect
            self.assertEqual(track.h, 6)
            self.assertGreater(track.w, 40)
            filled = self._rgb(mon, track.x + 4, track.centery)
            self.assertLess(abs(filled[0] - filled[1]), 20)
            self.assertLess(abs(filled[1] - filled[2]), 20)
            self.assertNotEqual(filled, PEACH)
            empty_x = track.x + int(track.w * 0.92)
            empty = self._rgb(mon, empty_x, track.centery)
            self.assertNotEqual(empty, PEACH)
            self.assertGreater(abs(filled[0] - empty[0]) + abs(filled[1] - empty[1]), 20)

            point = [0, 0]
            from sim.train_monitor import window_from_logical

            logical = (track.x + int(track.w * 0.40), track.centery)
            wx, wy = window_from_logical(logical, mon.window.get_width(), mon.window.get_height(), (mon.scroll_x, mon.scroll_y))
            point[0], point[1] = int(wx), int(wy)
            real = pygame.mouse.get_pos
            pygame.mouse.get_pos = lambda: (int(point[0]), int(point[1]))
            try:
                pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(point[0]), int(point[1])), button=1))
                inp = mon.pump()
            finally:
                pygame.mouse.get_pos = real
            self.assertIsNotNone(inp.teacher_conf)
            self.assertAlmostEqual(float(inp.teacher_conf), 0.40, delta=0.03)
            self.assertIsNone(inp.teacher_weight)

            out = __import__("pathlib").Path("/opt/cursor/artifacts/yolo_conf_slider.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            crop = pygame.Rect(max(0, track.x - 210), max(0, track.y - 14), track.w + 226, 40)
            crop.clamp_ip(mon.screen.get_rect())
            shot = mon.screen.subsurface(crop).copy()
            big = pygame.transform.smoothscale(shot, (shot.get_width() * 2, max(1, shot.get_height() * 2)))
            pygame.image.save(big, str(out))
        finally:
            pygame.quit()

    def _rgb(self, mon, x, y):
        color = mon.screen.get_at((int(x), int(y)))
        return (color.r, color.g, color.b)


if __name__ == "__main__":
    unittest.main()
