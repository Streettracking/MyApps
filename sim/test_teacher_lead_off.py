"""Teacher leads with learning off: YOLO is asked, no PAM/PPL1, stale frames stop it."""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from sim import recognize_train_live as live
from sim.teacher_lead import TeacherLead, approach_velocity, teacher_status_line, TEACHER_X_CAP
from sim.yolo_teacher import DetBox, TeacherRuntime, should_query


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeOnboard:
    def __init__(self):
        self.posts = []

    def post(self, op, **kw):
        self.posts.append((op, kw))

    def ops(self):
        return [p[0] for p in self.posts]


def make_session(weight=1.0, learn=False):
    teacher = TeacherRuntime("http://127.0.0.1:1")
    teacher._ensure = lambda: None  # no thread, no HTTP
    s = SimpleNamespace(
        teacher=teacher,
        learner_kind="mb",
        learn=learn,
        onboard=True,
        remote={"learning": learn, "autonomy": True},
        teacher_weight=weight,
        mb=None,
        eye_l_recognized=False,
        eye_r_recognized=False,
        overlap=0.4,
        log=[],
        yolo_probe=False,
        kin_label=".128",
    )
    s.now = lambda: 0.0
    s._log = s.log.append
    return s


def answer(session, boxes, frame_t):
    t = session.teacher
    with t._lock:
        pend = t._pending
        t._pending = None
        gen = pend[0] if pend else t._gen
        t._result = (gen, boxes, frame_t)


def box(cx, h=0.2, conf=0.9):
    return DetBox(cx - 0.05, 0.5 - h / 2, cx + 0.05, 0.5 + h / 2, conf)


class LeadOffTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self._mono = live._mono
        live._mono = self.clock

    def tearDown(self):
        live._mono = self._mono

    def tick(self, s, jpeg, fid, state, onboard, inp=None):
        inp = inp or SimpleNamespace(focused=True, estop=False, stop=False, takeover=False, steer_x=0.0, steer_z=0.0)
        s._teacher_halt = live.teacher_halt_now(inp, s.remote)
        live._drive_teacher(s, jpeg, fid, state, False, onboard)
        s._aim_post_t = -1.0
        live.publish_teacher_aim(s, onboard, self.clock.t, None, None)
        return onboard.posts[-1][1]

    def test_should_query_lead(self):
        self.assertFalse(should_query(False, False, False, False, 99.0))
        self.assertTrue(should_query(False, False, False, False, 99.0, lead=True))
        self.assertTrue(should_query(False, False, True, False, 99.0, lead=True))

    def test_lead_learning_off_queries_and_aims_without_pam(self):
        s = make_session(weight=1.0, learn=False)
        ob = FakeOnboard()
        state = {}
        self.tick(s, b"jpeg-1", 1, state, ob)
        self.assertIsNotNone(s.teacher._pending, "frame must be offered to YOLO")
        answer(s, [box(0.5)], self.clock.t)
        self.clock.t += 0.05
        aim = self.tick(s, b"jpeg-2", 2, state, ob)
        self.assertGreater(aim["x"], 0.0)
        self.assertLessEqual(aim["x"], TEACHER_X_CAP + 1e-9)
        self.assertEqual(s.teacher_state, "walk")
        self.assertEqual(s.teacher_status, "иду к .128")
        self.assertNotIn("teach_sides", ob.ops())
        self.assertFalse(aim["hit_l"] or aim["hit_r"])
        self.assertEqual(aim["yolo_l"], [0.0] * len(aim["yolo_l"]))
        self.assertEqual(live._teacher_note_view(SimpleNamespace(**{**vars(s), "pilot": SimpleNamespace(autonomy=True)})), "PAM/PPL1 не идут")

    def test_turn_first_outside_cone(self):
        s = make_session()
        ob = FakeOnboard()
        state = {}
        self.tick(s, b"a", 1, state, ob)
        answer(s, [box(0.85)], self.clock.t)
        aim = self.tick(s, b"b", 2, state, ob)
        self.assertEqual(aim["x"], 0.0)
        self.assertLess(aim["z"], 0.0)  # image right yaws right
        self.assertEqual(s.teacher_status, "поворот")

    def test_yaw_matches_pre_walk_teacher_velocity(self):
        from sim.teacher_lead import teacher_velocity, teacher_yaw

        for cx in (0.0, 0.1, 0.3, 0.45, 0.5, 0.53, 0.6, 0.75, 0.9, 1.0):
            _x, z_old = teacher_velocity(cx, None, None, 0.0, 1.0)
            self.assertAlmostEqual(teacher_yaw(cx - 0.5), z_old, places=9, msg=cx)
        x, z, st = approach_velocity(box(0.95), None, 0.45)
        self.assertEqual((x, st), (0.0, "turn"))
        self.assertLess(z, 0.0)
        x, z, st = approach_velocity(box(0.55), None, 0.45)
        self.assertEqual(st, "walk")
        self.assertAlmostEqual(x, 0.40)

    def test_near_stands(self):
        x, z, st = approach_velocity(box(0.5, h=0.6), None, 0.45)
        self.assertEqual((x, z, st), (0.0, 0.0, "near"))
        self.assertEqual(teacher_status_line(st), "рядом, стою")

    def test_lidar_safety_stop_small_box(self):
        x, _z, st = approach_velocity(box(0.5, h=0.1), 0.35, 0.45)
        self.assertEqual(st, "walk", "lidar stop is off by default")
        x, _z, st = approach_velocity(box(0.5, h=0.1), 0.63, 0.45, lidar_stop_m=0.4)
        self.assertEqual((x, st), (0.0, "lidar"), "enabled stop is raised to 0.7 m")
        x, _z, st = approach_velocity(box(0.5, h=0.1), 0.8, 0.45, lidar_stop_m=0.7)
        self.assertEqual(st, "walk")
        self.assertEqual(teacher_status_line("lidar", "", 0.63), "препятствие 0,63 м")
        x, _z, st = approach_velocity(box(0.5, h=0.1), 0.35, 0.45, lidar_stop_m=0.0)
        self.assertEqual(st, "walk")
        x, _z, st = approach_velocity(box(0.5, h=0.1), 3.0, 0.45)
        self.assertGreater(x, 0.0)

    def test_no_box_zero(self):
        s = make_session()
        ob = FakeOnboard()
        state = {}
        self.tick(s, b"a", 1, state, ob)
        answer(s, [box(0.5)], self.clock.t)
        self.assertGreater(self.tick(s, b"b", 2, state, ob)["x"], 0.0)
        answer(s, [], self.clock.t)  # box lost while walking
        aim = self.tick(s, b"c", 3, state, ob)
        self.assertEqual((aim["x"], aim["z"]), (0.0, 0.0))
        self.assertEqual(s.teacher_status, "нет рамки")

    def test_stale_frame_zero_then_resume(self):
        s = make_session()
        ob = FakeOnboard()
        state = {}
        self.tick(s, b"same", 1, state, ob)
        answer(s, [box(0.5)], self.clock.t)
        self.assertGreater(self.tick(s, b"same", 2, state, ob)["x"], 0.0)
        self.clock.t += 0.6  # frozen JPEG, no new bytes
        aim = self.tick(s, b"same", 3, state, ob)
        self.assertEqual((aim["x"], aim["z"]), (0.0, 0.0))
        self.assertEqual(s.teacher_status, "кадр старый")
        aim = self.tick(s, b"fresh", 4, state, ob)
        self.assertEqual(aim["x"], 0.0, "old box from the frozen frame must not drive")
        answer(s, [box(0.5)], self.clock.t)
        aim = self.tick(s, b"fresh2", 5, state, ob)
        self.assertGreater(aim["x"], 0.0)

    def test_weight_zero_learning_off_no_query(self):
        s = make_session(weight=0.0, learn=False)
        ob = FakeOnboard()
        self.tick(s, b"a", 1, {}, ob)
        self.assertIsNone(s.teacher._pending)
        self.assertEqual(s.teacher_state, "no_query")
        self.assertEqual(teacher_status_line("no_query"), "YOLO не опрашивается")

    def test_takeover_arrow_focus_stop(self):
        for kw in ({"focused": False}, {"takeover": True}, {"steer_z": 1.0}, {"estop": True}):
            s = make_session()
            ob = FakeOnboard()
            state = {}
            self.tick(s, b"a", 1, state, ob)
            answer(s, [box(0.5)], self.clock.t)
            self.assertGreater(self.tick(s, b"b", 2, state, ob)["x"], 0.0)
            base = dict(focused=True, estop=False, stop=False, takeover=False, steer_x=0.0, steer_z=0.0)
            base.update(kw)
            aim = self.tick(s, b"c", 3, state, ob, SimpleNamespace(**base))
            self.assertEqual((aim["x"], aim["z"]), (0.0, 0.0), kw)
            self.assertEqual(s.teacher_state, "halt")

    def test_halt_bypasses_throttle(self):
        s = make_session()
        ob = FakeOnboard()
        s._teacher_halt = False
        live.publish_teacher_aim(s, ob, 10.0, None, None)
        s._teacher_halt = True
        live.publish_teacher_aim(s, ob, 10.01, None, None)
        self.assertEqual(len(ob.posts), 2)

    def test_learning_on_still_teaches_path(self):
        s = make_session(weight=1.0, learn=True)
        self.assertTrue(s.teacher.wants_frame(True))


if __name__ == "__main__":
    unittest.main()
