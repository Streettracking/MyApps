import unittest
from types import SimpleNamespace

from sim import recognize_train_live as live
from sim.yolo_teacher import DetBox


def _session(boxes, cam_t=100.0, box_t=100.0):
    teacher = SimpleNamespace(on=True, conf=0.5, service_down=False, boxes_t=box_t, _teach_boxes=list(boxes), boxes=list(boxes))
    return SimpleNamespace(teacher=teacher, teacher_weight=1.0, learn=False, _cam_t=cam_t)


class TeacherYoloStateTest(unittest.TestCase):
    def setUp(self):
        self._orig = live._learning_now
        live._learning_now = lambda _s: False

    def tearDown(self):
        live._learning_now = self._orig

    def test_box(self):
        s = _session([DetBox(0.4, 0.4, 0.6, 0.6, 0.9, "dog")])
        self.assertEqual(live.teacher_yolo_state(s, 100.1), {"yolo_fresh": True, "yolo_box": True})

    def test_no_box_is_fresh_and_empty(self):
        self.assertEqual(live.teacher_yolo_state(_session([]), 100.1), {"yolo_fresh": True, "yolo_box": False})

    def test_weak_box_is_no_box(self):
        s = _session([DetBox(0.4, 0.4, 0.6, 0.6, 0.2, "dog")])
        self.assertEqual(live.teacher_yolo_state(s, 100.1)["yolo_box"], False)

    def test_stale_is_not_fresh(self):
        s = _session([DetBox(0.4, 0.4, 0.6, 0.6, 0.9, "dog")])
        self.assertEqual(live.teacher_yolo_state(s, 200.0), {"yolo_fresh": False, "yolo_box": False})

    def test_teacher_aim_carries_fields(self):
        posts = []
        onboard = SimpleNamespace(post=lambda op, **kw: posts.append((op, kw)))
        s = _session([])
        s.onboard = True
        s.remote = {"learning": False}
        s._teacher_halt = False
        live.publish_teacher_aim(s, onboard, 100.1, None, None)
        op, kw = posts[-1]
        self.assertEqual(op, "teacher_aim")
        self.assertIn("yolo_box", kw)
        self.assertIn("yolo_fresh", kw)
        self.assertFalse(kw["yolo_box"])


if __name__ == "__main__":
    unittest.main()