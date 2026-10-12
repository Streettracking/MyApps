import unittest
from types import SimpleNamespace

from sim import recognize_train_live as live
from sim.pilot import SEARCH_TURN
from sim.teacher_lead import TeacherLead
from sim.yolo_teacher import DetBox


def _session(t0):
    teacher = SimpleNamespace(on=True, conf=0.5, service_down=False, boxes_t=t0, _teach_boxes=[], boxes=[])
    return SimpleNamespace(teacher=teacher, teacher_weight=1.0, learn=False, _cam_t=t0, _teacher_halt=False)


def _box(cx):
    return DetBox(cx - 0.05, 0.4, cx + 0.05, 0.6, 0.9, "dog")


class LostTurnTest(unittest.TestCase):
    def setUp(self):
        self._orig = live._learning_now
        live._learning_now = lambda _s: False

    def tearDown(self):
        live._learning_now = self._orig

    def run_at(self, s, now, boxes):
        s.teacher._teach_boxes = list(boxes)
        s.teacher.boxes_t = now
        s._cam_t = now
        lead = TeacherLead()
        live._apply_approach(s, lead, None, now)
        return lead.teacher_x, lead.teacher_z

    def seen_then_lost(self, cx):
        s = _session(0.0)
        self.run_at(s, 0.0, [_box(cx)])
        return s

    def test_left_turns_left(self):
        s = self.seen_then_lost(0.1)
        x, z = self.run_at(s, 0.1, [])
        self.assertEqual(x, 0.0)
        self.assertAlmostEqual(z, SEARCH_TURN)

    def test_right_turns_right(self):
        s = self.seen_then_lost(0.9)
        x, z = self.run_at(s, 0.1, [])
        self.assertEqual(x, 0.0)
        self.assertAlmostEqual(z, -SEARCH_TURN)

    def test_stops_after_timeout(self):
        s = self.seen_then_lost(0.1)
        self.run_at(s, 0.1, [])
        self.assertEqual(self.run_at(s, 0.1 + live.LOST_TURN_S + 0.1, []), (0.0, 0.0))

    def test_stale_is_zero(self):
        s = self.seen_then_lost(0.1)
        s.teacher._teach_boxes = []
        lead = TeacherLead()
        live._apply_approach(s, lead, None, 0.1 + live.FRAME_STALE_S + 1.0)
        self.assertEqual((lead.teacher_x, lead.teacher_z), (0.0, 0.0))

    def test_halt_is_zero(self):
        s = self.seen_then_lost(0.1)
        s._teacher_halt = live.teacher_halt_now(SimpleNamespace(focused=False))
        self.assertEqual(self.run_at(s, 0.1, []), (0.0, 0.0))
        s._teacher_halt = live.teacher_halt_now(SimpleNamespace(focused=True, estop=False, stop=False, takeover=False, steer_x=0.0, steer_z=1.0))
        self.assertEqual(self.run_at(s, 0.2, []), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
