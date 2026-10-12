"""Onboard: 0.5 m forward lidar stop while the teacher leads, 1.0 m otherwise; floor filter."""

import unittest

import numpy as np

from robot.flybrain_onboard.app import TEACHER_LEAD_STOP_M, BrainLoop
from robot.flybrain_onboard.drive import SportDrive
from robot.flybrain_onboard import sensors


class LeadLidar(unittest.TestCase):
    def _loop(self):
        from sim.mb_train import MbTrainer

        loop = BrainLoop.__new__(BrainLoop)
        from sim.teacher_lead import TeacherLead

        loop.lead = TeacherLead()
        loop.teacher_weight = 0.0
        return loop

    def test_guard_range(self):
        loop = self._loop()
        self.assertEqual(loop._guard_range(10.0, 0.63), 0.63, "fly only keeps the 1 m stop")
        loop.teacher_weight = 1.0
        loop.lead.set_remote(10.0, 0.4, 0.0, False, False, 1.0)
        self.assertIsNone(loop._guard_range(10.1, 0.63), "teacher leading: 0.63 m does not stop")
        self.assertEqual(loop._guard_range(10.1, 0.45), 0.45, "nearer than 0.5 m still stops")
        self.assertEqual(loop._guard_range(11.0, 0.63), 0.63, "stale teacher_aim: back to 1 m")
        self.assertEqual(TEACHER_LEAD_STOP_M, 0.5)

    def test_floor_filter(self):
        rng = np.random.default_rng(0)
        ring = rng.uniform(0.7, 2.4, 200)
        ang = rng.uniform(-np.pi, np.pi, 200)
        floor = np.c_[ring * np.cos(ang), ring * np.sin(ang), rng.uniform(-0.06, -0.02, 200)]
        dog = np.c_[np.full(20, 3.0), rng.uniform(-0.2, 0.2, 20), rng.uniform(0.2, 0.4, 20)]
        pts = np.vstack([floor, dog]).astype(np.float32)
        kept = sensors.drop_floor(pts, np.zeros(2, dtype=np.float32))
        self.assertEqual(len(kept), 20)
        self.assertEqual(sensors.FLOOR_CUT_M, 0.20)
        few = pts[:5]
        self.assertEqual(len(sensors.drop_floor(few, np.zeros(2))), 5)


if __name__ == "__main__":
    unittest.main()
