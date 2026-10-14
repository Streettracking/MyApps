"""Neutral cube, reinforcement off, and the scripted memory probe.

The memory probe uses the moving-matched control: same speed and the same
trajectory as the conspecific. A static option stays available for a still scene.

Run: python3 -m unittest sim.test_control_probe
"""

from __future__ import annotations

import unittest

import numpy as np


class ControlSceneTests(unittest.TestCase):
    def test_cube_is_not_a_conspecific_on_either_side(self):
        from sim.control_scene import (
            CONTROL_RADIUS,
            arrange,
            conspecific_boxes,
            make_scene,
        )
        from sim.hemifield import DEFAULT_OVERLAP
        from sim.raw_sense import BODY_RADIUS, render_view
        from sim.yolo_teacher import hemisphere_hits

        self.assertEqual(CONTROL_RADIUS, BODY_RADIUS)
        world, learner = make_scene()
        cube = world.controls[0]
        peer = next(agent for agent in world.agents if agent.agent_id != learner.agent_id)
        self.assertNotEqual(tuple(cube.color), tuple(c / 255.0 for c in peer.color))

        arrange(world, learner, conspecific="left", control="right")
        both = conspecific_boxes(learner, world)
        self.assertEqual(len(both), 1)
        self.assertEqual(hemisphere_hits(both, DEFAULT_OVERLAP, center_only=True), (True, False))
        _feat, hit = render_view(learner, world)
        self.assertTrue(hit.dog)
        self.assertTrue(hit.control)
        self.assertFalse(hit.distractor)

        arrange(world, learner, conspecific=None, control="left")
        self.assertEqual(conspecific_boxes(learner, world), [])
        _feat, only = render_view(learner, world)
        self.assertFalse(only.dog)
        self.assertTrue(only.control)

        arrange(world, learner, conspecific="right", control="left")
        right = conspecific_boxes(learner, world)
        self.assertEqual(len(right), 1)
        self.assertEqual(hemisphere_hits(right, DEFAULT_OVERLAP, center_only=True), (False, True))

    def test_memory_probe_prefers_the_conspecific(self):
        from sim.control_scene import TRAIN_PULSES, run_memory_probe

        report = run_memory_probe()
        before_dog = float(report["before"]["conspecific"])
        before_cube = float(report["before"]["control"])
        after_dog = float(report["after"]["conspecific"])
        after_cube = float(report["after"]["control"])
        gap = abs(before_dog - before_cube)
        rise_dog = after_dog - before_dog
        rise_cube = after_cube - before_cube
        # Shared motion moves the floor-row baseline. The two drives still match
        # within 12%, and that gap stays far below the conspecific's learned rise.
        self.assertLess(gap, 0.12 * max(abs(before_dog), abs(before_cube), 1.0))
        self.assertGreater(rise_dog, 80.0)
        self.assertLess(abs(rise_cube), 40.0)
        self.assertGreater(rise_dog, 4.0 * abs(rise_cube))
        self.assertEqual(report["train_pulses"], TRAIN_PULSES)
        self.assertTrue(report["reinforcement_off"])
        self.assertEqual(report["teacher_weight"], 0.0)
        self.assertTrue(report["weights_frozen"])

        left = report["probe"]["conspecific_left"]
        right = report["probe"]["conspecific_right"]
        self.assertGreater(left["z"], 0.0)
        self.assertLess(right["z"], 0.0)
        self.assertAlmostEqual(left["mixed_z"], left["z"])
        self.assertAlmostEqual(right["mixed_z"], right["z"])
        self.assertEqual(left["boxes"], 1)
        self.assertEqual(right["boxes"], 1)
        self.assertEqual(report["motion"], "moving-matched")
        self.assertTrue(any("движение как у сородича" in line for line in report["log"]))
        names = [row["name"] for row in report["object_drives"]]
        self.assertEqual(names, ["conspecific", "control", "conspecific", "control"])
        self.assertTrue(any("до обучения" in line for line in report["log"]))
        self.assertTrue(any("подкрепление выключено" in line for line in report["log"]))

    def test_status_reports_the_switch_and_the_drives(self):
        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.control_scene import run_memory_probe
        from sim.mb_train import MbTrainer, default_npz
        from sim.raw_sense import probe_features

        report = run_memory_probe()
        trainer = MbTrainer(default_npz(), seed=1)
        loop = BrainLoop(SportDrive(None), trainer, "/tmp/control_probe_status.npz")
        loop.handle({"op": "teacher_weight", "weight": 0.0}, 1.0)
        quiet = loop.status()
        self.assertFalse(quiet["reinforcement_off"])
        self.assertEqual(quiet["object_drives"], [])
        self.assertEqual(quiet["teacher_weight"], 0.0)

        held = np.array(trainer.brain.kc_mbon_w, copy=True)
        loop.handle({"op": "reinforcement_off", "off": True}, 2.0)
        trainer.learn = True
        loop.learn = True
        trainer.forward(probe_features("dog"))
        trainer.teach(trainer.last_fwd, "pam", 2.0)
        trainer.settle(True, True)
        np.testing.assert_array_equal(trainer.brain.kc_mbon_w, held)

        trainer.object_drives = report["object_drives"]
        status = loop.status()
        self.assertTrue(status["reinforcement_off"])
        self.assertEqual(status["teacher_weight"], 0.0)
        self.assertEqual(status["object_drives"], report["object_drives"])
        self.assertIn("подкрепление выключено", status["log"][-1])

    def test_static_control_stays_still(self):
        from sim.control_scene import MOTION_STATIC, advance_scene, arrange, make_scene
        from sim.raw_sense import N_AZ, OFF_MOTION, render_view

        world, learner = make_scene()
        peer, cube = arrange(world, learner, conspecific="left", control="right", motion=MOTION_STATIC)
        start = ((peer.x, peer.y), (cube.x, cube.y))
        self.assertEqual((peer.vx, peer.vy, cube.vx, cube.vy), (0.0, 0.0, 0.0, 0.0))
        advance_scene(world, learner, 0.5)
        self.assertEqual(((peer.x, peer.y), (cube.x, cube.y)), start)
        feat, hit = render_view(learner, world)
        self.assertTrue(hit.dog and hit.control)
        self.assertTrue(np.all(feat[OFF_MOTION : OFF_MOTION + N_AZ] == 0.0))

    def test_moving_matched_shares_trajectory_and_speed(self):
        from sim.control_scene import (
            MATCHED_SPEED,
            MOTION_MATCHED,
            advance_scene,
            arrange,
            conspecific_boxes,
            make_scene,
        )
        from sim.raw_sense import N_AZ, OFF_MOTION, render_view
        from sim.recognize_train import WALK

        self.assertEqual(MATCHED_SPEED, WALK)
        world, learner = make_scene()

        def trace(kind: str):
            peer, cube = arrange(
                world,
                learner,
                conspecific="left" if kind == "conspecific" else None,
                control="left" if kind == "control" else None,
                motion=MOTION_MATCHED,
            )
            obj = peer if kind == "conspecific" else cube
            points = []
            motions = []
            for _ in range(8):
                points.append((obj.x, obj.y))
                feat, hit = render_view(learner, world)
                motions.append(np.array(feat[OFF_MOTION : OFF_MOTION + N_AZ], copy=True))
                if kind == "conspecific":
                    self.assertEqual(len(conspecific_boxes(learner, world)), 1)
                    self.assertTrue(hit.dog)
                    self.assertFalse(hit.control)
                else:
                    self.assertEqual(conspecific_boxes(learner, world), [])
                    self.assertFalse(hit.dog)
                    self.assertTrue(hit.control)
                advance_scene(world, learner, 0.1)
            return points, motions

        dog_pts, dog_mot = trace("conspecific")
        cube_pts, cube_mot = trace("control")
        np.testing.assert_allclose(cube_pts, dog_pts)
        for dog_m, cube_m in zip(dog_mot, cube_mot):
            np.testing.assert_allclose(cube_m, dog_m)
            self.assertGreater(float(np.max(dog_m)), 0.0)

        peer, cube = arrange(
            world, learner, conspecific="left", control="right", motion=MOTION_MATCHED
        )
        self.assertEqual((peer.vx, peer.vy), (cube.vx, cube.vy))
        self.assertAlmostEqual(float(np.hypot(peer.vx, peer.vy)), MATCHED_SPEED)
        origin_peer = (peer.x, peer.y)
        origin_cube = (cube.x, cube.y)
        for _ in range(6):
            advance_scene(world, learner, 0.1)
        dog_shift = (peer.x - origin_peer[0], peer.y - origin_peer[1])
        cube_shift = (cube.x - origin_cube[0], cube.y - origin_cube[1])
        np.testing.assert_allclose(cube_shift, dog_shift)
        self.assertGreater(abs(dog_shift[0]) + abs(dog_shift[1]), 0.0)
        self.assertEqual(len(conspecific_boxes(learner, world)), 1)


if __name__ == "__main__":
    unittest.main()
