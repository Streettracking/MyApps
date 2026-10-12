"""Camera-only recognition. Lidar stays for range and the map.

Run: python -m unittest sim.test_recog_camera
"""

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim.hemifield import hemifield
from sim.map_marks import feature_range
from sim.mb_train import MbTrainer, default_npz
from sim.npz_compat import open_npz
from sim.pilot import forward_clearance
from sim.raw_sense import (
    N_RAW,
    OFF_BODY,
    OFF_CLOSE,
    OFF_LIDAR,
    RECOG_CAMERA_LABEL,
    add_recog_camera_only_arg,
    drop_lidar,
    recog_label,
)


def _scene() -> np.ndarray:
    feat = np.zeros(N_RAW, dtype=np.float32)
    feat[OFF_BODY + 4 * 3] = 0.75
    feat[OFF_BODY + 4 * 3 + 1] = 0.25
    feat[OFF_BODY + 0] = 0.55
    feat[OFF_LIDAR + 3] = 0.5
    feat[OFF_CLOSE + 3] = 0.4
    return feat


class DropTests(unittest.TestCase):
    def test_lidar_slots_clear_and_the_caller_keeps_range(self):
        feat = _scene()
        original = feat.copy()
        cleared = drop_lidar(feat)
        np.testing.assert_array_equal(feat, original)
        self.assertEqual(float(cleared[OFF_LIDAR:].sum()), 0.0)
        self.assertGreater(float(cleared[OFF_BODY]), 0.0)
        self.assertAlmostEqual(float(cleared[OFF_BODY + 4 * 3]), 0.75)
        dist = feature_range(feat, 3, width=1)
        self.assertIsNotNone(dist)
        assert dist is not None
        self.assertGreater(dist, 1.0)
        self.assertIsNotNone(forward_clearance(feat))
        self.assertEqual(recog_label(True), RECOG_CAMERA_LABEL)
        self.assertEqual(recog_label(True), "узнавание: только камера")

    def test_flag_defaults_on(self):
        parser = argparse.ArgumentParser()
        add_recog_camera_only_arg(parser)
        self.assertTrue(parser.parse_args([]).recog_camera_only)
        self.assertTrue(parser.parse_args(["--recog-camera-only"]).recog_camera_only)
        self.assertFalse(parser.parse_args(["--no-recog-camera-only"]).recog_camera_only)


class HemisphereTests(unittest.TestCase):
    def test_both_eyes_ignore_lidar_and_the_file_keeps_its_arrays(self):
        feat = _scene()
        bare = drop_lidar(feat)
        left = hemifield(bare, "L", 0.4)
        right = hemifield(bare, "R", 0.4)
        self.assertEqual(float(left[OFF_LIDAR:].sum()), 0.0)
        self.assertEqual(float(right[OFF_LIDAR:].sum()), 0.0)
        self.assertGreater(float(np.abs(left[:OFF_LIDAR]).sum()), 0.0)
        self.assertGreater(float(np.abs(right[:OFF_LIDAR]).sum()), 0.0)

        closed = MbTrainer(default_npz(), seed=2, camera_only=True)
        twin = MbTrainer(default_npz(), seed=2, camera_only=True)
        closed.forward(feat)
        twin.forward(bare)
        self.assertAlmostEqual(closed.r_l, twin.r_l)
        self.assertAlmostEqual(closed.r_r, twin.r_r)
        self.assertAlmostEqual(closed.score_feature(feat), closed.score_feature(bare))
        np.testing.assert_array_equal(feat, _scene())

        opened = MbTrainer(default_npz(), seed=2, camera_only=False)
        self.assertFalse(np.allclose(opened.proj.project(feat), opened.proj.project(bare)))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mb_train_state.npz"
            closed.save(path)
            with open_npz(path) as saved:
                keys = set(saved.files)
            self.assertIn("kc_mbon_w", keys)
            self.assertIn("kc_mbon_w_r", keys)
            self.assertNotIn("recog_camera_only", keys)
            loaded = MbTrainer(default_npz(), seed=2, camera_only=True)
            loaded.load(path)
            self.assertTrue(loaded.camera_only)

    def test_onboard_status_reports_the_flag_and_range_still_reads_lidar(self):
        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive

        brain = MbTrainer(default_npz(), seed=1, camera_only=True)
        loop = BrainLoop(SportDrive(None), brain, "/tmp/mb_camera_only.npz")
        self.assertTrue(loop.status()["recog_camera_only"])
        feat = _scene()
        self.assertIsNotNone(forward_clearance(feat))
        loop.mb.forward(feat)
        self.assertIsNotNone(forward_clearance(feat))
        self.assertGreater(float(feat[OFF_LIDAR + 3]), 0.0)


class StepTests(unittest.TestCase):
    def test_brain_step_is_unchanged_when_lidar_changes(self):
        def step(lidar: float):
            brain = MbTrainer(default_npz(), seed=3, camera_only=True)
            brain.learn = True
            feat = _scene().copy()
            feat[OFF_LIDAR:] = 0.0
            feat[OFF_LIDAR + 2] = lidar
            feat[OFF_CLOSE + 2] = lidar * 0.5
            brain.forward(feat)
            brain.observe(feat, brain.last_readout)
            brain.teach_sides("ppl1", "pam", 1.0)
            return brain, feat

        quiet, near = step(0.05)
        loud, far = step(0.95)
        np.testing.assert_array_equal(quiet.brain.kc_mbon_w, loud.brain.kc_mbon_w)
        np.testing.assert_array_equal(quiet.brain_r.kc_mbon_w, loud.brain_r.kc_mbon_w)
        self.assertEqual(quiet.r_l, loud.r_l)
        self.assertEqual(quiet.r_r, loud.r_r)
        self.assertEqual(quiet.last_readout, loud.last_readout)
        self.assertEqual(quiet.teacher_ppl1_l, loud.teacher_ppl1_l)
        self.assertEqual(quiet.teacher_pam_r, loud.teacher_pam_r)
        self.assertGreater(float(far[OFF_LIDAR + 2]), float(near[OFF_LIDAR + 2]))
        self.assertAlmostEqual(float(far[OFF_CLOSE + 2]), 0.95 * 0.5)
        self.assertIsNotNone(forward_clearance(_scene()))


class PanelTests(unittest.TestCase):
    def test_panel_draws_the_camera_only_line(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        from sim.train_monitor import MonitorView, TrainMonitor

        monitor = TrainMonitor("camera only")
        camera = np.zeros((96, 160, 3), dtype=np.uint8)
        camera[:] = (18, 22, 28)
        view = MonitorView(
            camera=camera,
            learner="mb",
            onboard=True,
            overlap=0.4,
            eye_l_recognized=True,
            eye_r_recognized=True,
            eyes_line="R_L +4   R_R +4   ОБА ВИДЯТ → ИДУ",
            recog_line="узнавание: только камера",
        )
        monitor.draw(view)
        shot = Path("/opt/cursor/artifacts/recog_camera.png")
        shot.parent.mkdir(parents=True, exist_ok=True)
        monitor.save_screenshot(str(shot))
        import pygame

        pygame.quit()

    def test_lidar_card_keeps_the_picture_and_says_it_does_not_teach(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame

        pygame.init()
        from sim.train_monitor import INK_DIM, LIDAR_LEARN_NOTE, MonitorView, TrainMonitor

        self.assertEqual(LIDAR_LEARN_NOTE, "в обучении не участвует")
        monitor = TrainMonitor("lidar note")
        camera = np.zeros((96, 160, 3), dtype=np.uint8)
        camera[:] = (18, 22, 28)
        lidar = np.zeros((80, 80, 3), dtype=np.uint8)
        lidar[20:60, 20:60] = (30, 80, 140)
        view = MonitorView(
            camera=camera,
            lidar=lidar,
            learner="mb",
            fly_line="R_L +1   R_R +1   Δ +0   z +0.00  билатерально",
            range_line="вектор +0°  дальн 2.0 м  вперёд 2.4 м",
        )
        monitor.draw(view)
        rect = monitor.lidar_learn_rect
        self.assertGreater(rect.w, 8)
        self.assertGreater(rect.h, 4)
        self.assertTrue(monitor.lid_rect.colliderect(rect))
        self.assertIsNotNone(monitor.lid_inner)
        frame = pygame.surfarray.array3d(monitor.screen)
        patch = frame[rect.x : rect.x + rect.w, rect.y : rect.y + rect.h]
        ink = np.array([int(channel) for channel in INK_DIM], dtype=np.int16)
        distance = np.abs(patch.astype(np.int16) - ink).sum(axis=2)
        self.assertGreater(int(np.sum(distance < 40)), 8)
        shot = Path("/opt/cursor/artifacts/lidar_note.png")
        shot.parent.mkdir(parents=True, exist_ok=True)
        monitor.save_screenshot(str(shot))
        pygame.quit()


if __name__ == "__main__":
    unittest.main()
