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


if __name__ == "__main__":
    unittest.main()
