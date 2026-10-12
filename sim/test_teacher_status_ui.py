"""Teacher status line and «дистанция стопа» in the brain panel. Screens go to testlogs/."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

SHOTS = Path(__file__).resolve().parents[1] / "testlogs"


class TeacherStatusUi(unittest.TestCase):
    def setUp(self):
        fd, self._cfg = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.environ["RECOGNIZER_UI_CONFIG"] = self._cfg

    def tearDown(self):
        os.environ.pop("RECOGNIZER_UI_CONFIG", None)
        from sim.ui_theme import apply_theme

        apply_theme("light")
        try:
            os.remove(self._cfg)
        except OSError:
            pass

    def test_status_and_stop_slider(self):
        import pygame

        from sim.train_monitor import MonitorView, TrainMonitor

        pygame.init()
        try:
            mon = TrainMonitor("lead")
            mon._on_resize(1920, 1080)
            view = MonitorView(
                learner="mb",
                camera=np.full((90, 160, 3), 90, dtype=np.uint8),
                lidar=np.zeros((64, 64, 3), dtype=np.uint8),
                autonomy_on=True,
                teacher_weight=1.0,
                lead_plaque="ведёт: учитель",
                teacher_status="иду к .128",
                teacher_status_note="PAM/PPL1 не идут",
            )
            mon.draw(view)
            tags = [tag for _r, tag in getattr(mon, "_brain_rows", [])]
            SHOTS.mkdir(parents=True, exist_ok=True)
            pygame.image.save(pygame.transform.smoothscale(mon.screen, (1024, 576)), str(SHOTS / "teacher_lead_light_1024.png"))
            mon._set_theme("dark")
            view.teacher_status = "кадр старый"
            mon.draw(view)
            pygame.image.save(pygame.transform.smoothscale(mon.screen, (1024, 576)), str(SHOTS / "teacher_lead_dark_1024.png"))
            self.assertGreater(mon.teacher_rect.w, 8)
            if "teacher-status" in tags:
                self.assertIn("stop-slider", tags)
                self.assertGreater(mon.stop_slider_rect.w, 8)
        finally:
            pygame.quit()

    def test_stop_h_saved_and_clamped(self):
        from sim.teacher_lead import STOP_BOX_H_MAX, clamp_stop_h

        self.assertEqual(clamp_stop_h(5.0), STOP_BOX_H_MAX)
        self.assertEqual(clamp_stop_h("x"), 0.45)
        Path(self._cfg).write_text(json.dumps({"teacher_stop_h": 0.6}), encoding="utf-8")
        from sim.ui_settings import load_settings

        data = load_settings()
        self.assertAlmostEqual(float(data["teacher_stop_h"]), 0.6)


if __name__ == "__main__":
    unittest.main()
