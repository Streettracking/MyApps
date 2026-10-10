"""Second-eye confirmation, binocular overlap, and the trainer overlay.

Run: python -m unittest sim.test_eye_overlap
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim.hemifield import DEFAULT_OVERLAP, clamp_overlap, hemifield, overlap_bands
from sim.pilot import (
    EYE_CONFIRM_S,
    EYE_LOSE_N,
    SEARCH_TURN,
    EyeConfirm,
    Pilot,
    eyes_state_label,
    format_eyes_line,
    phase_label,
)
from sim.raw_sense import N_RAW, OFF_BODY, OFF_LIDAR
from sim.train_monitor import eye_tone


def _blob(sector: int, value: float = 1.0) -> np.ndarray:
    feat = np.zeros(N_RAW, dtype=np.float32)
    feat[OFF_LIDAR + int(sector)] = value
    feat[OFF_BODY + int(sector) * 3] = value
    return feat


def _energy(feat: np.ndarray) -> float:
    return float(np.sum(np.abs(feat)))


def _auto(pilot: Pilot, now: float, *, left: bool, right: bool, dist: float = 3.0, forward: float = 3.0, r_l: float = 5.0, r_r: float = 5.0):
    return pilot.command(
        now,
        (0.0, 0.0),
        focused=True,
        frames_ok=True,
        link_ok=True,
        recognized=left or right,
        sector=4,
        dist_m=dist,
        forward_m=forward,
        r_l=r_l if left else 0.0,
        r_r=r_r if right else 0.0,
        recognized_l=left,
        recognized_r=right,
    )


class OverlapTests(unittest.TestCase):
    def test_bands_and_clamp(self):
        self.assertEqual(DEFAULT_OVERLAP, 0.4)
        self.assertEqual(overlap_bands(0.4), (0.3, 0.7))
        self.assertEqual(overlap_bands(0.2), (0.4, 0.6))
        self.assertEqual(overlap_bands(0.0), (0.5, 0.5))
        self.assertEqual(overlap_bands(0.5), (0.25, 0.75))
        self.assertEqual(clamp_overlap(2.0), 0.5)
        self.assertEqual(clamp_overlap(-1.0), 0.0)

    def test_zero_overlap_matches_the_hard_split(self):
        rng = np.random.default_rng(0)
        feat = rng.random(N_RAW).astype(np.float32)
        np.testing.assert_array_equal(hemifield(feat, "L", 0.0), hemifield(feat, "L"))
        np.testing.assert_array_equal(hemifield(feat, "R", 0.0), hemifield(feat, "R"))
        left = hemifield(_blob(4), "R", 0.0)
        right = hemifield(_blob(3), "L", 0.0)
        self.assertEqual(_energy(left), 0.0)
        self.assertEqual(_energy(right), 0.0)
        self.assertGreater(_energy(hemifield(_blob(4), "L", 0.0)), 0.0)
        self.assertGreater(_energy(hemifield(_blob(3), "R", 0.0)), 0.0)

    def test_center_reaches_both_eyes_and_edges_stay_apart(self):
        center = _blob(3) + _blob(4)
        self.assertGreater(_energy(hemifield(center, "L", 0.2)), 0.0)
        self.assertGreater(_energy(hemifield(center, "R", 0.2)), 0.0)
        # Sector 4 sits just left of the midline and now spills into the right eye.
        self.assertGreater(_energy(hemifield(_blob(4), "R", 0.2)), 0.0)
        self.assertEqual(_energy(hemifield(_blob(0), "L", 0.2)), 0.0)
        self.assertGreater(_energy(hemifield(_blob(0), "R", 0.2)), 0.0)
        self.assertEqual(_energy(hemifield(_blob(7), "R", 0.2)), 0.0)
        self.assertGreater(_energy(hemifield(_blob(7), "L", 0.2)), 0.0)

    def test_resample_keeps_the_72d_shape(self):
        feat = _blob(4) + _blob(5)
        for side in ("L", "R"):
            out = hemifield(feat, side, 0.2)
            self.assertEqual(out.shape, (N_RAW,))
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.all(out[OFF_LIDAR : OFF_LIDAR + 4] == 0.0))
            self.assertTrue(np.all(out[OFF_BODY : OFF_BODY + 12] == 0.0))


class EyeConfirmTests(unittest.TestCase):
    def test_both_eyes_walk_one_eye_turns_in_place(self):
        pilot = Pilot()
        pilot.start_auto(0.0)
        both = _auto(pilot, 0.0, left=True, right=True)
        self.assertEqual(both.phase, "approach")
        self.assertAlmostEqual(both.x, 0.4)
        self.assertEqual(both.z, 0.0)
        self.assertEqual(phase_label(both.phase), "подтверждено — иду")

        left_only = Pilot()
        left_only.start_auto(0.0)
        turn = _auto(left_only, 0.0, left=True, right=False)
        self.assertEqual(turn.phase, "align_l")
        self.assertEqual(turn.x, 0.0)
        self.assertAlmostEqual(turn.z, SEARCH_TURN)
        self.assertEqual(phase_label(turn.phase), "доворот (Л)")

        right_only = Pilot()
        right_only.start_auto(0.0)
        turn = _auto(right_only, 0.0, left=False, right=True)
        self.assertEqual(turn.phase, "align_r")
        self.assertEqual(turn.x, 0.0)
        self.assertAlmostEqual(turn.z, -SEARCH_TURN)
        self.assertEqual(phase_label(turn.phase), "доворот (П)")

    def test_second_eye_timeout_returns_to_search(self):
        pilot = Pilot()
        pilot.start_auto(0.0)
        started = _auto(pilot, 0.0, left=True, right=False)
        self.assertEqual(started.phase, "align_l")
        held = _auto(pilot, 2.9, left=True, right=False)
        self.assertEqual(held.phase, "align_l")
        gave_up = _auto(pilot, EYE_CONFIRM_S, left=True, right=False)
        self.assertEqual(gave_up.phase, "search")
        self.assertEqual(gave_up.x, 0.0)
        still = _auto(pilot, EYE_CONFIRM_S + 0.4, left=True, right=False)
        self.assertEqual(still.phase, "search")
        _auto(pilot, EYE_CONFIRM_S + 0.6, left=False, right=False)
        again = _auto(pilot, EYE_CONFIRM_S + 0.8, left=True, right=False)
        self.assertEqual(again.phase, "align_l")

    def test_losing_one_eye_stops_the_walk(self):
        pilot = Pilot()
        pilot.start_auto(0.0)
        _auto(pilot, 0.0, left=True, right=True)
        for step in range(1, EYE_LOSE_N):
            cmd = _auto(pilot, 0.05 * step, left=True, right=False)
            self.assertEqual(cmd.phase, "approach", step)
            self.assertGreater(cmd.x, 0.0)
        dropped = _auto(pilot, 0.05 * EYE_LOSE_N, left=True, right=False)
        self.assertEqual(dropped.phase, "align_l")
        self.assertEqual(dropped.x, 0.0)

    def test_close_target_holds_and_sector_mode_ignores_eyes(self):
        pilot = Pilot()
        pilot.start_auto(0.0)
        held = _auto(pilot, 0.0, left=True, right=True, dist=0.8, forward=0.8)
        self.assertEqual(held.phase, "hold")
        self.assertEqual(held.x, 0.0)

        sectors = Pilot()
        sectors.set_steer("sectors")
        sectors.start_auto(0.0)
        last = None
        for step in range(8):
            last = sectors.command(
                0.1 * step,
                (0.0, 0.0),
                focused=True,
                frames_ok=True,
                link_ok=True,
                recognized=True,
                sector=4,
                dist_m=3.0,
                forward_m=3.0,
                recognized_l=False,
                recognized_r=False,
            )
        assert last is not None
        self.assertEqual(last.phase, "approach")
        self.assertGreater(last.x, 0.0)

    def test_eye_class_names_the_false_alarm(self):
        eyes = EyeConfirm(timeout_s=3.0, lose_n=5)
        self.assertEqual(eyes.update(0.0, True, False), "align_l")
        self.assertEqual(eyes.update(3.0, True, False), "search")
        self.assertEqual(eyes.update(3.1, True, False), "search")


class StateAndStatusTests(unittest.TestCase):
    def test_saved_state_keeps_the_old_arrays(self):
        from sim.mb_train import MbTrainer, default_npz
        from sim.npz_compat import open_npz

        brain = MbTrainer(default_npz(), seed=1, overlap=0.2)
        self.assertEqual(brain.overlap, 0.2)
        feat = np.zeros(N_RAW, dtype=np.float32)
        feat[OFF_LIDAR + 4] = 1.0
        brain.forward(feat)
        brain.observe(feat, brain.last_readout)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mb_train_state.npz"
            brain.save(path)
            with open_npz(path) as saved:
                keys = set(saved.files)
            self.assertIn("kc_mbon_w", keys)
            self.assertIn("kc_mbon_w_r", keys)
            self.assertIn("cal_readout", keys)
            self.assertIn("stat_tot", keys)
            self.assertNotIn("overlap", keys)
            self.assertNotIn("cal_readout_l", keys)
            other = MbTrainer(default_npz(), seed=1, overlap=0.0)
            other.load(path)
            self.assertEqual(other.overlap, 0.0)
            np.testing.assert_array_equal(other.brain_r.kc_mbon_w, brain.brain_r.kc_mbon_w)

    def test_status_reports_eyes_and_overlap(self):
        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import MbTrainer, default_npz

        brain = MbTrainer(default_npz(), seed=1, overlap=0.2)
        loop = BrainLoop(SportDrive(None), brain, "/tmp/mb_eye_status.npz")
        payload = loop.status()
        self.assertEqual(payload["overlap"], 0.2)
        self.assertIn("recognized_L", payload)
        self.assertIn("recognized_R", payload)
        self.assertIn("confidence_L", payload)
        self.assertIn("confidence_R", payload)
        self.assertEqual(payload["phase_ru"], phase_label(loop.pilot.phase, loop.pilot.steer))
        now = 10.0
        loop.last_hb = now
        loop.frames_ok = True
        loop.last_frame_t = now
        loop.pilot.start_auto(now)
        loop.recognized_l = True
        loop.recognized_r = True
        loop.dist_m = 3.0
        loop.forward_m = 3.0
        loop.mb.r_l = 5.0
        loop.mb.r_r = 5.0
        cmd = loop.tick(now)
        self.assertEqual(cmd.phase, "approach")
        self.assertAlmostEqual(cmd.x, 0.4)
        self.assertEqual(loop.status()["phase_ru"], "подтверждено — иду")
        self.assertTrue(loop.status()["recognized_L"])
        self.assertTrue(loop.status()["recognized_R"])


class EyesLineTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(eyes_state_label(True, True), "ОБА ВИДЯТ → ИДУ")
        self.assertEqual(eyes_state_label(True, False), "ОДИН ГЛАЗ → ДОВОРОТ")
        self.assertEqual(eyes_state_label(False, False), "НЕТ → ПОИСК")
        both = format_eyes_line(8, 1, True, True)
        one = format_eyes_line(None, None, False, True)
        wide = format_eyes_line(-168, 84, False, False)
        self.assertEqual(len(both), len(one))
        self.assertEqual(len(both), len(wide))
        self.assertEqual(both.index("R_L"), one.index("R_L"))
        self.assertEqual(both.index("R_R"), one.index("R_R"))
        self.assertEqual(both.index("R_R"), wide.index("R_R"))
        self.assertEqual(both.index("ОБА"), one.index("ОДИ"))
        self.assertEqual(both.index("ОБА"), wide.index("НЕТ"))
        self.assertIn("+8", both)
        self.assertIn("+1", both)
        self.assertIn("-", one)
        self.assertIn("ОБА ВИДЯТ → ИДУ", both)
        self.assertIn("ОДИН ГЛАЗ → ДОВОРОТ", one)


class OnboardEyeDriveTests(unittest.TestCase):
    def _loop(self):
        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import MbTrainer, default_npz

        brain = MbTrainer(default_npz(), seed=1, overlap=0.4)
        return BrainLoop(SportDrive(None), brain, "/tmp/mb_eye_drive.npz")

    def _arm(self, loop, now: float) -> None:
        loop.handle({"op": "autonomy_on"}, now)
        loop.frames_ok = True
        loop.last_frame_t = now
        loop.last_hb = now
        loop.dist_m = 3.0
        loop.forward_m = 3.0

    def test_both_eyes_walk_and_yaw_follows_the_readouts(self):
        loop = self._loop()
        self._arm(loop, 0.0)
        loop.recognized = True
        loop.recognized_l = True
        loop.recognized_r = True
        loop.mb.r_l = 8.0
        loop.mb.r_r = 1.0
        cmd = loop.tick(0.0)
        self.assertEqual(cmd.phase, "approach")
        self.assertAlmostEqual(cmd.x, 0.4)
        self.assertGreater(cmd.z, 0.2)
        self.assertAlmostEqual(loop.drive.moves[-1][0], 0.4)
        self.assertGreater(loop.drive.moves[-1][2], 0.0)
        self.assertEqual(loop.status()["eyes_ru"], "ОБА ВИДЯТ → ИДУ")

        other = self._loop()
        self._arm(other, 0.0)
        other.recognized = True
        other.recognized_l = True
        other.recognized_r = True
        other.mb.r_l = 1.0
        other.mb.r_r = 8.0
        cmd = other.tick(0.0)
        self.assertAlmostEqual(cmd.x, 0.4)
        self.assertLess(cmd.z, -0.2)
        self.assertEqual(other.status()["eyes_ru"], "ОБА ВИДЯТ → ИДУ")

    def test_one_eye_turns_and_a_loss_searches_that_side(self):
        loop = self._loop()
        self._arm(loop, 0.0)
        loop.recognized_l = False
        loop.recognized_r = True
        loop.mb.r_r = 6.0
        cmd = loop.tick(0.0)
        self.assertEqual(cmd.phase, "align_r")
        self.assertEqual(cmd.x, 0.0)
        self.assertAlmostEqual(cmd.z, -SEARCH_TURN)
        self.assertEqual(loop.drive.moves[-1][0], 0.0)
        self.assertEqual(loop.status()["eyes_ru"], "ОДИН ГЛАЗ → ДОВОРОТ")
        loop.recognized_r = False
        lost = loop.tick(0.2)
        self.assertEqual(lost.phase, "search")
        self.assertEqual(lost.x, 0.0)
        self.assertLess(lost.z, 0.0)
        self.assertEqual(loop.status()["last_seen_side"], "R")
        self.assertEqual(loop.status()["eyes_ru"], "НЕТ → ПОИСК")

        left = self._loop()
        self._arm(left, 1.0)
        left.recognized_l = True
        left.recognized_r = False
        left.mb.r_l = 6.0
        turn = left.tick(1.0)
        self.assertEqual(turn.phase, "align_l")
        self.assertEqual(turn.x, 0.0)
        self.assertAlmostEqual(turn.z, SEARCH_TURN)
        left.recognized_l = False
        found = left.tick(1.2)
        self.assertEqual(found.phase, "search")
        self.assertGreater(found.z, 0.0)
        self.assertEqual(left.status()["last_seen_side"], "L")


class OverlayTests(unittest.TestCase):
    def test_tones(self):
        self.assertEqual(eye_tone(False, False, 0.0), "grey")
        self.assertEqual(eye_tone(True, False, 0.0), "grey")
        self.assertEqual(eye_tone(True, False, 40.0), "yellow")
        self.assertEqual(eye_tone(True, True, 90.0), "green")

    def test_draw_does_not_touch_the_camera(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        from sim.train_monitor import MonitorView, TrainMonitor

        monitor = TrainMonitor("overlap test")
        camera = np.zeros((96, 160, 3), dtype=np.uint8)
        camera[:, :] = (18, 22, 28)
        camera[:, 70:96] = (40, 70, 90)
        original = camera.copy()
        view = MonitorView(
            camera=camera,
            learner="mb",
            steer="bilateral",
            overlap=0.2,
            eye_l_ready=True,
            eye_l_recognized=True,
            eye_l_confidence=88.0,
            eye_r_ready=True,
            eye_r_recognized=False,
            eye_r_confidence=35.0,
            phase_ru="доворот (П)",
            pilot_who="мозг",
            pilot_mode="АВТОНОМИЯ",
            autonomy_on=True,
        )
        monitor.draw(view)
        np.testing.assert_array_equal(camera, original)
        shot = Path("/opt/cursor/artifacts/trainer_overlap.png")
        shot.parent.mkdir(parents=True, exist_ok=True)
        monitor.save_screenshot(str(shot))
        import pygame

        pygame.quit()

    def test_plaques_mark_each_eye_and_do_not_paint_the_camera(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        from sim.train_monitor import MonitorView, TrainMonitor

        monitor = TrainMonitor("eyes")
        camera = np.zeros((96, 160, 3), dtype=np.uint8)
        camera[:] = (18, 22, 28)
        original = camera.copy()
        view = MonitorView(
            camera=camera,
            learner="mb",
            onboard=True,
            overlap=0.4,
            eye_l_recognized=True,
            eye_r_recognized=False,
            eyes_line="R_L +8   R_R +1   ОДИН ГЛАЗ → ДОВОРОТ",
        )
        monitor.draw(view)
        np.testing.assert_array_equal(camera, original)
        import pygame

        frame = pygame.surfarray.array3d(monitor.screen)
        band = frame[:1100, 70:150, :]
        green = (
            (np.abs(band[:, :, 0].astype(int) - 141) < 40)
            & (np.abs(band[:, :, 1].astype(int) - 181) < 40)
            & (np.abs(band[:, :, 2].astype(int) - 150) < 40)
        )
        gray = (
            (np.abs(band[:, :, 0].astype(int) - 176) < 20)
            & (np.abs(band[:, :, 1].astype(int) - 174) < 20)
            & (np.abs(band[:, :, 2].astype(int) - 170) < 20)
        )
        self.assertGreater(int(green.sum()), 20)
        self.assertGreater(int(gray.sum()), 20)
        self.assertGreater(float(np.where(green)[0].mean()), float(np.where(gray)[0].mean()))
        shot = Path("/opt/cursor/artifacts/eyes_onboard.png")
        shot.parent.mkdir(parents=True, exist_ok=True)
        monitor.save_screenshot(str(shot))
        pygame.quit()


class SearchSideTests(unittest.TestCase):
    def test_lost_target_searches_toward_the_last_side_without_swinging(self):
        pilot = Pilot()
        pilot.start_auto(0.0)
        _auto(pilot, 0.0, left=True, right=True, r_l=4.0, r_r=1.0)
        self.assertEqual(pilot.last_seen_side, "L")
        cmd = None
        for step in range(1, EYE_LOSE_N + 1):
            cmd = _auto(pilot, 0.05 * step, left=False, right=False, r_l=0.0, r_r=9.0)
        assert cmd is not None
        self.assertEqual(cmd.phase, "search")
        self.assertGreater(cmd.z, 0.0)
        self.assertEqual(phase_label(cmd.phase, search_sign=pilot.search_sign), "поиск ←")
        later = _auto(pilot, 0.05 * EYE_LOSE_N + 0.2, left=False, right=False, r_l=0.0, r_r=9.0)
        self.assertEqual(later.phase, "search")
        self.assertGreater(later.z, 0.0)
        self.assertEqual(pilot.search_sign, 1.0)
        paused = _auto(pilot, 0.05 * EYE_LOSE_N + 1.2, left=False, right=False)
        self.assertEqual(paused.z, 0.0)
        self.assertEqual(pilot.search_sign, 1.0)
        again = _auto(pilot, 0.05 * EYE_LOSE_N + 1.6, left=False, right=False)
        self.assertGreater(again.z, 0.0)
        self.assertEqual(pilot.search_sign, 1.0)

    def test_right_eye_timeout_searches_right(self):
        pilot = Pilot()
        pilot.start_auto(0.0)
        _auto(pilot, 0.0, left=False, right=True, r_l=0.0, r_r=3.0)
        gave = _auto(pilot, EYE_CONFIRM_S, left=False, right=True, r_l=0.0, r_r=3.0)
        self.assertEqual(gave.phase, "search")
        self.assertEqual(gave.x, 0.0)
        self.assertAlmostEqual(gave.z, -SEARCH_TURN)
        self.assertEqual(pilot.last_seen_side, "R")
        self.assertEqual(phase_label("search", search_sign=pilot.search_sign), "поиск →")

    def test_old_sighting_falls_back_to_the_left(self):
        pilot = Pilot()
        pilot.start_auto(0.0)
        _auto(pilot, 0.0, left=True, right=True, r_l=1.0, r_r=4.0)
        self.assertEqual(pilot.last_seen_side, "R")
        pilot.stop_auto()
        pilot.start_auto(12.0)
        cmd = _auto(pilot, 12.0, left=False, right=False)
        self.assertEqual(cmd.phase, "search")
        self.assertGreater(cmd.z, 0.0)
        self.assertEqual(pilot.search_sign, 1.0)

    def test_zero_difference_keeps_the_stored_side(self):
        pilot = Pilot()
        pilot.start_auto(0.0)
        _auto(pilot, 0.0, left=True, right=True, r_l=4.0, r_r=1.0)
        _auto(pilot, 0.2, left=True, right=True, r_l=2.0, r_r=2.0)
        self.assertEqual(pilot.last_seen_side, "L")
        self.assertAlmostEqual(pilot.last_seen_at, 0.2)


class DeployArgTests(unittest.TestCase):
    def test_start_forwards_overlap_and_status_still_matches_main(self):
        from robot.deploy_flybrain import _remote_args, _start_cmd

        cmd = _start_cmd(extra=["--overlap", "0.4", "--steer", "sectors"])
        self.assertIn("python3 /root/flybrain/main.py --overlap 0.4 --steer sectors", cmd)
        self.assertIn("grep -qx /root/flybrain/main.py", cmd)
        argv = "python3\n/root/flybrain/main.py\n--overlap\n0.4".split("\n")
        self.assertIn("/root/flybrain/main.py", argv)
        wrapper = "bash -lc 'nohup setsid python3 /root/flybrain/main.py --overlap 0.4'"
        self.assertNotIn("/root/flybrain/main.py", wrapper.split("\n"))
        with self.assertRaises(SystemExit):
            _remote_args(["0.4; rm"])


if __name__ == "__main__":
    unittest.main()
