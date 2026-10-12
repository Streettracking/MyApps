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
        loop._dog_in_frame = True
        loop._yolo_t = now
        cmd = loop.tick(now)
        self.assertEqual(cmd.phase, "approach")
        self.assertAlmostEqual(cmd.x, 0.4)
        self.assertEqual(loop.status()["phase_ru"], "подтверждено — иду")
        self.assertTrue(loop.status()["recognized_L"])
        self.assertTrue(loop.status()["recognized_R"])

    def test_status_adds_the_ui_brain_fields(self):
        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.hemifield import live_turn
        from sim.mb_train import MbTrainer, default_npz

        brain = MbTrainer(default_npz(), seed=1, overlap=0.4)
        self.assertEqual(brain.stimuli.status(), [])
        loop = BrainLoop(SportDrive(None), brain, "/tmp/mb_ui_status.npz")
        quiet = loop.status()
        self.assertEqual(
            quiet["stimuli"],
            [{"name": "conspecific", "active": False, "value": 0.0}],
        )
        self.assertIsInstance(quiet["stimuli"][0]["active"], bool)
        self.assertIsInstance(quiet["stimuli"][0]["value"], float)
        for key in (
            "kc_active_frac",
            "kc_active_frac_gd",
            "kc_active_frac_abp",
            "drive_l",
            "drive_r",
            "turn",
            "pam_event",
            "ppl1_event",
            "pam_count",
            "ppl1_count",
        ):
            self.assertIn(key, quiet)
        self.assertIn("n_pam", quiet)
        self.assertIn("n_ppl1", quiet)
        self.assertIn("r_l", quiet)
        self.assertIn("z_fly", quiet)
        self.assertIn("kc_on", quiet)
        self.assertIsInstance(quiet["pam_event"], bool)
        self.assertIsInstance(quiet["ppl1_event"], bool)
        self.assertIsInstance(quiet["pam_count"], int)
        self.assertIsInstance(quiet["ppl1_count"], int)
        self.assertFalse(quiet["pam_event"])
        self.assertFalse(quiet["ppl1_event"])
        self.assertEqual(quiet["pam_count"], 0)
        self.assertEqual(quiet["ppl1_count"], 0)
        self.assertGreaterEqual(quiet["kc_active_frac"], 0.0)
        self.assertLessEqual(quiet["kc_active_frac"], 1.0)
        self.assertAlmostEqual(
            quiet["turn"], live_turn(quiet["drive_l"], quiet["drive_r"], loop.mb.base_l, loop.mb.base_r)
        )

        loop.learn = True
        camera = np.zeros((16, 32, 3), dtype=np.uint8)
        camera[:, :8, 1] = 220
        loop.ingest(camera, None, 1.0)
        seen = loop.status()
        gd_on = loop.mb.brain.vision.last_gd_on + loop.mb.brain_r.vision.last_gd_on
        gd_n = loop.mb.brain.vision.n_gd + loop.mb.brain_r.vision.n_gd
        ab_on = loop.mb.brain.vision.last_abp_on + loop.mb.brain_r.vision.last_abp_on
        ab_n = loop.mb.brain.vision.n_abp + loop.mb.brain_r.vision.n_abp
        self.assertAlmostEqual(seen["kc_active_frac_gd"], gd_on / float(gd_n))
        self.assertAlmostEqual(seen["kc_active_frac_abp"], ab_on / float(ab_n))
        self.assertAlmostEqual(seen["kc_active_frac"], (gd_on + ab_on) / float(gd_n + ab_n))
        self.assertGreaterEqual(seen["kc_active_frac_gd"], 0.0)
        self.assertLessEqual(seen["kc_active_frac_gd"], 1.0)
        self.assertGreaterEqual(seen["kc_active_frac_abp"], 0.0)
        self.assertLessEqual(seen["kc_active_frac_abp"], 1.0)
        self.assertAlmostEqual(seen["drive_l"], loop.mb.brain.appetitive_drive(loop.mb.last_fwd.action_scores))
        self.assertAlmostEqual(seen["drive_r"], loop.mb.brain_r.appetitive_drive(loop.mb.last_fwd_r.action_scores))
        self.assertAlmostEqual(
            seen["turn"], live_turn(seen["drive_l"], seen["drive_r"], loop.mb.base_l, loop.mb.base_r)
        )
        self.assertFalse(seen["pam_event"])
        self.assertFalse(seen["ppl1_event"])
        self.assertEqual(seen["n_pam"], seen["pam_count"])
        self.assertEqual(seen["n_ppl1"], seen["ppl1_count"])

        loop.handle({"op": "teach_sides", "left": "pam", "right": "ppl1"}, 2.0)
        loop.ingest(camera, None, 2.0)
        taught = loop.status()
        self.assertTrue(taught["pam_event"])
        self.assertTrue(taught["ppl1_event"])
        self.assertEqual(taught["pam_count"], loop.mb.n_pam)
        self.assertEqual(taught["ppl1_count"], loop.mb.n_ppl1)
        self.assertEqual(taught["pam_count"], 1)
        self.assertEqual(taught["ppl1_count"], 1)
        self.assertEqual(taught["n_pam"], 1)
        self.assertEqual(taught["n_ppl1"], 1)

        loop.ingest(camera, None, 3.0)
        later = loop.status()
        self.assertFalse(later["pam_event"])
        self.assertFalse(later["ppl1_event"])
        self.assertEqual(later["pam_count"], 1)
        self.assertEqual(later["ppl1_count"], 1)


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

    def _see(self, loop, now: float) -> None:
        loop._dog_in_frame = True
        loop._yolo_t = float(now)

    def test_both_eyes_walk_and_yaw_follows_the_readouts(self):
        loop = self._loop()
        self._arm(loop, 0.0)
        self._see(loop, 0.0)
        loop.recognized = True
        loop.recognized_l = True
        loop.recognized_r = True
        loop.mb.r_l = 480.0
        loop.mb.r_r = 40.0
        loop.mb.base_l = 480.0
        loop.mb.base_r = 40.0
        cmd = loop.tick(0.0)
        self.assertEqual(cmd.phase, "approach")
        self.assertAlmostEqual(cmd.x, 0.4)
        self.assertGreater(cmd.z, 0.2)
        self.assertAlmostEqual(loop.drive.moves[-1][0], 0.4)
        self.assertGreater(loop.drive.moves[-1][2], 0.0)
        self.assertEqual(loop.status()["eyes_ru"], "ОБА ВИДЯТ → ИДУ")

        other = self._loop()
        self._arm(other, 0.0)
        self._see(other, 0.0)
        other.recognized = True
        other.recognized_l = True
        other.recognized_r = True
        other.mb.r_l = 40.0
        other.mb.r_r = 480.0
        other.mb.base_l = 40.0
        other.mb.base_r = 480.0
        cmd = other.tick(0.0)
        self.assertAlmostEqual(cmd.x, 0.4)
        self.assertLess(cmd.z, -0.2)
        self.assertEqual(other.status()["eyes_ru"], "ОБА ВИДЯТ → ИДУ")

    def test_one_eye_with_a_target_follows_the_drive(self):
        from sim.search_reflex import SEARCH_REFLEX_YAW

        loop = self._loop()
        self._arm(loop, 0.0)
        self._see(loop, 0.0)
        loop.recognized_l = False
        loop.recognized_r = True
        loop.mb.r_l = 0.0
        loop.mb.r_r = 480.0
        loop.mb.base_l = 0.0
        loop.mb.base_r = 480.0
        cmd = loop.tick(0.0)
        self.assertEqual(cmd.phase, "approach")
        self.assertAlmostEqual(cmd.x, 0.4)
        self.assertLess(cmd.z, -0.2)
        self.assertEqual(loop.status()["eyes_ru"], "ОДИН ГЛАЗ → ДОВОРОТ")
        loop._dog_in_frame = False
        loop.recognized_r = False
        lost = loop.tick(0.2)
        self.assertEqual(lost.phase, "search")
        self.assertEqual(lost.x, 0.0)
        self.assertAlmostEqual(lost.z, SEARCH_REFLEX_YAW)
        self.assertTrue(loop.searching)
        self.assertEqual(loop.status()["last_seen_side"], "R")
        self.assertEqual(loop.status()["eyes_ru"], "НЕТ → ПОИСК")

        left = self._loop()
        self._arm(left, 1.0)
        self._see(left, 1.0)
        left.recognized_l = True
        left.recognized_r = False
        left.mb.r_l = 480.0
        left.mb.r_r = 0.0
        left.mb.base_l = 480.0
        left.mb.base_r = 0.0
        turn = left.tick(1.0)
        self.assertEqual(turn.phase, "approach")
        self.assertGreater(turn.z, 0.2)
        left._dog_in_frame = False
        left.recognized_l = False
        found = left.tick(1.2)
        self.assertEqual(found.phase, "search")
        self.assertAlmostEqual(found.z, SEARCH_REFLEX_YAW)
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
        self.assertLess(float(np.where(green)[0].mean()), float(np.where(gray)[0].mean()))
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


class OnboardThemeTests(unittest.TestCase):
    def test_app_and_main_import_when_ui_theme_is_missing(self):
        import ast
        import importlib
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        sim_dir = root / "sim"
        onboard = root / "robot" / "flybrain_onboard"

        def imported(path: Path, package: str) -> set:
            """Modules imported while the file loads. Imports inside functions wait."""
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found = set()

            class Scan(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    return

                def visit_AsyncFunctionDef(self, node):
                    return

                def visit_ImportFrom(self, node):
                    if not node.module:
                        return
                    if package == "sim" and node.level == 1:
                        found.add(node.module.split(".")[0])
                    elif node.module == "sim" or node.module.startswith("sim."):
                        parts = node.module.split(".")
                        if len(parts) >= 2:
                            found.add(parts[1])

            Scan().visit(tree)
            return found

        def closure(start) -> set:
            pending = list(start)
            seen = set()
            while pending:
                name = pending.pop()
                if name in seen:
                    continue
                seen.add(name)
                path = sim_dir / ("%s.py" % name)
                if not path.is_file():
                    continue
                for child in imported(path, "sim"):
                    if child not in seen:
                        pending.append(child)
            return seen

        needed = set()
        for name in ("app", "main", "drive", "sensors"):
            needed |= imported(onboard / ("%s.py" % name), "onboard")
        needed |= closure(sorted(needed))
        from robot.deploy_flybrain import SIM_FILES

        listed = {Path(name).stem for name in SIM_FILES}
        self.assertNotIn("ui_theme", listed)
        missing = sorted((needed - {"ui_theme"}) - listed)
        self.assertEqual(missing, [])

        saved = {}
        drop = (
            "sim.learn_flash",
            "sim.ui_theme",
            "robot.flybrain_onboard.app",
            "robot.flybrain_onboard.main",
        )
        for name in drop:
            saved[name] = sys.modules.get(name)
            sys.modules.pop(name, None)
        sys.modules["sim.ui_theme"] = None
        try:
            app = importlib.import_module("robot.flybrain_onboard.app")
            main = importlib.import_module("robot.flybrain_onboard.main")
            self.assertTrue(callable(app.make_brain))
            self.assertTrue(callable(main.main))
            flash = importlib.import_module("sim.learn_flash")
            self.assertEqual(flash._PAPER, (30, 31, 34))
            self.assertEqual(flash._LINE, (70, 72, 78))
            self.assertEqual(flash._LABEL, (248, 248, 248))
            self.assertEqual(flash._VALUE, (255, 255, 255))
            self.assertIsNone(sys.modules["sim.ui_theme"])
        finally:
            for name in drop:
                sys.modules.pop(name, None)
            for name, mod in saved.items():
                if mod is not None:
                    sys.modules[name] = mod

class SideOfFrameTests(unittest.TestCase):
    def test_left_columns_feed_the_left_eye_and_right_columns_the_right(self):
        """Image left is the dog's left. The JPEG is not mirrored.

        A compact object in the left columns reaches MB_L / Л and a positive
        yaw. The right columns reach MB_R / П and a negative yaw. The 3D
        labels sit on those same anatomical sides. The default camera does
        not mirror x: it looks forward from behind the brain, and a view
        from the front still puts «Л» on the right of that screen.
        """
        from sim.frame_sense import features_from_frames
        from sim.hemifield import bilateral_yaw
        from sim.mb_flywire import _MESH_FRAG
        from sim.mb_view3d import DIST0, PITCH0, YAW0, BrainCloud, _project, eye_layout
        from sim.pilot import SEARCH_TURN, Pilot
        from sim.raw_sense import N_AZ, OFF_BODY
        from sim.yolo_teacher import DetBox, hemisphere_hits, zone_of

        def frame(side: str) -> np.ndarray:
            image = np.zeros((80, 160, 3), dtype=np.uint8)
            if side == "left":
                image[8:28, 4:28] = 255
            else:
                image[8:28, 132:156] = 255
            return image

        def body(feat: np.ndarray) -> np.ndarray:
            energy = np.zeros(N_AZ, dtype=np.float64)
            for index in range(N_AZ):
                energy[index] = float(np.sum(feat[OFF_BODY + index * 3 : OFF_BODY + index * 3 + 3]))
            return energy

        left_feat, _ego, _near = features_from_frames(frame("left"), None)
        right_feat, _ego, _near = features_from_frames(frame("right"), None)
        left_body = body(left_feat)
        right_body = body(right_feat)
        self.assertGreater(left_body[7], left_body[0])
        self.assertGreater(right_body[0], right_body[7])
        self.assertGreater(_energy(hemifield(left_feat, "L", DEFAULT_OVERLAP)), 0.0)
        self.assertEqual(_energy(hemifield(left_feat, "R", DEFAULT_OVERLAP)), 0.0)
        self.assertGreater(_energy(hemifield(right_feat, "R", DEFAULT_OVERLAP)), 0.0)
        self.assertEqual(_energy(hemifield(right_feat, "L", DEFAULT_OVERLAP)), 0.0)

        left_box = DetBox(0.02, 0.1, 0.22, 0.4, 0.9)
        right_box = DetBox(0.78, 0.1, 0.98, 0.4, 0.9)
        self.assertEqual(hemisphere_hits([left_box], DEFAULT_OVERLAP, center_only=True), (True, False))
        self.assertEqual(hemisphere_hits([right_box], DEFAULT_OVERLAP, center_only=True), (False, True))
        self.assertEqual(zone_of(left_box, DEFAULT_OVERLAP), "Л")
        self.assertEqual(zone_of(right_box, DEFAULT_OVERLAP), "П")

        self.assertGreater(bilateral_yaw(1.0, 0.0), 0.0)
        self.assertLess(bilateral_yaw(0.0, 1.0), 0.0)
        left_turn = Pilot()
        left_turn.start_auto(0.0)
        left_cmd = _auto(left_turn, 0.0, left=True, right=False)
        self.assertEqual(left_cmd.phase, "align_l")
        self.assertAlmostEqual(left_cmd.z, SEARCH_TURN)
        right_turn = Pilot()
        right_turn.start_auto(0.0)
        right_cmd = _auto(right_turn, 0.0, left=False, right=True)
        self.assertEqual(right_cmd.phase, "align_r")
        self.assertAlmostEqual(right_cmd.z, -SEARCH_TURN)

        cloud = BrainCloud()
        eyes, _arrow = eye_layout(cloud)
        placed = {item["label"]: np.asarray(item["pos"], dtype=np.float64) for item in eyes}
        self.assertLess(float(placed["зрит. доля Л"][0]), 0.0)
        self.assertGreater(float(placed["зрит. доля П"][0]), 0.0)
        screen_x, _screen_y, depth = _project(
            np.vstack((placed["зрит. доля Л"], placed["зрит. доля П"])),
            YAW0,
            PITCH0,
            DIST0,
            480.0,
            360.0,
            500.0,
        )
        self.assertLess(float(screen_x[0]), float(screen_x[1]))
        behind = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        _bx, _by, depths = _project(behind, YAW0, PITCH0, DIST0, 480.0, 360.0, 500.0)
        self.assertLess(float(depths[0]), float(depths[1]))
        self.assertLess(float(depth[0]), 1e9)

        body = _MESH_FRAG.split("void main()", 1)[1]
        right, left = body.split("else if", 1)
        self.assertIn("rec_r", right)
        self.assertNotIn("rec_l", right)
        self.assertIn("rec_l", left)
        self.assertNotIn("rec_r", left)

    def test_object_on_the_flys_left_turns_left(self):
        """The fly's left is the left side of this unmirrored frame.

        Object there → left optic lobe → left mushroom body → turn left.
        «зрит. доля Л» is the fly's left (negative x), so a view from the
        front puts it on the right of the 3D screen.
        """
        import math

        from sim.frame_sense import features_from_frames
        from sim.hemifield import bilateral_yaw
        from sim.mb_flywire import load_scene, lobe_anchors
        from sim.mb_train import MbTrainer, default_npz
        from sim.mb_view3d import DIST0, PITCH0, YAW0, BrainCloud, _project, eye_layout
        from sim.pilot import Pilot
        from sim.raw_sense import N_AZ, OFF_BODY

        image = np.zeros((80, 160, 3), dtype=np.uint8)
        image[8:28, 4:28] = 255
        feat, _ego, _near = features_from_frames(image, None)
        energy = np.zeros(N_AZ, dtype=np.float64)
        for index in range(N_AZ):
            energy[index] = float(np.sum(feat[OFF_BODY + index * 3 : OFF_BODY + index * 3 + 3]))
        self.assertGreater(energy[7], energy[0])
        left_view = hemifield(feat, "L", DEFAULT_OVERLAP)
        right_view = hemifield(feat, "R", DEFAULT_OVERLAP)
        self.assertGreater(_energy(left_view), 0.0)
        self.assertEqual(_energy(right_view), 0.0)

        brain = MbTrainer(default_npz(), seed=1, overlap=DEFAULT_OVERLAP)
        brain.forward(feat)
        self.assertGreater(float(np.abs(brain.last_fwd.pn).sum()), 0.0)
        self.assertEqual(float(np.abs(brain.last_fwd_r.pn).sum()), 0.0)
        self.assertGreater(float(np.abs(brain.last_fwd.kc).sum()) + float(np.abs(brain.last_fwd.mbon).sum()), 0.0)

        self.assertGreater(bilateral_yaw(1.0, 0.0), 0.0)
        pilot = Pilot()
        pilot.start_auto(0.0)
        turned = _auto(pilot, 0.0, left=True, right=True, r_l=5.0, r_r=0.0)
        self.assertEqual(turned.phase, "approach")
        self.assertGreater(turned.z, 0.0)

        cloud = BrainCloud()
        eyes, _arrow = eye_layout(cloud)
        placed = {item["label"]: np.asarray(item["pos"], dtype=np.float64) for item in eyes}
        self.assertLess(float(placed["зрит. доля Л"][0]), 0.0)
        self.assertGreater(float(placed["зрит. доля П"][0]), 0.0)
        self.assertLess(float(placed["зрит. доля Л"][0]), float(placed["зрит. доля П"][0]))
        front = YAW0 + math.pi
        screen_x, _sy, depth = _project(
            np.vstack((placed["зрит. доля Л"], placed["зрит. доля П"])),
            front,
            PITCH0,
            DIST0,
            480.0,
            360.0,
            500.0,
        )
        self.assertGreater(float(screen_x[0]), float(screen_x[1]))
        axis = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        _ax, _ay, depths = _project(axis, front, PITCH0, DIST0, 480.0, 360.0, 500.0)
        self.assertGreater(float(depths[0]), float(depths[1]))
        self.assertLess(float(depth[0]), 1e9)

        scene = load_scene()
        self.assertIsNotNone(scene)
        anchors = {item["label"]: item for item in lobe_anchors(scene)}
        left_lobe = np.asarray(anchors["зрит. доля Л"]["pos"], dtype=np.float64)
        right_lobe = np.asarray(anchors["зрит. доля П"]["pos"], dtype=np.float64)
        self.assertEqual(anchors["зрит. доля Л"]["tag"], 1)
        self.assertEqual(anchors["зрит. доля Л"]["side"], 1)
        self.assertLess(float(left_lobe[0]), 0.0)
        self.assertGreater(float(right_lobe[0]), 0.0)
        lobe_x, _ly, _ld = _project(
            np.vstack((left_lobe, right_lobe)),
            front,
            PITCH0,
            DIST0,
            480.0,
            360.0,
            500.0,
        )
        self.assertGreater(float(lobe_x[0]), float(lobe_x[1]))


class DepressionTests(unittest.TestCase):
    def test_dopamine_only_weakens_its_own_compartment(self):
        from sim.mb_runtime import ACTIONS, MBForward
        from sim.mb_train import MbTrainer, default_npz

        sample = MbTrainer(default_npz(), seed=1).brain
        kc = np.ones(sample.n_kc, dtype=np.float32)
        mbon, scores = sample.scores_of_kc(kc)
        fwd = MBForward(
            pn=np.zeros(sample.n_pn, dtype=np.float32),
            kc=kc,
            mbon=mbon,
            action="Explore",
            action_scores=scores,
        )
        approach = [ACTIONS.index("Approach_A"), ACTIONS.index("Approach_B")]
        avoid = [ACTIONS.index("Avoid_A"), ACTIONS.index("Avoid_B")]
        post = sample.mbon_action[sample.kc_mbon_post]

        def delta(sign: float) -> np.ndarray:
            fresh = MbTrainer(default_npz(), seed=1).brain
            before = fresh.kc_mbon_w.copy()
            fresh.plasticity(fwd, sign)
            return fresh.kc_mbon_w - before

        pam = delta(1.0)
        self.assertTrue(bool(np.all(pam <= 1e-6)))
        self.assertGreater(int(np.sum((pam < -1e-6) & np.isin(post, avoid))), 0)
        self.assertEqual(int(np.sum((np.abs(pam) > 1e-6) & np.isin(post, approach))), 0)

        punish = delta(-1.0)
        self.assertTrue(bool(np.all(punish <= 1e-6)))
        self.assertGreater(int(np.sum((punish < -1e-6) & np.isin(post, approach))), 0)
        self.assertEqual(int(np.sum((np.abs(punish) > 1e-6) & np.isin(post, avoid))), 0)


class WeightFileTests(unittest.TestCase):
    def test_a_mismatched_file_starts_clean_and_is_not_deleted(self):
        from sim.mb_train import MbTrainer, default_npz
        from sim.npz_compat import open_npz

        born = MbTrainer(default_npz(), seed=1)
        left0 = born.brain.kc_mbon_w.copy()
        right0 = born.brain_r.kc_mbon_w.copy()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mb_train_state.npz"
            born.save(path)
            with open_npz(path) as saved:
                payload = {key: np.array(saved[key]) for key in saved.files}
            payload["kc_mbon_w"] = np.zeros(3, dtype=np.float32)
            np.savez(path, **payload)
            other = MbTrainer(default_npz(), seed=1)
            self.assertFalse(other.load(path))
            self.assertIn("не загружен", other.load_note)
            self.assertIn("не удалён", other.load_note)
            self.assertTrue(path.is_file())
            np.testing.assert_array_equal(other.brain.kc_mbon_w, left0)
            np.testing.assert_array_equal(other.brain_r.kc_mbon_w, right0)

            born.save(path)
            with open_npz(path) as saved:
                legacy = {key: np.array(saved[key]) for key in saved.files if key != "kc_mbon_w_r"}
            np.savez(path, **legacy)
            kept = MbTrainer(default_npz(), seed=1)
            self.assertTrue(kept.load(path))
            self.assertEqual(kept.load_note, "")
            np.testing.assert_array_equal(kept.brain.kc_mbon_w, born.brain.kc_mbon_w)
            np.testing.assert_array_equal(kept.brain_r.kc_mbon_w, born.brain.kc_mbon_w)


class VisualPathwayTests(unittest.TestCase):
    def test_each_group_receives_only_its_own_features(self):
        from sim.kc_pathways import (
            N_MEDULLA,
            OFF_LOBULA_YOLO,
            _group_indices,
            lobula_channels,
            medulla_channels,
        )
        from sim.mb_constants import KC_ABP_PER_SIDE, KC_GD_PER_SIDE
        from sim.mb_train import MbTrainer, default_npz
        from sim.raw_sense import N_AZ, N_RAW, OFF_CLOSE, OFF_LIDAR, OFF_MOTION

        trainer = MbTrainer(default_npz(), seed=1)
        left = trainer.brain.vision
        right = trainer.brain_r.vision
        self.assertEqual(left.n_gd, KC_GD_PER_SIDE)
        self.assertEqual(right.n_gd, KC_GD_PER_SIDE)
        self.assertEqual(left.n_abp, KC_ABP_PER_SIDE)
        self.assertEqual(right.n_abp, KC_ABP_PER_SIDE)
        self.assertFalse(bool(np.any(left.gamma_mask & left.abp_mask)))
        self.assertEqual(len(set(np.unique(left.gamma_idx).tolist()) & set(np.unique(right.gamma_idx).tolist())), 0)
        self.assertEqual(len(set(np.unique(left.abp_idx).tolist()) & set(np.unique(right.abp_idx).tolist())), 0)
        for vision, side, type_name, idx in (
            (left, "left", "KCg-d", left.gamma_idx),
            (right, "right", "KCg-d", right.gamma_idx),
            (left, "left", "KCab-p", left.abp_idx),
            (right, "right", "KCab-p", right.abp_idx),
        ):
            seeds = _group_indices(trainer.brain if side == "left" else trainer.brain_r, type_name, side)
            values, counts = np.unique(idx, return_counts=True)
            self.assertEqual(set(values.tolist()), set(seeds.tolist()))
            self.assertLessEqual(int(counts.max() - counts.min()), 1)

        motion = np.zeros(N_RAW, dtype=np.float32)
        motion[OFF_MOTION + 5] = 1.0
        med = medulla_channels(motion)
        self.assertGreater(float(med[N_MEDULLA - N_AZ + 5]), 0.0)
        self.assertEqual(float(med[: N_MEDULLA - N_AZ].sum()), 0.0)
        self.assertEqual(float(lobula_channels(motion).sum()), 0.0)
        kc = left.kenyon(motion)
        self.assertGreater(int(np.count_nonzero(left.last_gamma)), 0)
        self.assertEqual(int(np.count_nonzero(left.last_abp)), 0)
        on = np.flatnonzero(left.last_gamma > 0)
        self.assertTrue(bool(np.all(left.gamma_chan[on] >= N_MEDULLA - 8)))

        yolo = np.array([0.9, 0.4, 0.5, 0.2], dtype=np.float32)
        blank = np.zeros(N_RAW, dtype=np.float32)
        self.assertEqual(float(medulla_channels(blank).sum()), 0.0)
        lob = lobula_channels(blank, yolo)
        self.assertGreater(float(lob[OFF_LOBULA_YOLO]), 0.0)
        self.assertEqual(float(lob[:OFF_LOBULA_YOLO].sum()), 0.0)
        seen = left.kenyon(blank, yolo)
        self.assertEqual(int(np.count_nonzero(left.last_gamma)), 0)
        self.assertGreater(int(np.count_nonzero(left.last_abp)), 0)
        picked = np.flatnonzero(left.last_abp > 0)
        self.assertTrue(bool(np.all(left.abp_chan[picked] >= OFF_LOBULA_YOLO)))
        self.assertEqual(int(((seen > 0) & ~left.gamma_mask & ~left.abp_mask).sum()), 0)

        lidar = np.zeros(N_RAW, dtype=np.float32)
        lidar[OFF_LIDAR + 4] = 1.0
        lidar[OFF_CLOSE + 4] = 1.0
        self.assertEqual(int(np.count_nonzero(left.kenyon(lidar))), 0)
        again = left.kenyon(motion)
        np.testing.assert_array_equal(again, kc)

    def test_onboard_step_fits_the_frame_slot(self):
        import time

        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import MbTrainer, default_npz

        brain = MbTrainer(default_npz(), seed=1)
        loop = BrainLoop(SportDrive(None), brain, "/tmp/mb_step_slot.npz")
        loop.learn = True
        camera = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera[200:400, 100:500] = (40, 180, 60)
        lidar = np.zeros((240, 240, 3), dtype=np.uint8)
        loop.ingest(camera, lidar, 1.0)
        started = time.perf_counter()
        loop.ingest(camera, lidar, 2.0)
        loop.tick(2.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # main.py sleeps 0.1 s, so the step has to finish inside that slot.
        self.assertLess(elapsed_ms, 100.0)

    def test_visual_value_is_learned_on_gamma_only(self):
        from sim.mb_train import MbTrainer, default_npz
        from sim.raw_sense import probe_features

        trainer = MbTrainer(default_npz(), seed=1)
        trainer.forward(probe_features("dog"))
        brain = trainer.brain
        pre = brain.kc_mbon_pre
        before = brain.kc_mbon_w.copy()
        brain.plasticity(trainer.last_fwd, 1.0)
        dw = brain.kc_mbon_w - before
        changed = np.abs(dw) > 1e-6
        self.assertGreater(int(changed.sum()), 0)
        self.assertTrue(bool(np.all(brain.gamma_d_kc[pre][changed])))
        self.assertEqual(int(np.sum(changed & brain.vision.abp_mask[pre])), 0)
        self.assertEqual(int(np.sum(changed & ~brain.gamma_d_kc[pre])), 0)


class PlasticityBiologyTests(unittest.TestCase):
    def test_reward_raises_approach_minus_avoidance(self):
        from sim.mb_runtime import ACTIONS
        from sim.mb_train import MbTrainer, default_npz
        from sim.raw_sense import probe_features

        trainer = MbTrainer(default_npz(), seed=1)
        feat = probe_features("dog")
        trainer.forward(feat)
        before = trainer.brain.appetitive_drive(trainer.last_fwd.action_scores)
        trainer.brain.plasticity(trainer.last_fwd, 1.0)
        trainer.forward(feat)
        after = trainer.brain.appetitive_drive(trainer.last_fwd.action_scores)
        self.assertGreater(after, before)

        fresh = MbTrainer(default_npz(), seed=1)
        fresh.forward(feat)
        punish_before = fresh.brain.appetitive_drive(fresh.last_fwd.action_scores)
        fresh.brain.plasticity(fresh.last_fwd, -1.0)
        fresh.forward(feat)
        punish_after = fresh.brain.appetitive_drive(fresh.last_fwd.action_scores)
        self.assertLess(punish_after, punish_before)

        post = trainer.brain.mbon_action[trainer.brain.kc_mbon_post]
        avoid = [ACTIONS.index("Avoid_A"), ACTIONS.index("Avoid_B")]
        approach = [ACTIONS.index("Approach_A"), ACTIONS.index("Approach_B")]
        dw = trainer.brain.kc_mbon_w - trainer.brain.kc_mbon_w0
        self.assertGreater(int(np.sum((dw < -1e-6) & np.isin(post, avoid))), 0)
        self.assertEqual(int(np.sum((np.abs(dw) > 1e-6) & np.isin(post, approach))), 0)

    def test_one_frame_keeps_a_sparse_distinct_kenyon_code(self):
        from sim.mb_constants import KC_FRAC_MAX, KC_FRAC_MIN, kc_winner_count
        from sim.mb_train import MbTrainer, default_npz
        from sim.raw_sense import probe_features

        trainer = MbTrainer(default_npz(), seed=1)

        def _group_ok(sparse):
            n = int(sparse.shape[0])
            k = kc_winner_count(n)
            on = int(np.count_nonzero(sparse))
            self.assertEqual(on, k)
            self.assertGreaterEqual(on / float(n), KC_FRAC_MIN)
            self.assertLessEqual(on / float(n), KC_FRAC_MAX)

        trainer.forward(probe_features("dog"))
        dog_g = trainer.brain.vision.last_gamma > 0
        mirror = trainer.brain.vision
        for vis, kc, home in (
            (trainer.brain.vision, trainer.last_fwd.kc, trainer.brain.vision),
            (trainer.brain_r.vision, trainer.last_fwd_r.kc, mirror),
        ):
            _group_ok(vis.last_gamma)
            _group_ok(vis.last_abp)
            self.assertEqual(int(((kc > 0) & ~home.gamma_mask & ~home.abp_mask).sum()), 0)

        trainer.forward(probe_features("distractor"))
        other = trainer.brain.vision.last_gamma > 0
        _group_ok(trainer.brain.vision.last_gamma)
        inter = int(np.logical_and(dog_g, other).sum())
        union = int(np.logical_or(dog_g, other).sum())
        self.assertGreater(union, 0)
        self.assertLess(inter / float(union), 0.20)

        trainer.forward(probe_features("empty"))
        self.assertEqual(int(np.count_nonzero(trainer.last_fwd.kc)), 0)
        self.assertEqual(int(np.count_nonzero(trainer.last_fwd_r.kc)), 0)

    def test_depressed_weights_recover_and_stay_bounded(self):
        from sim.mb_constants import W_CEIL_FRAC, W_FLOOR_FRAC
        from sim.mb_train import MbTrainer, default_npz
        from sim.raw_sense import probe_features

        trainer = MbTrainer(default_npz(), seed=1)
        trainer.forward(probe_features("dog"))
        brain = trainer.brain
        fwd = trainer.last_fwd
        born = brain.kc_mbon_w0
        floor = born * np.float32(W_FLOOR_FRAC)
        ceil = born * np.float32(W_CEIL_FRAC)
        for _ in range(80):
            brain.plasticity(fwd, -1.0)
        floored = brain.kc_mbon_w.copy()
        self.assertTrue(bool(np.all(floored + 1e-4 >= floor)))
        self.assertGreater(float(floored.min()), 0.0)
        self.assertGreater(int(np.sum(np.abs(floored - floor) < 1e-3)), 0)
        gap_before = float(np.mean(np.abs(floored - born)))
        for _ in range(300):
            brain.recover(fwd.kc, dopamine=True)
        restored = brain.kc_mbon_w
        gap_after = float(np.mean(np.abs(restored - born)))
        self.assertLess(gap_after, gap_before * 0.5)
        self.assertEqual(int(np.sum(np.abs(restored - floor) < 1e-3)), 0)
        self.assertTrue(bool(np.all(restored + 1e-4 >= floor)))
        self.assertTrue(bool(np.all(restored <= ceil + 1e-3)))
        self.assertGreater(float(restored.min()), 0.0)
        self.assertGreater(float(restored.std()), float(born.std()) * 0.5)

        quiet = MbTrainer(default_npz(), seed=1)
        quiet.forward(probe_features("dog"))
        pre = quiet.brain.kc_mbon_pre
        gamma = quiet.brain.gamma_d_kc[pre]
        abp = quiet.brain.vision.abp_mask[pre]
        active = quiet.last_fwd.kc[pre] > 0
        gamma_i = np.flatnonzero(active & gamma)
        self.assertGreater(len(gamma_i), 0)
        idx = int(gamma_i[0])
        birth = float(quiet.brain.kc_mbon_w[idx])
        abp_before = quiet.brain.kc_mbon_w[abp].copy()
        quiet.brain.recover(quiet.last_fwd.kc, dopamine=True)
        self.assertAlmostEqual(float(quiet.brain.kc_mbon_w[idx]), birth)
        quiet.brain.recover(quiet.last_fwd.kc, dopamine=False)
        grown = float(quiet.brain.kc_mbon_w[idx])
        self.assertGreater(grown, birth)
        self.assertLessEqual(grown, float(ceil[idx]) + 1e-3)
        np.testing.assert_array_equal(quiet.brain.kc_mbon_w[abp], abp_before)

    def test_identical_image_gives_the_same_sign(self):
        from sim.hemifield import hemifield, live_turn
        from sim.mb_train import MbTrainer, default_npz
        from sim.raw_sense import probe_features

        trainer = MbTrainer(default_npz(), seed=1)
        half = hemifield(probe_features("dog"), "L", 0.0)
        _fwd_l, left = trainer._drive(trainer.brain, half, None)
        _fwd_r, right = trainer._drive(trainer.brain_r, half, None)
        self.assertGreater(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(right, left, places=3)
        self.assertLess(abs(live_turn(left, right)), 0.05)
        blank = hemifield(probe_features("empty"), "L", 0.0)
        _empty_l, zero_l = trainer._drive(trainer.brain, blank, None)
        _empty_r, zero_r = trainer._drive(trainer.brain_r, blank, None)
        self.assertEqual(zero_l, 0.0)
        self.assertEqual(zero_r, 0.0)

    def test_turn_is_left_drive_minus_right_drive(self):
        from sim.hemifield import fly_turn
        from sim.mb_runtime import ACTIONS
        from sim.mb_train import MbTrainer, default_npz

        brain = MbTrainer(default_npz(), seed=1).brain
        left = np.zeros(len(ACTIONS), dtype=np.float32)
        right = np.zeros(len(ACTIONS), dtype=np.float32)
        left[ACTIONS.index("Approach_A")] = 4.0
        left[ACTIONS.index("Avoid_A")] = 1.0
        right[ACTIONS.index("Approach_A")] = 1.0
        right[ACTIONS.index("Avoid_A")] = 4.0
        drive_l = brain.appetitive_drive(left)
        drive_r = brain.appetitive_drive(right)
        self.assertAlmostEqual(drive_l, 3.0)
        self.assertAlmostEqual(drive_r, -3.0)
        self.assertGreater(fly_turn(drive_l, drive_r), 0.0)
        self.assertLess(fly_turn(drive_r, drive_l), 0.0)
        self.assertEqual(fly_turn(1.05, 1.0), 0.0)

    def test_a_quiet_onboard_frame_settles_when_learning_is_on(self):
        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import MbTrainer, default_npz
        from sim.raw_sense import probe_features

        trainer = MbTrainer(default_npz(), seed=1)
        trainer.forward(probe_features("dog"))
        for _ in range(40):
            trainer.brain.plasticity(trainer.last_fwd, -1.0)
        born = trainer.brain.kc_mbon_w0
        dropped = trainer.brain.kc_mbon_w.copy()
        gap_dropped = float(np.mean(np.abs(dropped - born)))
        loop = BrainLoop(SportDrive(None), trainer, "/tmp/mb_plastic_settle.npz")
        camera = np.zeros((16, 16, 3), dtype=np.uint8)
        loop.ingest(camera, None, 1.0)
        np.testing.assert_array_equal(trainer.brain.kc_mbon_w, dropped)
        loop.learn = True
        loop.ingest(camera, None, 2.0)
        gap_settled = float(np.mean(np.abs(trainer.brain.kc_mbon_w - born)))
        self.assertLess(gap_settled, gap_dropped)
        self.assertGreater(float(trainer.brain.kc_mbon_w.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
