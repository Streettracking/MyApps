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

    def test_seed_changes_the_probe(self):
        from sim.control_scene import arrange, make_scene, run_memory_probe
        from sim.raw_sense import render_view

        first = make_scene(1)
        second = make_scene(2)
        self.assertEqual(float(first[0].probe_bearing), 0.0)
        self.assertGreater(float(second[0].probe_bearing), 0.0)
        self.assertGreater(float(second[0].probe_dist), 0.0)
        arrange(first[0], first[1], conspecific="left", control="right")
        arrange(second[0], second[1], conspecific="left", control="right")
        feat_a, _hit_a = render_view(first[1], first[0])
        feat_b, _hit_b = render_view(second[1], second[0])
        self.assertFalse(np.array_equal(feat_a, feat_b))
        one = run_memory_probe(seed=2)
        other = run_memory_probe(seed=3)
        self.assertNotEqual(one["before"], other["before"])
        self.assertNotEqual(one["seed"], other["seed"])

    def test_conspecific_motion_changes_the_camera(self):
        from sim.control_scene import make_scene
        from sim.frame_sense import sim_previews
        from sim.raw_sense import BODY_RADIUS

        world, learner = make_scene()
        peer = next(agent for agent in world.agents if agent.agent_id != learner.agent_id)
        cube = world.controls[0]
        # Same start, a sub-bin step. The old body sprite stayed put inside a sector.
        peer.x, peer.y = learner.x + 1.4, learner.y + 0.15
        cube.x, cube.y = -10.0, -10.0
        before, _lid = sim_previews(learner, world)
        peer.x += 0.07
        peer.y += 0.04
        after, _lid = sim_previews(learner, world)
        dog_delta = int(np.abs(before.astype(np.int16) - after.astype(np.int16)).sum())
        self.assertGreater(dog_delta, 0)

        peer.x, peer.y = -10.0, -10.0
        cube.x, cube.y = learner.x + 1.4, learner.y + 0.15
        cube.radius = BODY_RADIUS
        still, _lid = sim_previews(learner, world)
        cube.x += 0.07
        cube.y += 0.04
        moved, _lid = sim_previews(learner, world)
        cube_delta = int(np.abs(still.astype(np.int16) - moved.astype(np.int16)).sum())
        self.assertGreater(cube_delta, 0)

    def test_approach_probe_reaches_the_conspecific(self):
        from sim.control_scene import run_approach_probe

        report = run_approach_probe()
        turn = report["turn"]
        self.assertGreater(turn["raw_left"], 0.0)
        self.assertLess(turn["raw_right"], 0.0)
        self.assertGreater(turn["gain_left"], 0.0)
        self.assertLess(turn["gain_right"], 0.0)
        before = report["before"]
        self.assertGreater(before["rate_conspecific"] + before["rate_control"], 0.5)
        for name in ("after", "swap"):
            row = report[name]
            self.assertGreater(row["rate_conspecific"], 0.5, report)
            self.assertLess(row["rate_control"], 0.25, report)
            self.assertTrue(row["reached_conspecific"], report)
            self.assertLess(row["search_steps"], 100, report)
            self.assertGreater(row["seen_conspecific"], 35, report)


class SimRoundTests(unittest.TestCase):
    def test_busy_udp_without_main_py_is_silent(self):
        import socket

        from sim.onboard_link import main_py_listening, udp_bridge_listening
        from sim.recognize_train_live import _note_bridge

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("127.0.0.1", 0))
        except OSError:
            sock.close()
            self.assertFalse(main_py_listening())
            return
        port = int(sock.getsockname()[1])
        try:
            self.assertTrue(udp_bridge_listening("127.0.0.1", port))
            self.assertFalse(main_py_listening(port))
        finally:
            sock.close()

        class _Session:
            bridge_open = False
            _bridge_warned = False
            log = []

            def _log(self, text):
                self.log.append(text)

        session = _Session()
        _note_bridge(session, 0.0, {}, link_ok=False)
        _note_bridge(session, 2.0, {}, link_ok=True)
        self.assertFalse(session.bridge_open)
        self.assertFalse(any("два источника" in line for line in session.log))

    def test_sim_status_pause_on_localhost(self):
        import json
        import urllib.request

        from sim.recognize_train import RecognizeTrainSim
        from sim.sim_status import serve_sim

        session = RecognizeTrainSim(n_agents=1, seed=1, learner="mb", dan="teacher", auto_teach=False)
        server = serve_sim(session, 0)
        self.assertIsNotNone(server)
        self.assertEqual(server.server_address[0], "127.0.0.1")
        port = int(server.server_address[1])
        try:
            with urllib.request.urlopen("http://127.0.0.1:%s/status" % port, timeout=2) as resp:
                status = json.loads(resp.read().decode("utf-8"))
            self.assertFalse(status["paused"])
            self.assertIn("drift", status)
            self.assertIn("session_drift", status)
            req = urllib.request.Request(
                "http://127.0.0.1:%s/pause" % port,
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                paused = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(paused["paused"])
            self.assertTrue(paused["learning_frozen"])
            req = urllib.request.Request(
                "http://127.0.0.1:%s/cmd" % port,
                data=json.dumps({"op": "resume"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                resumed = json.loads(resp.read().decode("utf-8"))
            self.assertFalse(resumed["paused"])
            self.assertTrue(resumed["learning_frozen"])
        finally:
            server.shutdown()
            server.server_close()

    def test_session_drift_is_zero_at_load(self):
        from pathlib import Path

        from sim.brain_panel import live_drift, session_drift
        from sim.mb_train import MbTrainer, default_npz

        trainer = MbTrainer(default_npz(), seed=1)
        path = Path("logs/mb_train_state.npz")
        if path.is_file():
            self.assertTrue(trainer.load(path))
            self.assertAlmostEqual(session_drift(trainer), 0.0, places=4)
            self.assertAlmostEqual(live_drift(trainer), trainer.drift_session0, places=4)
        trainer.drift_session0 = float(live_drift(trainer))
        self.assertAlmostEqual(session_drift(trainer), 0.0, places=4)
        shifted = np.array(trainer.brain.kc_mbon_w, dtype=np.float32, copy=True)
        shifted.flat[0] = float(shifted.flat[0]) + 3.0
        trainer.brain.kc_mbon_w = shifted
        self.assertGreater(abs(session_drift(trainer)), 0.5)
        self.assertAlmostEqual(
            live_drift(trainer) - session_drift(trainer),
            float(trainer.drift_session0),
            places=3,
        )

    def test_too_close_fires_ppl1(self):
        from sim.recognize_train import run_too_close
        from sim.stimuli import CLOSE_HEIGHT, TOO_CLOSE_M, camera_stimuli
        from sim.yolo_teacher import DetBox

        close = DetBox(0.4, 0.5 - CLOSE_HEIGHT / 2, 0.6, 0.5 + CLOSE_HEIGHT / 2, 0.9)
        far = DetBox(0.45, 0.4, 0.55, 0.55, 0.9)
        self.assertTrue(camera_stimuli([close], None)[0])
        self.assertFalse(camera_stimuli([far], None)[0])
        report = run_too_close()
        self.assertLess(report["distance_m"], TOO_CLOSE_M)
        self.assertGreater(report["n_ppl1"], report["n_ppl1_before"])
        self.assertEqual(report["far_added"], 0)

    def test_empty_frame_scans_without_learning(self):
        from sim.recognize_train import RecognizeTrainSim, _drive_sim
        from sim.search_reflex import SEARCH_REFLEX_YAW
        from sim.train_monitor import MonitorInput

        session = RecognizeTrainSim(n_agents=2, seed=1, auto_teach=True)
        session.state_path = None
        session.learner.x, session.learner.y, session.learner.yaw = 3.0, 2.5, 0.0
        for peer in session.peers:
            peer.x, peer.y = session.learner.x - 1.6, session.learner.y
        session.world.distractors.clear()
        session.freeze_peers = True
        session.pilot.start_auto(0.0)
        born = np.array(session.mb.brain.kc_mbon_w, copy=True)
        pam = int(session.mb.n_pam)
        inp = MonitorInput()
        for _ in range(6):
            sx, sz, rate, _name = _drive_sim(session, inp, float(session.world.t))
            self.assertEqual(sx, 0.0)
            self.assertAlmostEqual(sz, SEARCH_REFLEX_YAW)
            self.assertTrue(rate)
            self.assertTrue(session.searching)
            session.step(sx, sz, rate=rate)
            self.assertEqual(session.mb.n_pam, pam)
        np.testing.assert_array_equal(session.mb.brain.kc_mbon_w, born)
        self.assertFalse(session.target_in_view)

        session.learner.yaw = 0.0
        session.peers[0].x = session.learner.x + 1.2
        session.peers[0].y = session.learner.y
        session.step(0.0, 0.0)
        self.assertTrue(session.target_in_view)
        sx, sz, rate, _name = _drive_sim(session, inp, float(session.world.t))
        self.assertFalse(session.searching)
        self.assertGreater(sx, 0.0)
        self.assertTrue(rate)

    def test_opposed_probe_alternates_sides(self):
        from sim.control_scene import conspecific_side, run_approach_probe
        from sim.raw_sense import FOV

        self.assertEqual(conspecific_side(1), "left")
        self.assertEqual(conspecific_side(2), "right")
        report = run_approach_probe(seed=1, placement="opposed")
        self.assertEqual(report["side"], "left")
        self.assertEqual(report["swap"]["side"], "right")
        half = float(FOV) / 2.0
        for name in ("before", "after", "swap"):
            row = report[name]
            self.assertTrue(row["in_view_conspecific"], row)
            self.assertTrue(row["in_view_control"], row)
            self.assertLess(row["search_steps"], row["steps"], row)
        left = report["after"]
        right = report["swap"]
        self.assertGreater(left["z"], 0.2, report)
        self.assertLess(right["z"], -0.2, report)
        self.assertLess(abs(left["z"]), 0.999, report)
        self.assertLess(abs(right["z"]), 0.999, report)
        self.assertLess(abs(report["before"]["z"]), 0.08, report)
        self.assertGreater(left["index"], 0.0, report)
        self.assertGreater(right["index"], 0.0, report)
        self.assertGreater(left["seen_conspecific"], 0, report)
        self.assertGreater(right["seen_conspecific"], 0, report)
        self.assertLess(half, 1.0)

    def test_live_turn_follows_the_side(self):
        from sim.control_scene import _hemidrives, _place_opposed, _train_side, _view, arrange, make_scene
        from sim.hemifield import live_turn
        from sim.mb_train import MbTrainer, default_npz
        from sim.raw_sense import OFF_BODY, OFF_MOTION, probe_features
        from sim.teacher_lead import attach_recognition_stimuli

        world, learner = make_scene(1)
        trainer = MbTrainer(default_npz(), seed=1, camera_only=True)
        trainer.learn = True
        attach_recognition_stimuli(trainer.stimuli)

        def yaw(feat):
            _hemidrives(trainer, feat)
            return float(live_turn(trainer.r_l, trainer.r_r, trainer.base_l, trainer.base_r))

        self.assertEqual(yaw(probe_features("empty")), 0.0)
        symmetric = probe_features("empty")
        for sector, value in ((3, 0.8), (2, 0.45)):
            mirror = 7 - sector
            for slot in (sector, mirror):
                symmetric[OFF_BODY + slot * 3 : OFF_BODY + slot * 3 + 3] = value
                symmetric[OFF_MOTION + slot] = 0.3
        self.assertLess(abs(yaw(symmetric)), 0.05)
        arrange(world, learner, conspecific="left", control=None, motion="static")
        left_before = yaw(_view(learner, world)[0])
        arrange(world, learner, conspecific="right", control=None, motion="static")
        right_before = yaw(_view(learner, world)[0])
        _place_opposed(world, learner, "left", "static")
        pair_before = yaw(_view(learner, world)[0])
        self.assertGreater(left_before, 0.2)
        self.assertLess(right_before, -0.2)
        self.assertLess(abs(pair_before), 0.08)
        self.assertLess(abs(left_before), 0.999)
        self.assertLess(abs(right_before), 0.999)
        now = 0.0
        for side in ("left", "right"):
            now = _train_side(trainer, world, learner, side, now, "static")
        arrange(world, learner, conspecific="left", control=None, motion="static")
        left_after = yaw(_view(learner, world)[0])
        arrange(world, learner, conspecific="right", control=None, motion="static")
        right_after = yaw(_view(learner, world)[0])
        _place_opposed(world, learner, "left", "static")
        pair_left = yaw(_view(learner, world)[0])
        _place_opposed(world, learner, "right", "static")
        pair_right = yaw(_view(learner, world)[0])
        self.assertGreater(left_after, 0.2)
        self.assertLess(right_after, -0.2)
        self.assertGreater(pair_left, 0.2)
        self.assertLess(pair_right, -0.2)
        self.assertLess(abs(left_after), 1.0)
        self.assertLess(abs(right_after), 1.0)
        self.assertNotAlmostEqual(abs(left_after), 1.0, places=3)
        self.assertNotAlmostEqual(abs(pair_right), 1.0, places=3)

    def test_live_memory_trial_prefers_both_sides(self):
        from sim.control_scene import run_live_memory_trial

        report = run_live_memory_trial(seeds=(1, 2, 3), runs=4)
        self.assertEqual(report["teacher_weight"], 0.0)
        self.assertTrue(report["reinforcement_off"])
        self.assertEqual(report["motion"], "static")
        self.assertLess(abs(report["untrained"]), 0.05, report)
        self.assertGreater(report["trained"], 0.5, report)
        self.assertGreater(report["trained_left"], 0.5, report)
        self.assertGreater(report["trained_right"], 0.5, report)
        bearings = {}
        trained_left_z = set()
        for row in report["rows"]:
            self.assertEqual(row["teacher_weight"], 0.0)
            self.assertTrue(row["reinforcement_off"])
            self.assertTrue(row["in_view_conspecific"], row)
            self.assertTrue(row["in_view_control"], row)
            self.assertLess(abs(row["z"]), 0.999, row)
            if row["trained"]:
                self.assertGreater(row["index"], 0.0, row)
                self.assertTrue(row["weights_frozen"], row)
                if row["side"] == "left":
                    self.assertGreater(row["z"], 0.2, row)
                    trained_left_z.add(round(row["z"], 3))
                else:
                    self.assertLess(row["z"], -0.2, row)
            else:
                self.assertLess(abs(row["z"]), 0.08, row)
            bearings.setdefault(int(row["seed"]), set()).add(round(float(row["bearing"]), 3))
        self.assertGreater(len(bearings[1]), 1)
        self.assertNotEqual(bearings[1], bearings[2])
        self.assertNotEqual(bearings[1], bearings[3])
        self.assertGreater(len(trained_left_z), 1)

    def test_search_lamp_and_drift_span(self):
        import pygame

        from sim.tabnum import format_drift_span
        from sim.test_brain_panel import _populated_view
        from sim.train_monitor import TrainMonitor
        from sim.ui_theme import apply_theme

        line = format_drift_span(12.4, 169.8)
        self.assertIn("дрейф: сессия", line)
        self.assertIn("с начала", line)
        self.assertIn("+12.4", line)
        self.assertIn("+169.8", line)
        pygame.init()
        mon = TrainMonitor("search-lamp")
        apply_theme("dark")
        mon._bg = None
        mon._win_bg = None
        view = _populated_view()
        view.session_drift = 12.4
        view.drift = 169.8
        view.search_lamp = True
        try:
            mon._on_resize(1920, 1017)
            mon._brain_panel_cache = None
            mon.draw(view)
            x, y = mon.search_lamp_at
            self.assertGreater(x, 0)
            pixel = mon.screen.get_at((int(x), int(y)))
            self.assertGreater(pixel.r, 140)
            view.search_lamp = False
            mon._brain_panel_cache = None
            mon.draw(view)
            dim = mon.screen.get_at((int(mon.search_lamp_at[0]), int(mon.search_lamp_at[1])))
            self.assertLess(dim.g, pixel.g)
        finally:
            apply_theme("light")
            pygame.quit()


class RunMarkerTests(unittest.TestCase):
    def test_label_changes_once_and_frames_keep_it(self):
        from sim.run_marker import RunMarker, apply_marker_fields
        from sim.sim_status import apply_sim_command, sim_status

        marker = RunMarker()
        self.assertEqual(marker.label(), "до · слева")
        self.assertFalse(marker.set_stage("до"))
        self.assertTrue(marker.set_stage("train"))
        self.assertEqual(marker.stage, "обучение")
        self.assertTrue(marker.set_side("right"))
        marker.note(1.0, 480.0, 12.5)
        marker.note(1.1, -3.0, 20.0)
        self.assertEqual(marker.drive_log[-1]["label"], "обучение · справа")
        self.assertEqual(marker.drive_log[0]["drive_l"], 480.0)
        self.assertEqual(marker.drive_log[0]["drive_r"], 12.5)
        self.assertEqual(marker.drive_log[0]["label"], "обучение · справа")

        class _Session:
            paused = False
            learn = True
            mb = None
            approach = None
            pilot = None
            log = []

            def __init__(self):
                self.run_marker = RunMarker()
                self.world = type("W", (), {"t": 2.0})()

            def _log(self, text):
                self.log.append(text)

        session = _Session()
        session.run_marker.note(2.0, 11.0, 22.0)
        body = sim_status(session)
        self.assertEqual(body["run_label"], "до · слева")
        self.assertEqual(body["drive_log"][0]["drive_l"], 11.0)
        self.assertEqual(body["drive_log"][0]["drive_r"], 22.0)
        self.assertEqual(body["drive_log"][0]["label"], "до · слева")
        apply_sim_command(session, {"op": "run_marker", "stage": "после", "side": "слева"})
        self.assertEqual(session.run_marker.label(), "после · слева")
        self.assertEqual(sum(1 for line in session.log if line.startswith("заход:")), 1)
        self.assertIn("после", session.log[-1])
        self.assertIn("слева", session.log[-1])
        apply_sim_command(session, {"op": "run_marker", "stage": "после", "side": "слева"})
        self.assertEqual(sum(1 for line in session.log if line.startswith("заход:")), 1)
        apply_marker_fields(session, "после", "справа")
        self.assertTrue(any("после" in line and "справа" in line for line in session.log))
        self.assertEqual(sum(1 for line in session.log if line.startswith("заход:")), 2)

    def test_sim_step_logs_raw_drives_with_the_mark(self):
        from sim.recognize_train import RecognizeTrainSim, _apply_pilot_keys
        from sim.sim_status import sim_status
        from sim.train_monitor import MonitorInput

        session = RecognizeTrainSim(n_agents=1, seed=1, learner="mb", dan="teacher", auto_teach=False)
        _apply_pilot_keys(session, MonitorInput(run_stage="обучение", run_side="справа"), 0.0)
        self.assertTrue(any("заход:" in line and "обучение" in line and "справа" in line for line in session.log))
        before = sum(1 for line in session.log if "заход:" in line)
        _apply_pilot_keys(session, MonitorInput(run_stage="обучение", run_side="справа"), 0.0)
        self.assertEqual(sum(1 for line in session.log if "заход:" in line), before)
        session.step(0.0, 0.0)
        session.step(0.0, 0.0)
        body = sim_status(session)
        self.assertEqual(body["run_label"], "обучение · справа")
        self.assertGreaterEqual(len(body["drive_log"]), 2)
        for row in body["drive_log"]:
            self.assertEqual(row["label"], "обучение · справа")
            self.assertEqual(row["stage"], "обучение")
            self.assertEqual(row["side"], "справа")
            self.assertIsInstance(row["drive_l"], float)
            self.assertIsInstance(row["drive_r"], float)
        view = session.view(True, "", "", "")
        self.assertEqual(view.run_stage, "обучение")
        self.assertEqual(view.run_side, "справа")

    def test_onboard_status_keeps_the_mark_on_each_frame(self):
        import json

        import numpy as np

        from robot.flybrain_onboard.app import BrainLoop
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import MbTrainer, default_npz

        loop = BrainLoop(SportDrive(None), MbTrainer(default_npz(), seed=1), "/tmp/run_marker_status.npz")
        loop.handle({"op": "run_marker", "stage": "до", "side": "слева"}, 1.0)
        camera = np.zeros((48, 64, 3), dtype=np.uint8)
        lidar = np.zeros_like(camera)
        loop.ingest(camera, lidar, 1.0)
        loop.handle({"op": "run_marker", "stage": "после", "side": "справа"}, 1.2)
        loop.ingest(camera, lidar, 1.2)
        body = json.loads(json.dumps(loop.status()))
        self.assertEqual(body["run_stage"], "после")
        self.assertEqual(body["run_side"], "справа")
        self.assertEqual(body["run_label"], "после · справа")
        self.assertEqual(len(body["drive_log"]), 2)
        self.assertEqual(body["drive_log"][0]["label"], "до · слева")
        self.assertEqual(body["drive_log"][1]["label"], "после · справа")
        self.assertEqual(body["drive_log"][1]["t"], 1.2)
        self.assertIsInstance(body["drive_log"][1]["drive_l"], float)
        self.assertIsInstance(body["drive_log"][1]["drive_r"], float)
        self.assertEqual(body["drive_l"], body["drive_log"][1]["drive_l"])
        self.assertEqual(body["drive_r"], body["drive_log"][1]["drive_r"])


class LinkHoldTests(unittest.TestCase):
    def test_get_polling_lets_the_link_expire(self):
        import json
        import time
        import urllib.request

        from robot.flybrain_onboard.app import BrainLoop, serve
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import MbTrainer, default_npz
        from sim.pilot import LINK_HOLD_S

        loop = BrainLoop(SportDrive(None), MbTrainer(default_npz(), seed=1), "/tmp/link_hold.npz")
        server = serve(loop, 0)
        port = int(server.server_address[1])
        try:
            def get():
                with urllib.request.urlopen("http://127.0.0.1:%d/status" % port, timeout=2.0) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            def post(path, payload):
                req = urllib.request.Request(
                    "http://127.0.0.1:%d%s" % (port, path),
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            body = post("/cmd", {"op": "autonomy_on"})
            self.assertEqual(body["mode"], "auto")
            held = loop.last_hb
            for _ in range(4):
                got = get()
                self.assertEqual(got["mode"], "auto")
            self.assertEqual(loop.last_hb, held)

            fresh = held + 0.50
            self.assertLess(fresh - held, LINK_HOLD_S)
            loop.frames_ok = True
            loop.last_frame_t = fresh
            loop.dist_m = 3.0
            loop.forward_m = 3.0
            moving = loop.tick(fresh)
            self.assertFalse(moving.stop)
            self.assertGreater(abs(moving.z) + abs(moving.x), 0.05)
            self.assertEqual(loop.drive.stops, 0)
            self.assertGreater(len(loop.drive.moves), 0)
            for _ in range(3):
                get()
            self.assertEqual(loop.last_hb, held)

            expired = held + LINK_HOLD_S + 0.05
            loop.frames_ok = True
            loop.last_frame_t = expired
            for _ in range(3):
                get()
            self.assertEqual(loop.last_hb, held)
            stopped = loop.tick(expired)
            self.assertTrue(stopped.stop)
            self.assertEqual((stopped.x, stopped.z), (0.0, 0.0))
            self.assertEqual(loop.drive.stops, 1)
            again = loop.tick(expired + 0.2)
            self.assertTrue(again.stop)
            self.assertEqual(loop.drive.stops, 1)

            post("/heartbeat", {})
            self.assertGreater(loop.last_hb, held)
            revived = loop.last_hb + 0.10
            loop.frames_ok = True
            loop.last_frame_t = revived
            walking = loop.tick(revived)
            self.assertFalse(walking.stop)
            self.assertEqual(loop.drive.stops, 1)
        finally:
            server.shutdown()
            server.server_close()

    def test_onboard_log_is_about_two_hertz_and_the_trainer_posts_a_heartbeat(self):
        import json
        import tempfile
        import time
        from pathlib import Path

        from sim.onboard_link import HEARTBEAT_PERIOD_S
        from sim.onboard_status_log import STATUS_PERIOD_S, append_onboard_status, onboard_status_path
        from sim.pilot import LINK_HOLD_S
        from sim.recognize_train_live import LiveSession, _drive_onboard
        from sim.train_monitor import MonitorInput

        self.assertLess(HEARTBEAT_PERIOD_S, LINK_HOLD_S)
        self.assertAlmostEqual(STATUS_PERIOD_S, 0.5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = {}
            wall = 1_760_000_000.0
            first = append_onboard_status(root, {"mode": "auto", "drive_l": 1.5}, 10.0, state, wall=wall)
            self.assertIsNotNone(first)
            self.assertEqual(first.name, "onboard_status_%s.jsonl" % time.strftime("%Y-%m-%d", time.localtime(wall)))
            skipped = append_onboard_status(root, {"mode": "auto", "drive_l": 2.0}, 10.0 + STATUS_PERIOD_S - 0.01, state, wall=wall)
            self.assertIsNone(skipped)
            second = append_onboard_status(root, {"mode": "auto", "drive_l": 3.0}, 10.0 + STATUS_PERIOD_S, state, wall=wall)
            self.assertEqual(second, first)
            lines = first.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            row = json.loads(lines[1])
            self.assertEqual(row["drive_l"], 3.0)
            self.assertEqual(row["logged_at"], time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall)))
            self.assertEqual(onboard_status_path(root, wall), first)

            class _Link:
                def __init__(self):
                    self.ops = []
                    self.status = {"mode": "auto", "phase": "search", "learning": False}
                    self.ok = True

                def poll(self):
                    self.ops.append("GET")
                    return self.status

                def heartbeat(self):
                    self.ops.append("heartbeat")
                    return self.status

                def post(self, op, **_extra):
                    self.ops.append(op)
                    return self.status

            session = LiveSession(root / "state.npz", onboard=True)
            session.status_log_root = root
            link = _Link()
            tele = {"poll": 0.0, "bridge": 50.0, "manual": 0.0, "axes": False}
            _drive_onboard(session, link, MonitorInput(), tele, 50.0)
            self.assertIn("heartbeat", link.ops)
            self.assertIn("GET", link.ops)
            self.assertTrue(any("статус борта:" in line and "onboard_status_" in line for line in session.log))
            beats = link.ops.count("heartbeat")
            _drive_onboard(session, link, MonitorInput(), tele, 50.0 + HEARTBEAT_PERIOD_S - 0.05)
            self.assertEqual(link.ops.count("heartbeat"), beats)
            _drive_onboard(session, link, MonitorInput(), tele, 50.0 + HEARTBEAT_PERIOD_S + 0.01)
            self.assertEqual(link.ops.count("heartbeat"), beats + 1)


class OnboardTakeoverTests(unittest.TestCase):
    def test_takeover_pauses_on_8090_and_the_stick_moves_while_learning_stays_frozen(self):
        import numpy as np

        from robot.flybrain_onboard.app import BrainLoop, serve
        from robot.flybrain_onboard.drive import SportDrive
        from sim.mb_train import PAUSE_GRACE_S, MbTrainer, default_npz
        from sim.onboard_link import OnboardLink
        from sim.pilot import Pilot
        from sim.recognize_train_live import _apply_live_keys
        from sim.train_monitor import MonitorInput

        loop = BrainLoop(SportDrive(None), MbTrainer(default_npz(), seed=1, overlap=0.4), "/tmp/takeover_pause.npz")
        loop.learn = True
        loop.mb.learn = True
        server = serve(loop, 0)
        port = int(server.server_address[1])
        try:
            link = OnboardLink("127.0.0.1", port)
            link.post("autonomy_on")
            camera = np.zeros((32, 32, 3), dtype=np.uint8)
            loop.ingest(camera, camera, loop.last_hb)
            loop.frames_ok = True
            loop.last_frame_t = loop.last_hb
            loop.dist_m = 3.0
            loop.forward_m = 3.0
            loop.mb.r_l = 4.0
            loop.mb.r_r = 4.0
            moving = loop.tick(loop.last_hb)
            self.assertFalse(moving.stop)
            self.assertGreater(abs(moving.x) + abs(moving.z), 0.05)
            self.assertFalse(loop.status()["paused"])

            # Pause alone still ignores the stick. Takeover is what hands it over.
            link.pause()
            held_at = loop.last_hb
            loop.handle({"op": "manual", "x": 0.5, "z": 0.6}, held_at)
            blocked = loop.tick(held_at)
            self.assertTrue(blocked.stop)
            self.assertEqual((blocked.x, blocked.z), (0.0, 0.0))
            self.assertFalse(loop.status()["took_over"])
            self.assertTrue(loop.status()["paused"])
            self.assertTrue(loop.status()["learning_frozen"])
            loop.last_manual_t = 0.0
            loop.manual_x = 0.0
            loop.manual_z = 0.0

            calls = []
            original = link._request

            def _spy(method, path, payload=None, timeout=0.8):
                calls.append((method, path, None if payload is None else dict(payload)))
                return original(method, path, payload, timeout)

            link._request = _spy

            class _Desk:
                def __init__(self):
                    self.pilot = Pilot()
                    self.learner_kind = "mb"
                    self.onboard = True
                    self.remote = {}
                    self.lines = []

                def _log(self, text):
                    self.lines.append(str(text))

            desk = _Desk()
            pressed = MonitorInput(takeover=True)
            _apply_live_keys(desk, pressed, link)
            pause_at = next(i for i, row in enumerate(calls) if row[0] == "POST" and row[1] == "/pause")
            take_at = next(
                i
                for i, row in enumerate(calls)
                if row[0] == "POST" and row[1] == "/cmd" and row[2].get("op") == "takeover"
            )
            self.assertLess(pause_at, take_at)
            self.assertEqual(calls[pause_at][2], {})
            self.assertTrue(any("перехват" in line for line in desk.lines))

            body = loop.status()
            self.assertTrue(body["paused"])
            self.assertTrue(body["learning_frozen"])
            self.assertEqual(body["mode"], "manual")
            self.assertTrue(body["took_over"])
            self.assertFalse(body["autonomy"])

            now = loop.last_hb
            loop.frames_ok = True
            loop.last_frame_t = now
            quiet = loop.tick(now)
            self.assertTrue(quiet.stop)
            self.assertEqual((quiet.x, quiet.z), (0.0, 0.0))

            loop.handle({"op": "manual", "x": 0.45, "z": -0.8}, now)
            driven = loop.tick(now)
            self.assertFalse(driven.stop)
            self.assertGreater(abs(driven.x) + abs(driven.z), 0.05)
            self.assertEqual(driven.phase, "manual")
            self.assertEqual(driven.who, "оператор")
            held = loop.status()
            self.assertTrue(held["paused"])
            self.assertTrue(held["learning_frozen"])
            self.assertEqual(held["mode"], "manual")

            loop.handle({"op": "external_stop", "on": True}, now)
            loop.handle({"op": "manual", "x": 0.5, "z": 0.5}, now)
            locked = loop.tick(now)
            self.assertTrue(locked.stop)
            self.assertEqual((locked.x, locked.z), (0.0, 0.0))
            self.assertTrue(loop.status()["learning_frozen"])
            loop.handle({"op": "external_stop", "on": False}, now)
            loop.last_manual_t = 0.0

            left = np.array(loop.mb.brain.kc_mbon_w, copy=True)
            right = np.array(loop.mb.brain_r.kc_mbon_w, copy=True)
            before = int(loop.mb.n_ppl1)
            loss = {
                "op": "teacher_aim",
                "x": 0.0,
                "z": 0.35,
                "weight": 1.0,
                "lost": True,
                "last_l": True,
            }
            loop.handle(loss, now)
            loop.handle({"op": "punish"}, now)
            loop.ingest(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8), now)
            loop.tick(now)
            self.assertEqual(loop.mb.n_ppl1, before)
            self.assertEqual(loop.mb.teacher_ppl1_l, 0)
            self.assertFalse(loop.status()["ppl1_event"])
            np.testing.assert_array_equal(loop.mb.brain.kc_mbon_w, left)
            np.testing.assert_array_equal(loop.mb.brain_r.kc_mbon_w, right)
            self.assertTrue(loop.status()["paused"])
            self.assertTrue(loop.status()["learning_frozen"])

            desk.remote = dict(loop.status())
            calls.clear()
            _apply_live_keys(desk, MonitorInput(autonomy_toggle=True), link)
            resume_at = next(i for i, row in enumerate(calls) if row[0] == "POST" and row[1] == "/resume")
            auto_at = next(
                i
                for i, row in enumerate(calls)
                if row[0] == "POST" and row[1] == "/cmd" and row[2].get("op") == "autonomy_on"
            )
            self.assertLess(resume_at, auto_at)
            returned = loop.status()
            self.assertFalse(returned["paused"])
            self.assertTrue(returned["autonomy"])
            self.assertFalse(returned["took_over"])
            self.assertEqual(returned["mode"], "auto")
            self.assertTrue(returned["learning_frozen"])

            back = loop.last_hb
            loop.frames_ok = True
            loop.last_frame_t = back
            loop.dist_m = 3.0
            loop.forward_m = 3.0
            loop.teacher_weight = 0.0
            walking = loop.tick(back)
            self.assertFalse(walking.stop)
            self.assertTrue(loop.status()["learning_frozen"])
            later = back + PAUSE_GRACE_S
            loop.frames_ok = True
            loop.last_frame_t = later
            loop.handle(dict(loss), later)
            learned = loop.tick(later)
            self.assertFalse(learned.stop)
            self.assertFalse(loop.status()["paused"])
            self.assertFalse(loop.status()["learning_frozen"])
            self.assertEqual(loop.mb.n_ppl1, before + 1)
            self.assertTrue(loop.status()["ppl1_event"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
