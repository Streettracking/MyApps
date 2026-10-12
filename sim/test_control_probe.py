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
        self.assertGreater(turn["raw_right"], 0.0)
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
        self.assertGreater(left["approach_steps"], 0, report)
        self.assertGreater(left["search_steps"], left["approach_steps"], report)
        self.assertGreater(left["rate_conspecific"], left["rate_control"], report)
        self.assertGreater(right["rate_control"], right["rate_conspecific"], report)
        self.assertAlmostEqual(left["rate_conspecific"], right["rate_control"], places=2)
        self.assertGreater(left["seen_conspecific"], 0, report)
        self.assertGreater(right["seen_control"], 0, report)
        self.assertLess(half, 1.0)

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


if __name__ == "__main__":
    unittest.main()
