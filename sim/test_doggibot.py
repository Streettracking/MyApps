"""DoggiBOT snapshot: local files only, written from bytes the main thread copied.

Run: SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -X dev -m unittest sim.test_doggibot
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class DoggiBotPathTests(unittest.TestCase):
    def test_launchers_live_on_the_desktop_and_never_in_internal(self):
        from sim.doggibot_snap import appdata_root, desktop_flywire, doggibot_root, exe_launchers

        home = Path(tempfile.mkdtemp())
        desktop = home / "Desktop" / "FlyWire-Go2"
        desktop.mkdir(parents=True)
        self.assertEqual(desktop_flywire(home), desktop)
        internal = home / "Desktop" / "FlyWire-Go2" / "_internal"
        self.assertIsNone(exe_launchers(internal / "recognize_trainer.exe", frozen=False))
        self.assertEqual(exe_launchers(internal / "recognize_trainer.exe", frozen=True), desktop)
        self.assertNotIn("_internal", appdata_root().parts)

        os.environ["RECOGNIZER_DOGGIBOT_DIR"] = str(home / "shots")
        try:
            self.assertEqual(doggibot_root(), home / "shots")
            os.environ["RECOGNIZER_DOGGIBOT_DIR"] = str(home / "_internal" / "to_doggibot")
            self.assertNotIn("_internal", doggibot_root().parts)
        finally:
            os.environ.pop("RECOGNIZER_DOGGIBOT_DIR", None)

    def test_journal_keeps_the_last_200_lines(self):
        from sim.doggibot_snap import JOURNAL_KEEP, journal_text

        lines = ["строка %d" % i for i in range(250)]
        text = journal_text(lines)
        rows = text.strip().splitlines()
        self.assertEqual(len(rows), JOURNAL_KEEP)
        self.assertEqual(rows[0], "строка 50")
        self.assertEqual(rows[-1], "строка 249")


class DoggiBotSnapshotTests(unittest.TestCase):
    def test_snapshot_contains_the_bundle_and_drops_old_folders(self):
        import pygame
        import numpy as np

        from sim.doggibot_snap import reserve_folder, write_snapshot
        from sim.doggibot_snap import commit as commit_snap
        from sim.train_monitor import MonitorView, TrainMonitor, window_from_logical

        root = Path(tempfile.mkdtemp())
        config = root / "window.json"
        config.write_text('{"theme": "light", "w": 1920, "h": 1080}\n', encoding="utf-8")
        (root / "crash.log").write_text("boom\n" * 20, encoding="utf-8")
        os.environ["RECOGNIZER_UI_CONFIG"] = str(config)
        os.environ["RECOGNIZER_DOGGIBOT_DIR"] = str(root / "to_doggibot")
        pygame.init()
        try:
            mon = TrainMonitor("doggibot")
            mon._on_resize(1920, 1080)
            camera = np.zeros((48, 64, 3), dtype=np.uint8)
            camera[10:20, 10:20] = (20, 140, 40)
            lines = ["строка %d" % i for i in range(210)]
            view = MonitorView(
                learner="mb",
                learning_on=True,
                camera=camera,
                log_lines=lines,
                yolo_state="учит",
                n_pam=3,
                n_ppl1=1,
                total_pam=40,
                total_ppl1=9,
                hemi_l=0.25,
                hemi_r=0.75,
                eye_l_recognized=True,
                eye_r_recognized=False,
                eye_l_confidence=82.0,
                eye_r_confidence=4.0,
                drift=1.5,
                overlap=0.4,
                onboard=False,
            )
            mon.menu_open = True
            mon.draw(view)
            self.assertIn(("doggibot",), [action for _rect, action in mon._menu_hits])
            self.assertTrue(mon.doggibot_rect.w > 40)

            clicked = self._click(mon, mon.doggibot_rect.center)
            self.assertTrue(clicked.doggibot)
            self.assertFalse(clicked.quit)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F9))
            keyed = mon.pump()
            self.assertFalse(keyed.doggibot)

            mon.menu_open = True
            mon.draw(view)
            menu = next(rect for rect, action in mon._menu_hits if action[0] == "doggibot")
            opened = self._click(mon, menu.center)
            self.assertTrue(opened.doggibot)
            self.assertFalse(mon.menu_open)

            note = reserve_folder()
            lines.append(note.line)
            view.log_lines = lines
            mon.show_toast("снимок DoggiBOT")
            mon.draw(view)
            thread = commit_snap(
                mon,
                view,
                note,
                mode="sim",
                link_ok=False,
                teacher=_Teacher(),
                learning=True,
            )
            self.assertEqual(thread.name, "doggibot-snap")
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            source = inspect.getsource(write_snapshot)
            self.assertNotIn("import pygame", source)
            self.assertNotIn("pygame.", source)

            folder = note.folder
            self.assertTrue((folder / "screen.png").is_file())
            self.assertTrue((folder / "screen.png").read_bytes().startswith(b"\x89PNG"))
            self.assertEqual((folder / "screen.png").stat().st_size > 1000, True)
            saved = pygame.image.load(str(folder / "screen.png"))
            self.assertEqual(saved.get_size(), mon.screen.get_size())
            jpeg = (folder / "camera.jpg").read_bytes()
            self.assertTrue(jpeg.startswith(b"\xff\xd8"))
            journal = (folder / "journal.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(journal), 200)
            self.assertEqual(journal[-1], note.line)
            self.assertTrue(journal[-1].startswith("снимок для DoggiBOT:"))
            status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["mode"], "sim")
            self.assertTrue(status["build"])
            self.assertGreaterEqual(status["uptime_s"], 0)
            self.assertTrue(status["learning"])
            self.assertTrue(status["teacher_on"])
            self.assertEqual(status["teacher_health"], "учит")
            self.assertFalse(status["link_ok"])
            self.assertEqual(status["pam_session"], 3)
            self.assertEqual(status["ppl1_session"], 1)
            self.assertEqual(status["pam_total"], 40)
            self.assertEqual(status["ppl1_total"], 9)
            self.assertEqual(status["r_l"], 0.25)
            self.assertEqual(status["r_r"], 0.75)
            self.assertTrue(status["eye_l_recognized"])
            self.assertFalse(status["eye_r_recognized"])
            self.assertEqual(status["false_unpunished"], 4)
            self.assertEqual(status["drift_kc_mbon"], 1.5)
            self.assertEqual(status["overlap"], 0.4)
            self.assertEqual(status["theme"], "light")
            self.assertEqual(status["window"][0], 1920)
            self.assertEqual(status["logical"][0], mon.screen.get_width())
            self.assertIn("theme", (folder / "window.json").read_text(encoding="utf-8"))
            self.assertIn("boom", (folder / "crash.log").read_text(encoding="utf-8"))
            latest = (root / "to_doggibot" / "latest.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(latest, str(folder.resolve()))
            out = Path("/opt/cursor/artifacts")
            out.mkdir(parents=True, exist_ok=True)
            (out / "doggibot_status.json").write_text(
                json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            plain = MonitorView(log_lines=["без камеры"], learning_on=False)
            mon.draw(plain)
            empty_note = reserve_folder()
            started = commit_snap(
                mon,
                plain,
                empty_note,
                mode="robot",
                link_ok=True,
                teacher=None,
                learning=False,
            )
            started.join(timeout=5)
            self.assertFalse((empty_note.folder / "camera.jpg").exists())
            robot = json.loads((empty_note.folder / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(robot["mode"], "robot")
            self.assertTrue(robot["link_ok"])
            self.assertFalse(robot["teacher_on"])
            self.assertEqual(robot["false_unpunished"], 0)

            shot_root = root / "keep"
            for index in range(31):
                folder_i = shot_root / ("20200101_%06d" % index)
                write_snapshot(folder_i, shot_root, {"status.json": b"{}\n"})
            left = [path.name for path in shot_root.iterdir() if path.is_dir()]
            self.assertEqual(len(left), 30)
            self.assertNotIn("20200101_000000", left)
            self.assertIn("20200101_000030", left)

            box = []

            def off_thread():
                try:
                    from sim.doggibot_snap import logical_png

                    logical_png(mon.screen)
                except AssertionError as exc:
                    box.append(str(exc))

            worker = threading.Thread(target=off_thread, name="doggibot-off")
            worker.start()
            worker.join()
            self.assertTrue(box)
        finally:
            os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            os.environ.pop("RECOGNIZER_DOGGIBOT_DIR", None)
            import sim.doggibot_snap as snapmod
            from sim.ui_theme import apply_theme

            snapmod._BUILD = None
            apply_theme("light")
            pygame.quit()

    def _click(self, mon, logical):
        import pygame

        from sim.train_monitor import window_from_logical

        wx, wy = window_from_logical(logical, 1920, 1080)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(wx), int(wy)), button=1))
        return mon.pump()


class _Teacher:
    on = True
    skip_n = 4

    def status_text(self, learning: bool) -> str:
        return "учит" if learning else "выкл"
