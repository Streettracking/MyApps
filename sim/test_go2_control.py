"""The control panel is a separate process. A second click does not spawn another."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


def _fake_exe(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "go2_control.exe"
    script.write_text(
        "#!%s\nimport os, time\nopen('started.txt', 'a', encoding='utf-8').write(os.getcwd() + '\\n')\ntime.sleep(30)\n"
        % sys.executable,
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _wait_lines(path: Path, n: int, seconds: float = 2.0) -> list:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.is_file():
            rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            if len(rows) >= n:
                return rows
        time.sleep(0.05)
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


class Go2ControlTests(unittest.TestCase):
    def test_default_path_follows_the_launcher_folder(self):
        from sim.go2_control import default_exe, resolve_exe, windows_creationflags

        root = Path(tempfile.mkdtemp())
        home = root / "home"
        desktop = home / "Desktop" / "FlyWire-Go2"
        desktop.mkdir(parents=True)
        launchers = root / "beside_exe"
        chosen = default_exe(home, launchers)
        self.assertEqual(chosen, desktop / "go2_control" / "go2_control.exe")
        self.assertNotIn("7dtra", str(chosen))
        self.assertFalse(str(chosen).upper().startswith("E:"))
        bare = root / "no_desktop"
        bare.mkdir()
        fallback = default_exe(bare, launchers)
        self.assertEqual(fallback, launchers / "go2_control" / "go2_control.exe")
        self.assertEqual(resolve_exe("/tmp/from-flag", "/tmp/from-json", bare), Path("/tmp/from-flag"))
        self.assertEqual(resolve_exe("", "/tmp/from-json", bare), Path("/tmp/from-json"))
        self.assertEqual(resolve_exe("", "", home), desktop / "go2_control" / "go2_control.exe")
        flags = windows_creationflags()
        self.assertEqual(flags & 0x00000008, 0x00000008)
        self.assertEqual(flags & 0x00000200, 0x00000200)
        source = Path(__file__).resolve().parent.joinpath("go2_control.py").read_text(encoding="utf-8")
        self.assertIn("DETACHED_PROCESS", source)
        self.assertIn("CREATE_NEW_PROCESS_GROUP", source)
        self.assertNotIn(".terminate", source)
        self.assertNotIn(".kill(", source)

    def test_window_json_field_overrides_the_launcher_folder(self):
        from sim.go2_control import configured_exe, remember_exe
        from sim.ui_settings import load_settings

        root = Path(tempfile.mkdtemp())
        cfg = root / "window.json"
        os.environ["RECOGNIZER_UI_CONFIG"] = str(cfg)
        try:
            cfg.write_text(json.dumps({"go2_control_exe": str(root / "saved.exe")}), encoding="utf-8")
            self.assertEqual(configured_exe(""), root / "saved.exe")
            self.assertEqual(configured_exe(str(root / "flag.exe")), root / "flag.exe")
            remember_exe(str(root / "flag.exe"))
            self.assertEqual(load_settings()["go2_control_exe"], str(root / "flag.exe"))
        finally:
            os.environ.pop("RECOGNIZER_UI_CONFIG", None)

    def test_fake_exe_starts_once_and_a_missing_file_is_reported(self):
        from sim.go2_control import Go2Panel, drive_go2, raise_process

        root = Path(tempfile.mkdtemp())
        script = _fake_exe(root / "go2_control")
        panel = Go2Panel(script)
        proc = None
        try:
            self.assertTrue(panel.found)
            self.assertEqual(panel.open_or_focus(), "управление открыто")
            proc = panel.proc
            self.assertIsNotNone(proc)
            self.assertIsNone(proc.poll())
            rows = _wait_lines(script.parent / "started.txt", 1)
            self.assertEqual([Path(row).resolve() for row in rows], [script.parent.resolve()])
            pid = int(proc.pid)
            self.assertEqual(panel.open_or_focus(), "управление уже открыто")
            self.assertEqual(int(panel.proc.pid), pid)
            self.assertFalse(raise_process(pid))
            time.sleep(0.3)
            again = _wait_lines(script.parent / "started.txt", 1)
            self.assertEqual([Path(row).resolve() for row in again], [script.parent.resolve()])
            self.assertIsNone(proc.poll())

            class _Session:
                go2_control_exe = ""
                lines = []

                def _log(self, text):
                    self.lines.append(text)

            class _Inp:
                go2_control = False

            class _Mon:
                go2_open = False
                go2_missing = False

            session = _Session()
            session._go2_panel = panel
            mon = _Mon()
            drive_go2(session, _Inp(), mon)
            self.assertTrue(mon.go2_open)
            self.assertFalse(mon.go2_missing)
            self.assertEqual(session.lines, [])

            missing = Go2Panel(root / "absent" / "go2_control.exe")
            message = missing.open_or_focus()
            self.assertIn("нет файла", message)
            self.assertIn(str(missing.path), message)
            self.assertIsNone(missing.proc)
            session2 = _Session()
            session2.go2_control_exe = str(missing.path)
            session2.lines = []
            clicked = _Inp()
            clicked.go2_control = True
            mon2 = _Mon()
            os.environ["RECOGNIZER_UI_CONFIG"] = str(root / "unused.json")
            try:
                drive_go2(session2, clicked, mon2)
            finally:
                os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            self.assertTrue(mon2.go2_missing)
            self.assertFalse(mon2.go2_open)
            self.assertTrue(session2.lines)
            self.assertIn("нет файла", session2.lines[0])
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

    def test_button_and_menu_request_the_panel_without_a_new_key(self):
        import pygame

        from sim.recognize_train import SIM_KEYS
        from sim.recognize_train_live import LIVE_KEYS
        from sim.train_monitor import MonitorView, TrainMonitor, window_from_logical
        from sim.ui_help import page

        self.assertNotIn("УПРАВЛЕНИЕ", SIM_KEYS)
        self.assertNotIn("УПРАВЛЕНИЕ", LIVE_KEYS)
        _title, blocks = page("bar")
        text = "\n".join(line for _kind, line in blocks)
        self.assertIn("УПРАВЛЕНИЕ", text)
        self.assertIn("5451", text)
        self.assertIn("нет файла", text)
        self.assertIn("8090", text)
        fd, cfg = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.environ["RECOGNIZER_UI_CONFIG"] = cfg
        pygame.init()
        try:
            mon = TrainMonitor("go2")
            mon._on_resize(1920, 1080)
            view = MonitorView(title="recognition training", learner="mb", learning_on=True)
            mon.draw(view)
            self.assertLess(mon.help_bar_rect.right, mon.go2_rect.x)
            self.assertLess(mon.go2_rect.right, mon.brain_rect.x)
            self.assertFalse(mon.go2_rect.colliderect(mon.brain_rect))
            wx, wy = window_from_logical(mon.go2_rect.center, 1920, 1080)
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(wx), int(wy)), button=1))
            clicked = mon.pump()
            self.assertTrue(clicked.go2_control)
            self.assertFalse(clicked.brain_toggle)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q, mod=0, unicode="q", scancode=0))
            keyed = mon.pump()
            self.assertFalse(keyed.go2_control)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0, unicode="j", scancode=0))
            brain = mon.pump()
            self.assertTrue(brain.brain_toggle)
            self.assertFalse(brain.go2_control)
            mon.menu_open = True
            mon.draw(view)
            actions = [action for _rect, action in mon._menu_hits]
            self.assertIn(("go2",), actions)
            self.assertNotIn(("toggle", "go2"), actions)
            rect = next(hit for hit, action in mon._menu_hits if action == ("go2",))
            wx, wy = window_from_logical(rect.center, 1920, 1080)
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(wx), int(wy)), button=1))
            menu = mon.pump()
            self.assertTrue(menu.go2_control)
            self.assertFalse(menu.doggibot)
            self.assertFalse(mon.menu_open)
            out = Path("/opt/cursor/artifacts")
            out.mkdir(parents=True, exist_ok=True)
            mon.menu_open = False
            mon.go2_missing = False
            mon.go2_open = True
            mon.draw(view)
            _save_bar(mon, out / "go2_control_button.png")
            mon.go2_open = False
            mon.go2_missing = True
            mon.draw(view)
            _save_bar(mon, out / "go2_control_missing.png")
        finally:
            os.environ.pop("RECOGNIZER_UI_CONFIG", None)
            from sim.ui_theme import apply_theme

            apply_theme("light")
            pygame.quit()
            try:
                os.remove(cfg)
            except OSError:
                pass


def _save_bar(mon, path: Path) -> None:
    import pygame

    crop = pygame.Rect(
        mon.help_bar_rect.x - 8,
        0,
        mon.record_rect.right - mon.help_bar_rect.x + 16,
        36,
    )
    crop = crop.clamp(mon.screen.get_rect())
    shot = pygame.Surface((crop.w, crop.h))
    shot.blit(mon.screen, (0, 0), area=crop)
    pygame.image.save(shot, str(path))
