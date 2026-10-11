"""The control button starts the command server, then the panel. A second click does not."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path


def _write_exe(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / name
    script.write_text("#!%s\n%s" % (sys.executable, body), encoding="utf-8")
    script.chmod(0o755)
    return script


def _fake_exe(folder: Path) -> Path:
    return _write_exe(
        folder,
        "go2_control.exe",
        "import os, time\nopen('started.txt', 'a', encoding='utf-8').write(os.getcwd() + '\\n')\ntime.sleep(30)\n",
    )


def _fake_server(folder: Path) -> Path:
    return _write_exe(
        folder,
        "go2_wr_server.exe",
        "import os, socket, time\n"
        "from pathlib import Path\n"
        "port = int(Path('port.txt').read_text(encoding='utf-8').strip())\n"
        "sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "sock.bind(('127.0.0.1', port))\n"
        "open('started.txt', 'a', encoding='utf-8').write(os.getcwd() + '\\n')\n"
        "time.sleep(30)\n",
    )


def _fake_idle(folder: Path, name: str) -> Path:
    return _write_exe(folder, name, "import time\ntime.sleep(30)\n")


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


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


def _stop(*procs) -> None:
    for proc in procs:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


class _Session:
    def __init__(self, panel: str = "", server: str = ""):
        self.go2_control_exe = panel
        self.go2_server_exe = server
        self.lines = []

    def _log(self, text):
        self.lines.append(text)


class _Inp:
    def __init__(self, clicked: bool = False):
        self.go2_control = clicked


class _Mon:
    def __init__(self):
        self.go2_open = False
        self.go2_server_open = False
        self.go2_missing = False


class Go2ControlTests(unittest.TestCase):
    def test_default_path_follows_the_launcher_folder(self):
        from sim.go2_control import (
            default_exe,
            default_server,
            panel_creationflags,
            resolve_exe,
            resolve_server,
            server_creationflags,
            windows_creationflags,
        )

        root = Path(tempfile.mkdtemp())
        home = root / "home"
        desktop = home / "Desktop" / "FlyWire-Go2"
        desktop.mkdir(parents=True)
        launchers = root / "beside_exe"
        chosen = default_exe(home, launchers)
        server = default_server(home, launchers)
        self.assertEqual(chosen, desktop / "go2_control" / "go2_control.exe")
        self.assertEqual(server, desktop / "go2_server" / "go2_wr_server.exe")
        for path in (chosen, server):
            self.assertNotIn("7dtra", str(path))
            self.assertFalse(str(path).upper().startswith("E:"))
        bare = root / "no_desktop"
        bare.mkdir()
        self.assertEqual(default_exe(bare, launchers), launchers / "go2_control" / "go2_control.exe")
        self.assertEqual(default_server(bare, launchers), launchers / "go2_server" / "go2_wr_server.exe")
        self.assertEqual(resolve_exe("/tmp/from-flag", "/tmp/from-json", bare), Path("/tmp/from-flag"))
        self.assertEqual(resolve_exe("", "/tmp/from-json", bare), Path("/tmp/from-json"))
        self.assertEqual(resolve_server("/tmp/server-flag", "/tmp/server-json", bare), Path("/tmp/server-flag"))
        self.assertEqual(resolve_server("", "/tmp/server-json", bare), Path("/tmp/server-json"))
        flags = windows_creationflags()
        hidden = server_creationflags()
        self.assertEqual(flags & 0x00000008, 0x00000008)
        self.assertEqual(flags & 0x00000200, 0x00000200)
        self.assertEqual(flags & 0x08000000, 0)
        self.assertEqual(hidden & 0x00000008, 0x00000008)
        self.assertEqual(hidden & 0x00000200, 0x00000200)
        self.assertEqual(hidden & 0x08000000, 0x08000000)
        panel_flags = panel_creationflags()
        self.assertEqual(panel_flags & 0x00000008, 0x00000008)
        self.assertEqual(panel_flags & 0x00000200, 0x00000200)
        self.assertEqual(panel_flags & 0x08000000, 0x08000000)
        source = Path(__file__).resolve().parent.joinpath("go2_control.py").read_text(encoding="utf-8")
        self.assertIn("DETACHED_PROCESS", source)
        self.assertIn("CREATE_NEW_PROCESS_GROUP", source)
        self.assertIn("CREATE_NO_WINDOW", source)
        self.assertNotIn(".terminate", source)
        self.assertNotIn(".kill(", source)
        live = Path(__file__).resolve().parent.joinpath("recognize_train_live.py").read_text(encoding="utf-8")
        sim = Path(__file__).resolve().parent.joinpath("recognize_train.py").read_text(encoding="utf-8")
        self.assertIn("robot_server_message", live)
        self.assertNotIn("robot_server_message", sim)
        entry = Path(__file__).resolve().parents[1].joinpath("tools", "recognize_trainer_entry.py").read_text(encoding="utf-8")
        self.assertEqual(entry.count('["--go2-server-exe", args.go2_server_exe]'), 2)
        self.assertEqual(entry.count('["--go2-control-exe", args.go2_control_exe]'), 2)

    def test_window_json_field_overrides_the_launcher_folder(self):
        from sim.go2_control import configured_exe, configured_server, remember_exe, remember_paths
        from sim.ui_settings import load_settings

        root = Path(tempfile.mkdtemp())
        cfg = root / "window.json"
        os.environ["RECOGNIZER_UI_CONFIG"] = str(cfg)
        try:
            cfg.write_text(
                json.dumps(
                    {
                        "go2_control_exe": str(root / "saved.exe"),
                        "go2_server_exe": str(root / "saved-server.exe"),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(configured_exe(""), root / "saved.exe")
            self.assertEqual(configured_exe(str(root / "flag.exe")), root / "flag.exe")
            self.assertEqual(configured_server(""), root / "saved-server.exe")
            self.assertEqual(configured_server(str(root / "flag-server.exe")), root / "flag-server.exe")
            remember_exe(str(root / "flag.exe"))
            remember_paths(server=str(root / "flag-server.exe"))
            saved = load_settings()
            self.assertEqual(saved["go2_control_exe"], str(root / "flag.exe"))
            self.assertEqual(saved["go2_server_exe"], str(root / "flag-server.exe"))
        finally:
            os.environ.pop("RECOGNIZER_UI_CONFIG", None)

    def test_fake_pair_starts_server_then_panel_and_a_second_click_does_not(self):
        from sim.go2_control import Go2Launch, drive_go2, raise_process

        root = Path(tempfile.mkdtemp())
        server = _fake_server(root / "go2_server")
        panel = _fake_exe(root / "go2_control")
        port = _free_udp_port()
        (server.parent / "port.txt").write_text(str(port), encoding="utf-8")
        launch = Go2Launch(panel, server, port=port, wait_s=5.0)
        try:
            message = launch.open_or_focus()
            self.assertIn("сервер команд открыт", message)
            self.assertIn("управление открыто", message)
            self.assertIsNotNone(launch.server_proc)
            self.assertIsNotNone(launch.panel_proc)
            self.assertIsNone(launch.server_proc.poll())
            self.assertIsNone(launch.panel_proc.poll())
            server_rows = _wait_lines(server.parent / "started.txt", 1)
            panel_rows = _wait_lines(panel.parent / "started.txt", 1)
            self.assertEqual([Path(row).resolve() for row in server_rows], [server.parent.resolve()])
            self.assertEqual([Path(row).resolve() for row in panel_rows], [panel.parent.resolve()])
            self.assertLessEqual((server.parent / "started.txt").stat().st_mtime, (panel.parent / "started.txt").stat().st_mtime)
            server_pid = int(launch.server_proc.pid)
            panel_pid = int(launch.panel_proc.pid)
            again = launch.open_or_focus()
            self.assertIn("уже", again)
            self.assertEqual(int(launch.server_proc.pid), server_pid)
            self.assertEqual(int(launch.panel_proc.pid), panel_pid)
            self.assertFalse(raise_process(panel_pid))
            time.sleep(0.3)
            self.assertEqual(len(_wait_lines(server.parent / "started.txt", 1)), 1)
            self.assertEqual(len(_wait_lines(panel.parent / "started.txt", 1)), 1)

            session = _Session()
            session._go2_panel = launch
            mon = _Mon()
            drive_go2(session, _Inp(False), mon)
            self.assertEqual(session.lines, [])
            self.assertTrue(mon.go2_open)
            self.assertTrue(mon.go2_server_open)
            self.assertFalse(mon.go2_missing)
        finally:
            _stop(launch.server_proc, launch.panel_proc)

    def test_busy_port_skips_the_server_and_still_starts_the_panel(self):
        from sim.go2_control import Go2Launch

        root = Path(tempfile.mkdtemp())
        server = _fake_server(root / "go2_server")
        panel = _fake_exe(root / "go2_control")
        port = _free_udp_port()
        holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        holder.bind(("127.0.0.1", port))
        launch = Go2Launch(panel, server, port=port, wait_s=5.0)
        try:
            message = launch.open_or_focus()
            self.assertIn("уже слушает", message)
            self.assertIn("управление открыто", message)
            self.assertIsNone(launch.server_proc)
            self.assertFalse((server.parent / "started.txt").is_file())
            rows = _wait_lines(panel.parent / "started.txt", 1)
            self.assertEqual([Path(row).resolve() for row in rows], [panel.parent.resolve()])
        finally:
            holder.close()
            _stop(launch.server_proc, launch.panel_proc)

    def test_missing_server_does_not_start_the_panel(self):
        from sim.go2_control import Go2Launch, drive_go2

        root = Path(tempfile.mkdtemp())
        panel = _fake_exe(root / "go2_control")
        port = _free_udp_port()
        launch = Go2Launch(panel, root / "absent" / "go2_wr_server.exe", port=port, wait_s=0.2)
        session = _Session(str(panel), str(root / "absent" / "go2_wr_server.exe"))
        session._go2_panel = launch
        mon = _Mon()
        drive_go2(session, _Inp(True), mon)
        self.assertTrue(session.lines)
        self.assertIn("нет файла сервера", session.lines[0])
        self.assertIsNone(launch.server_proc)
        self.assertIsNone(launch.panel_proc)
        self.assertFalse((panel.parent / "started.txt").is_file())
        self.assertFalse(mon.go2_open)

    def test_server_that_never_binds_does_not_start_the_panel(self):
        from sim.go2_control import Go2Launch

        root = Path(tempfile.mkdtemp())
        server = _fake_idle(root / "go2_server", "go2_wr_server.exe")
        panel = _fake_exe(root / "go2_control")
        port = _free_udp_port()
        launch = Go2Launch(panel, server, port=port, wait_s=0.2)
        try:
            message = launch.open_or_focus()
            self.assertIn("не занял", message)
            self.assertIsNone(launch.panel_proc)
            self.assertFalse((panel.parent / "started.txt").is_file())
            self.assertIsNotNone(launch.server_proc)
            server_pid = int(launch.server_proc.pid)
            again = launch.open_or_focus()
            self.assertIn("не занял", again)
            self.assertEqual(int(launch.server_proc.pid), server_pid)
            self.assertIsNone(launch.panel_proc)
        finally:
            _stop(launch.server_proc, launch.panel_proc)

    def test_missing_panel_is_reported_after_the_server_is_up(self):
        from sim.go2_control import Go2Launch

        root = Path(tempfile.mkdtemp())
        server = _fake_server(root / "go2_server")
        port = _free_udp_port()
        (server.parent / "port.txt").write_text(str(port), encoding="utf-8")
        missing = root / "absent" / "go2_control.exe"
        launch = Go2Launch(missing, server, port=port, wait_s=5.0)
        try:
            message = launch.open_or_focus()
            self.assertIn("сервер команд открыт", message)
            self.assertIn("нет файла", message)
            self.assertIn(str(missing), message)
            self.assertNotIn("нет файла сервера", message)
            self.assertIsNone(launch.panel_proc)
            rows = _wait_lines(server.parent / "started.txt", 1)
            self.assertEqual([Path(row).resolve() for row in rows], [server.parent.resolve()])
        finally:
            _stop(launch.server_proc, launch.panel_proc)

    def test_robot_journal_says_the_server_is_down_and_sim_does_not(self):
        from sim.go2_control import drive_go2, robot_server_message

        text, told = robot_server_message(False, False, "127.0.0.1", 5451)
        self.assertIn("молчит", text)
        self.assertIn("УПРАВЛЕНИЕ", text)
        self.assertIn("5451", text)
        self.assertTrue(told)
        again, still = robot_server_message(False, True, "127.0.0.1", 5451)
        self.assertEqual(again, "")
        self.assertTrue(still)
        cleared, reset = robot_server_message(True, True, "127.0.0.1", 5451)
        self.assertEqual(cleared, "")
        self.assertFalse(reset)

        root = Path(tempfile.mkdtemp())
        panel = _fake_exe(root / "go2_control")
        server = _fake_idle(root / "go2_server", "go2_wr_server.exe")
        session = _Session(str(panel), str(server))
        try:
            drive_go2(session, _Inp(False), _Mon())
            self.assertEqual(session.lines, [])
            self.assertIsNone(session._go2_panel.server_proc)
            self.assertIsNone(session._go2_panel.panel_proc)
        finally:
            _stop(session._go2_panel.server_proc, session._go2_panel.panel_proc)

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
        self.assertIn("5452", text)
        self.assertIn("5000", text)
        self.assertIn("нет файла", text)
        self.assertIn("trainer.log", text)
        self.assertIn("yolo_teacher.log", text)
        self.assertIn("go2_server.log", text)
        self.assertIn("crash.log", text)
        self.assertIn("не занял", text)
        self.assertIn("8090", text)
        self.assertIn("go2_wr_server", text)
        self.assertIn("--go2-server-exe", text)
        fd, cfg = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.environ["RECOGNIZER_UI_CONFIG"] = cfg
        pygame.init()
        try:
            mon = TrainMonitor("go2")
            self.assertFalse(mon.go2_server_open)
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
            mon.go2_server_open = True
            mon.go2_open = True
            mon.draw(view)
            self.assertTrue(_lamp_peach(mon, mon.go2_server_lamp[0]))
            self.assertTrue(_lamp_peach(mon, mon.go2_panel_lamp[0]))
            self.assertLess(mon.go2_server_lamp[0], mon.go2_panel_lamp[0])
            _save_bar(mon, out / "go2_control_button.png")
            mon.go2_open = False
            mon.go2_server_open = False
            mon.go2_missing = True
            mon.draw(view)
            self.assertFalse(_lamp_peach(mon, mon.go2_server_lamp[0]))
            self.assertFalse(_lamp_peach(mon, mon.go2_panel_lamp[0]))
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


def _lamp_peach(mon, x: int) -> bool:
    color = mon.screen.get_at((int(x), int(mon.go2_rect.centery)))
    return color.r > color.b + 40


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
