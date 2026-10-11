"""Launch the Go2 command server and the control panel. The trainer does not own them.

A click starts ``go2_server/go2_wr_server.exe`` when UDP 5451 is free, waits
until that port is taken, then starts ``go2_control/go2_control.exe``.
A second click does not start another copy of either. Closing the trainer
does not stop them: the panel may be driving a different dog, and the robot
window sends its own commands to the same server.

The simulator does not start the server on its own. Only this click does.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .doggibot_snap import desktop_flywire, exe_launchers

# Windows process flags. Present here so a Linux test can still see them.
# DETACHED_PROCESS keeps the child off the trainer console.
# CREATE_NEW_PROCESS_GROUP lets the trainer exit without delivering a break.
# CREATE_NO_WINDOW hides a console. The panel still opens its own window.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000
SERVER_PORT = 5451
SERVER_WAIT_S = 5.0


def windows_creationflags() -> int:
    detached = getattr(subprocess, "DETACHED_PROCESS", _DETACHED_PROCESS)
    group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", _CREATE_NEW_PROCESS_GROUP)
    return int(detached) | int(group)


def server_creationflags() -> int:
    hidden = getattr(subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW)
    return windows_creationflags() | int(hidden)


def panel_creationflags() -> int:
    """Same detach flags as the server, and no console. The GUI window stays."""
    hidden = getattr(subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW)
    return windows_creationflags() | int(hidden)


def panel_argv(path: Path) -> list:
    """Launch arguments. The panel has no theme switch of its own.

    ``--theme`` is the trainer theme at the moment of the click. The panel
    then rereads ``theme`` in the trainer's window.json about once a second.
    A missing BOM and a half-written file are ignored, and the last good
    theme stays. Those sources are not in this repository.
    """
    from .ui_theme import theme_name

    name = theme_name()
    if name not in ("light", "dark"):
        name = "light"
    return [str(Path(path)), "--theme", name]


def panel_detach_kwargs() -> dict:
    """The control panel gets its own group. A console is not allocated."""
    if os.name == "nt":
        return {"creationflags": panel_creationflags()}
    return {"start_new_session": True}


def server_detach_kwargs() -> dict:
    """The console server gets no window. Its own ``logs`` folder is the log."""
    if os.name == "nt":
        return {"creationflags": server_creationflags()}
    return {"start_new_session": True}


def _launcher_root(home: Path | None, launchers: Path | None) -> Path | None:
    desktop = desktop_flywire(home)
    if desktop is not None:
        return desktop
    if launchers is not None:
        return Path(launchers)
    return exe_launchers()


def default_exe(home: Path | None = None, launchers: Path | None = None) -> Path:
    """``<launcher folder>/go2_control/go2_control.exe``. No drive letter is pinned."""
    folder = _launcher_root(home, launchers)
    if folder is not None:
        return Path(folder) / "go2_control" / "go2_control.exe"
    root = Path.home() if home is None else Path(home)
    return root / "Desktop" / "FlyWire-Go2" / "go2_control" / "go2_control.exe"


def default_server(home: Path | None = None, launchers: Path | None = None) -> Path:
    """``<launcher folder>/go2_server/go2_wr_server.exe``. No drive letter is pinned."""
    folder = _launcher_root(home, launchers)
    if folder is not None:
        return Path(folder) / "go2_server" / "go2_wr_server.exe"
    root = Path.home() if home is None else Path(home)
    return root / "Desktop" / "FlyWire-Go2" / "go2_server" / "go2_wr_server.exe"


def resolve_exe(cli: str = "", saved: str = "", home: Path | None = None) -> Path:
    """Flag, then the ``window.json`` field, then the launcher folder."""
    chosen = str(cli or "").strip() or str(saved or "").strip()
    if chosen:
        return Path(chosen)
    return default_exe(home)


def resolve_server(cli: str = "", saved: str = "", home: Path | None = None) -> Path:
    chosen = str(cli or "").strip() or str(saved or "").strip()
    if chosen:
        return Path(chosen)
    return default_server(home)


def _saved(key: str) -> str:
    from .ui_settings import load_settings, persist_enabled

    if not persist_enabled():
        return ""
    return str(load_settings().get(key) or "")


def configured_exe(cli: str = "") -> Path:
    return resolve_exe(cli, _saved("go2_control_exe"))


def configured_server(cli: str = "") -> Path:
    return resolve_server(cli, _saved("go2_server_exe"))


def remember_paths(panel: str = "", server: str = "") -> None:
    """Store flag overrides. An empty flag does not erase a saved path."""
    panel_text = str(panel or "").strip()
    server_text = str(server or "").strip()
    if not panel_text and not server_text:
        return
    from .ui_settings import load_settings, persist_enabled, save_settings

    if not persist_enabled():
        return
    data = load_settings()
    changed = False
    if panel_text and str(data.get("go2_control_exe") or "") != panel_text:
        data["go2_control_exe"] = panel_text
        changed = True
    if server_text and str(data.get("go2_server_exe") or "") != server_text:
        data["go2_server_exe"] = server_text
        changed = True
    if changed:
        save_settings(data)


def remember_exe(path: str) -> None:
    remember_paths(panel=path)


def add_go2_control_arg(parser) -> None:
    parser.add_argument(
        "--go2-control-exe",
        default="",
        help="Path to go2_control.exe. Empty uses window.json, then the launcher folder.",
    )
    parser.add_argument(
        "--go2-server-exe",
        default="",
        help="Path to go2_wr_server.exe. Empty uses window.json, then the launcher folder.",
    )


def apply_go2_control_arg(session, args) -> None:
    panel = str(getattr(args, "go2_control_exe", "") or "").strip()
    server = str(getattr(args, "go2_server_exe", "") or "").strip()
    session.go2_control_exe = panel
    session.go2_server_exe = server
    remember_paths(panel, server)


def robot_server_message(listening: bool, told: bool, host: str = "127.0.0.1", port: int = SERVER_PORT) -> tuple:
    """One journal line when the robot window has nobody on its UDP port.

    The simulator does not call this. Coming back up clears the latch so a
    later drop can be said again.
    """
    if listening:
        return "", False
    if told:
        return "", True
    return (
        "сервер команд не запущен: UDP %s:%s молчит, команды тренера туда не доходят. "
        "Кнопка «УПРАВЛЕНИЕ» поднимает go2_wr_server."
        % (host, int(port)),
        True,
    )


def raise_process(pid: int) -> bool:
    """Bring a Windows window of this pid forward. Other systems do nothing."""
    if os.name != "nt" or int(pid) <= 0:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        found = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _visit(hwnd, _lparam):
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if int(owner.value) == int(pid) and user32.IsWindowVisible(hwnd):
                found.append(hwnd)
            return True

        user32.EnumWindows(_visit, 0)
        if not found:
            return False
        user32.ShowWindow(found[0], 9)
        user32.SetForegroundWindow(found[0])
        return True
    except Exception:
        return False


class Go2Launch:
    """The command server and the control panel. Nothing here calls terminate or kill."""

    def __init__(self, panel: Path, server: Path, port: int = SERVER_PORT, wait_s: float = SERVER_WAIT_S):
        self.panel_path = Path(panel)
        self.server_path = Path(server)
        self.port = int(port)
        self.wait_s = float(wait_s)
        self.panel_proc = None
        self.server_proc = None
        self._probe_at = 0.0
        self._probe_value = None

    @property
    def found(self) -> bool:
        return self.panel_path.is_file()

    def panel_up(self) -> bool:
        return self.panel_proc is not None and self.panel_proc.poll() is None

    def _port_open(self) -> bool:
        from .onboard_link import udp_bridge_listening

        return bool(udp_bridge_listening("127.0.0.1", self.port))

    def server_up(self) -> bool:
        """Our server process, or anything already holding UDP 5451."""
        if self.server_proc is not None and self.server_proc.poll() is None:
            return True
        now = time.monotonic()
        if self._probe_value is not None and now - self._probe_at < 0.5:
            return bool(self._probe_value)
        self._probe_at = now
        self._probe_value = self._port_open()
        return bool(self._probe_value)

    def _spawn(self, path: Path, kind: str):
        from .quiet_proc import open_log

        kwargs = server_detach_kwargs() if kind == "server" else panel_detach_kwargs()
        handle = open_log("go2_server.log" if kind == "server" else "go2_control.log")
        command = panel_argv(path) if kind == "panel" else [str(path)]
        try:
            return subprocess.Popen(
                command,
                cwd=str(path.parent),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        finally:
            # The child already has its own handle. Closing this copy is not a kill.
            if handle is not subprocess.DEVNULL:
                try:
                    handle.close()
                except OSError:
                    pass

    def _wait_port(self) -> bool:
        deadline = time.monotonic() + self.wait_s
        while True:
            if self._port_open():
                self._probe_value = True
                self._probe_at = time.monotonic()
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def open_or_focus(self) -> str:
        notes = []
        if self._port_open():
            notes.append("сервер уже слушает UDP %s" % self.port)
        elif self.server_proc is not None and self.server_proc.poll() is None:
            if not self._wait_port():
                waited = int(self.wait_s) if self.wait_s >= 1 else self.wait_s
                return "управление: сервер не занял UDP %s за %s с" % (self.port, waited)
            notes.append("сервер команд открыт")
        elif not self.server_path.is_file():
            return "управление: нет файла сервера %s" % self.server_path
        else:
            try:
                self.server_proc = self._spawn(self.server_path, "server")
            except OSError as exc:
                self.server_proc = None
                return "управление: сервер не запустился %s (%s)" % (self.server_path, exc)
            if not self._wait_port():
                waited = int(self.wait_s) if self.wait_s >= 1 else self.wait_s
                return "управление: сервер не занял UDP %s за %s с" % (self.port, waited)
            notes.append("сервер команд открыт")
        if not self.panel_path.is_file():
            notes.append("управление: нет файла %s" % self.panel_path)
            return ". ".join(notes)
        if self.panel_up():
            raise_process(int(self.panel_proc.pid))
            notes.append("управление уже открыто")
            return ". ".join(notes)
        try:
            self.panel_proc = self._spawn(self.panel_path, "panel")
        except OSError as exc:
            self.panel_proc = None
            notes.append("управление: не запустилось %s (%s)" % (self.panel_path, exc))
            return ". ".join(notes)
        notes.append("управление открыто")
        return ". ".join(notes)


def drive_go2(session, inp, mon=None) -> None:
    """Start the pair on a click. A later frame only refreshes the lamps.

    Sim mode does not call this until the operator clicks. It does not start
    the server by itself.
    """
    link = getattr(session, "_go2_panel", None)
    if link is None:
        link = Go2Launch(
            configured_exe(getattr(session, "go2_control_exe", "")),
            configured_server(getattr(session, "go2_server_exe", "")),
        )
        session._go2_panel = link
    if getattr(inp, "go2_control", False):
        session._log(link.open_or_focus())
    if mon is not None:
        mon.go2_open = bool(link.panel_up())
        mon.go2_server_open = bool(link.server_up())
        mon.go2_missing = not link.found
