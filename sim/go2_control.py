"""Launch the external Go2 control panel. The trainer does not own its life.

The exe lives next to the launchers, in ``go2_control/go2_control.exe``.
A click starts one process with that folder as its cwd. A second click
does not start another copy. Closing the trainer does not stop the panel:
it may be driving a different dog.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .doggibot_snap import desktop_flywire, exe_launchers

# Windows process flags. Present here so a Linux test can still see them.
# DETACHED_PROCESS keeps the panel off the trainer console.
# CREATE_NEW_PROCESS_GROUP lets the trainer exit without delivering it a break.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200


def windows_creationflags() -> int:
    detached = getattr(subprocess, "DETACHED_PROCESS", _DETACHED_PROCESS)
    group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", _CREATE_NEW_PROCESS_GROUP)
    return int(detached) | int(group)


def detach_kwargs() -> dict:
    """Flags that keep the panel in its own process group."""
    if os.name == "nt":
        return {"creationflags": windows_creationflags()}
    return {"start_new_session": True}


def default_exe(home: Path | None = None, launchers: Path | None = None) -> Path:
    """``<launcher folder>/go2_control/go2_control.exe``. No drive letter is pinned."""
    desktop = desktop_flywire(home)
    if desktop is not None:
        return desktop / "go2_control" / "go2_control.exe"
    folder = exe_launchers() if launchers is None else Path(launchers)
    if folder is not None:
        return Path(folder) / "go2_control" / "go2_control.exe"
    root = Path.home() if home is None else Path(home)
    return root / "Desktop" / "FlyWire-Go2" / "go2_control" / "go2_control.exe"


def resolve_exe(cli: str = "", saved: str = "", home: Path | None = None) -> Path:
    """Flag, then the ``window.json`` field, then the launcher folder."""
    chosen = str(cli or "").strip() or str(saved or "").strip()
    if chosen:
        return Path(chosen)
    return default_exe(home)


def configured_exe(cli: str = "") -> Path:
    """Resolve with ``window.json`` when the trainer is allowed to read it."""
    from .ui_settings import load_settings, persist_enabled

    saved = ""
    if persist_enabled():
        saved = str(load_settings().get("go2_control_exe") or "")
    return resolve_exe(cli, saved)


def remember_exe(path: str) -> None:
    """Store a flag override in ``window.json``. An empty flag does not erase it."""
    text = str(path or "").strip()
    if not text:
        return
    from .ui_settings import load_settings, persist_enabled, save_settings

    if not persist_enabled():
        return
    data = load_settings()
    if str(data.get("go2_control_exe") or "") == text:
        return
    data["go2_control_exe"] = text
    save_settings(data)


def add_go2_control_arg(parser) -> None:
    parser.add_argument(
        "--go2-control-exe",
        default="",
        help="Path to go2_control.exe. Empty uses window.json, then the launcher folder.",
    )


def apply_go2_control_arg(session, args) -> None:
    text = str(getattr(args, "go2_control_exe", "") or "").strip()
    session.go2_control_exe = text
    if text:
        remember_exe(text)


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


class Go2Panel:
    """One external control process. Nothing here calls terminate or kill."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.proc = None

    @property
    def found(self) -> bool:
        return self.path.is_file()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def open_or_focus(self) -> str:
        if not self.found:
            return "управление: нет файла %s" % self.path
        if self.alive:
            raise_process(int(self.proc.pid))
            return "управление уже открыто"
        try:
            self.proc = subprocess.Popen(
                [str(self.path)],
                cwd=str(self.path.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **detach_kwargs(),
            )
        except OSError as exc:
            self.proc = None
            return "управление: не запустилось %s (%s)" % (self.path, exc)
        return "управление открыто"


def drive_go2(session, inp, mon=None) -> None:
    """Start the panel on a click. A later frame only refreshes the lamp."""
    link = getattr(session, "_go2_panel", None)
    if link is None:
        link = Go2Panel(configured_exe(getattr(session, "go2_control_exe", "")))
        session._go2_panel = link
    if getattr(inp, "go2_control", False):
        session._log(link.open_or_focus())
    if mon is not None:
        mon.go2_open = bool(link.alive)
        mon.go2_missing = not link.found
