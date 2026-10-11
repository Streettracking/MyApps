"""Local DoggiBOT snapshot. No network.

The main thread copies pixels and text into bytes. A background thread writes
those bytes under the launcher folder, outside ``_internal``, so a rebuild
does not delete them. ``RECOGNIZER_DOGGIBOT_DIR`` pins the folder in tests.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

from .ui_settings import crash_log_path, default_settings, settings_path

KEEP = 30
JOURNAL_KEEP = 200
CRASH_TAIL = 65536

_RESERVED: set[Path] = set()
_WRITE_LOCK = threading.Lock()
_BUILD: str | None = None


class SnapNote:
    def __init__(self, root: Path, folder: Path):
        self.root = root
        self.folder = folder
        self.line = "снимок для DoggiBOT: %s" % folder


def build_label() -> str:
    """A git hash, a frozen exe stamp, or ``dev``."""
    global _BUILD
    if _BUILD is not None:
        return _BUILD
    override = os.environ.get("RECOGNIZER_BUILD", "").strip()
    if override:
        _BUILD = override
        return _BUILD
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable)
        try:
            stamp = int(exe.stat().st_mtime)
        except OSError:
            stamp = 0
        _BUILD = "frozen:%s:%s" % (exe.name, stamp)
        return _BUILD
    _BUILD = _git_head() or "dev"
    return _BUILD


def _git_head() -> str:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.decode("utf-8", "replace").strip()


def _has_internal(path: Path) -> bool:
    return any(part.lower() == "_internal" for part in path.parts)


def desktop_flywire(home: Path | None = None) -> Path | None:
    """``Desktop/FlyWire-Go2`` when that launcher folder exists."""
    root = Path.home() if home is None else Path(home)
    names = (
        root / "Desktop" / "FlyWire-Go2",
        root / "OneDrive" / "Desktop" / "FlyWire-Go2",
        root / "Рабочий стол" / "FlyWire-Go2",
    )
    profile = os.environ.get("USERPROFILE")
    if profile and home is None:
        names = names + (Path(profile) / "Desktop" / "FlyWire-Go2",)
    for path in names:
        if path.is_dir() and not _has_internal(path):
            return path
    return None


def exe_launchers(exe: Path | None = None, frozen: bool | None = None) -> Path | None:
    """Folder of a frozen exe, stepping out of ``_internal``."""
    is_frozen = getattr(sys, "frozen", False) if frozen is None else bool(frozen)
    if not is_frozen:
        return None
    raw = Path(sys.executable) if exe is None else Path(exe)
    folder = raw.resolve().parent
    if folder.name.lower() == "_internal":
        folder = folder.parent
    if folder.name.lower() == "_internal" or _has_internal(folder):
        return None
    if not folder.is_dir():
        return None
    return folder


def appdata_root() -> Path:
    """Same directory as ``window.json``, never inside ``_internal``."""
    parent = settings_path().parent
    if _has_internal(parent):
        parent = Path.home() / "recognize_trainer"
    return parent


def doggibot_root() -> Path:
    """``to_doggibot`` beside the launchers, else under the trainer config dir."""
    override = os.environ.get("RECOGNIZER_DOGGIBOT_DIR", "").strip()
    if override:
        path = Path(override)
        if not _has_internal(path):
            return path
    desktop = desktop_flywire()
    if desktop is not None:
        return desktop / "to_doggibot"
    exe = exe_launchers()
    if exe is not None:
        return exe / "to_doggibot"
    return appdata_root() / "to_doggibot"


def reserve_folder(root: Path | None = None, when: datetime | None = None) -> SnapNote:
    """Pick a timestamp folder. Does not write the snapshot."""
    base = doggibot_root() if root is None else Path(root)
    moment = datetime.now() if when is None else when
    stamp = moment.strftime("%Y%m%d_%H%M%S")
    folder = base / stamp
    n = 2
    while folder in _RESERVED or folder.exists():
        folder = base / ("%s_%d" % (stamp, n))
        n += 1
    _RESERVED.add(folder)
    return SnapNote(base, folder)


def journal_text(lines) -> str:
    rows = [str(line) for line in list(lines or [])][-JOURNAL_KEEP:]
    if not rows:
        return ""
    return "\n".join(rows) + "\n"


def status_dict(
    view,
    *,
    mode: str,
    link_ok: bool,
    teacher_on: bool,
    teacher_health: str,
    false_n: int,
    uptime_s: float,
    fps: float,
    window,
    logical,
    theme: str,
) -> dict:
    return {
        "mode": str(mode),
        "build": build_label(),
        "uptime_s": round(float(uptime_s), 3),
        "learning": bool(getattr(view, "learning_on", False)),
        "teacher_on": bool(teacher_on),
        "teacher_health": str(teacher_health or ""),
        "link_ok": bool(link_ok),
        "pam_session": int(getattr(view, "n_pam", 0) or 0),
        "ppl1_session": int(getattr(view, "n_ppl1", 0) or 0),
        "pam_total": int(getattr(view, "total_pam", 0) or 0),
        "ppl1_total": int(getattr(view, "total_ppl1", 0) or 0),
        "r_l": _opt(getattr(view, "hemi_l", None)),
        "r_r": _opt(getattr(view, "hemi_r", None)),
        "eye_l_recognized": bool(getattr(view, "eye_l_recognized", False)),
        "eye_r_recognized": bool(getattr(view, "eye_r_recognized", False)),
        "eye_l_confidence": float(getattr(view, "eye_l_confidence", 0.0) or 0.0),
        "eye_r_confidence": float(getattr(view, "eye_r_confidence", 0.0) or 0.0),
        "false_unpunished": int(false_n),
        "drift_kc_mbon": float(getattr(view, "drift", 0.0) or 0.0),
        "overlap": float(getattr(view, "overlap", 0.0) or 0.0),
        "fps": round(float(fps), 2),
        "window": [int(window[0]), int(window[1])],
        "logical": [int(logical[0]), int(logical[1])],
        "theme": str(theme),
    }


def _opt(value):
    if value is None:
        return None
    return float(value)


def teacher_bits(teacher, learning: bool, yolo_state: str) -> tuple:
    """On/off, health text, and unpunished false recognitions. No HTTP."""
    if teacher is None:
        return False, str(yolo_state or ""), 0
    on = bool(getattr(teacher, "on", False))
    health = yolo_state
    status = getattr(teacher, "status_text", None)
    if status is not None:
        health = status(bool(learning))
    false_n = int(getattr(teacher, "skip_n", 0) or 0)
    return on, str(health or ""), false_n


def logical_png(screen) -> bytes:
    """PNG of the logical canvas. pygame stays on the main thread."""
    import io

    import pygame

    from .sdl_thread import require_main_thread

    require_main_thread()
    buf = io.BytesIO()
    pygame.image.save(screen, buf, "png")
    return bytes(buf.getvalue())


def camera_jpeg(camera) -> bytes | None:
    if camera is None:
        return None
    from .frame_record import encode_camera_jpeg

    try:
        return bytes(encode_camera_jpeg(camera))
    except (OSError, ValueError):
        return None


def window_copy() -> bytes:
    path = settings_path()
    try:
        return bytes(path.read_bytes())
    except OSError:
        return json.dumps(default_settings(), ensure_ascii=False, indent=2).encode("utf-8")


def _tail_bytes(path: Path) -> bytes | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    if len(data) > CRASH_TAIL:
        data = data[-CRASH_TAIL:]
    return bytes(data)


def crash_tail() -> bytes | None:
    return _tail_bytes(crash_log_path())


def child_log_tails() -> dict:
    """Tails of the teacher and the command server. Missing files are omitted."""
    files = {}
    for name in ("yolo_teacher.log", "go2_server.log"):
        data = _tail_bytes(appdata_root() / name)
        if data:
            files[name] = data
    return files


def collect_files(mon, view, *, mode: str, link_ok: bool, teacher, learning: bool) -> dict:
    """Copy every snapshot payload on the main thread."""
    from .ui_theme import theme_name

    on, health, false_n = teacher_bits(teacher, learning, getattr(view, "yolo_state", ""))
    window = (0, 0)
    if getattr(mon, "window", None) is not None:
        window = mon.window.get_size()
    elif getattr(mon, "_window_size", None):
        window = mon._window_size
    logical = mon.screen.get_size()
    payload = status_dict(
        view,
        mode=mode,
        link_ok=link_ok,
        teacher_on=on,
        teacher_health=health,
        false_n=false_n,
        uptime_s=mon.uptime_s(),
        fps=float(mon.clock.get_fps()) if getattr(mon, "clock", None) is not None else 0.0,
        window=window,
        logical=logical,
        theme=theme_name(),
    )
    files = {
        "screen.png": logical_png(mon.screen),
        "journal.txt": journal_text(getattr(view, "log_lines", ())).encode("utf-8"),
        "status.json": json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        "window.json": window_copy(),
    }
    jpeg = camera_jpeg(getattr(view, "camera", None))
    if jpeg:
        files["camera.jpg"] = jpeg
    tail = crash_tail()
    if tail:
        files["crash.log"] = tail
    files.update(child_log_tails())
    return files


def write_snapshot(folder: Path, root: Path, files: dict) -> None:
    """Write an already-copied bundle. Pixels were encoded before this runs."""
    with _WRITE_LOCK:
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (folder / name).write_bytes(data)
        (root / "latest.txt").write_text(str(folder.resolve()) + "\n", encoding="utf-8")
        _prune(root, KEEP)


def start_write(folder: Path, root: Path, files: dict) -> threading.Thread:
    copied = {name: bytes(data) for name, data in files.items()}
    thread = threading.Thread(
        target=write_snapshot,
        args=(Path(folder), Path(root), copied),
        name="doggibot-snap",
        daemon=True,
    )
    thread.start()
    return thread


def commit(mon, view, note: SnapNote, *, mode: str, link_ok: bool, teacher, learning: bool) -> threading.Thread:
    """Capture on this thread, then hand bytes to the writer."""
    files = collect_files(mon, view, mode=mode, link_ok=link_ok, teacher=teacher, learning=learning)
    return start_write(note.folder, note.root, files)


def _prune(root: Path, keep: int) -> None:
    folders = [path for path in root.iterdir() if path.is_dir()]
    folders.sort(key=lambda path: (path.stat().st_mtime, path.name))
    extra = len(folders) - int(keep)
    for path in folders[: max(0, extra)]:
        shutil.rmtree(path, ignore_errors=True)
        _RESERVED.discard(path)
