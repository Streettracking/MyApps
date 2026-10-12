"""Window and dock settings outside the connectome npz and outside _internal.

Windows uses ``%APPDATA%/recognize_trainer/window.json``. Everywhere else
uses ``$XDG_CONFIG_HOME`` or ``~/.config``. ``RECOGNIZER_UI_CONFIG`` overrides
the file path for tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

FILE_NAME = "window.json"
CRASH_NAME = "crash.log"
TRAINER_LOG_NAME = "trainer.log"
BRAIN_WINDOW_NAME = "brain_window.json"
APP_DIR = "recognize_trainer"


def persist_enabled() -> bool:
    """Dummy-driver tests stay on the built-in layout unless they opt in."""
    if os.environ.get("RECOGNIZER_UI_CONFIG"):
        return True
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        return False
    return True


def settings_path() -> Path:
    override = os.environ.get("RECOGNIZER_UI_CONFIG")
    if override:
        return Path(override)
    if os.name == "nt":
        root = os.environ.get("APPDATA") or str(Path.home())
        return Path(root) / APP_DIR / FILE_NAME
    root = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(root) / APP_DIR / FILE_NAME


def crash_log_path() -> Path:
    """Native crash stack beside window.json, outside npz and outside _internal."""
    return settings_path().with_name(CRASH_NAME)


def trainer_log_path() -> Path:
    """Stdout and stderr of a windowed trainer. Same folder as crash.log."""
    try:
        from .doggibot_snap import appdata_root

        return appdata_root() / TRAINER_LOG_NAME
    except Exception:
        return settings_path().with_name(TRAINER_LOG_NAME)


class _Silent:
    """Stand-in when the log file cannot be opened and the console is missing."""

    def write(self, _data):
        return 0

    def flush(self):
        return None

    def isatty(self):
        return False


def ensure_stdio():
    """A windowed exe has no stdout. Point the holes at trainer.log.

    A normal console is left alone. ``print`` must not raise when both
    streams are None.
    """
    import sys

    if sys.stdout is not None and sys.stderr is not None:
        return trainer_log_path()
    path = trainer_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        handle = _Silent()
    if sys.stdout is None:
        sys.stdout = handle
    if sys.stderr is None:
        sys.stderr = handle
    ensure_stdio.handle = handle
    return path


def install_crash_log():
    """Enable faulthandler for the exe. The handle stays alive on this function."""
    import sys

    import faulthandler

    if getattr(install_crash_log, "handle", None) is not None:
        return crash_log_path()
    path = crash_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a", encoding="utf-8")
        faulthandler.enable(file=handle, all_threads=True)
        install_crash_log.handle = handle
        return path
    except OSError:
        err = sys.stderr
        if err is None:
            return None
        try:
            faulthandler.enable(file=err, all_threads=True)
        except Exception:
            return None
        return None


install_crash_log.handle = None
ensure_stdio.handle = None


def default_settings() -> dict:
    return {
        "theme": "light",
        "maximized": False,
        "fullscreen": False,
        "x": None,
        "y": None,
        "w": None,
        "h": None,
        "layout": None,
        "go2_control_exe": "",
        "go2_server_exe": "",
        "pin_on_top": False,
        "teacher_stop_h": 0.45,
    }


def read_saved_theme():
    """``theme`` from window.json, or None when the file cannot be trusted.

    A UTF-8 BOM is skipped. A missing file, an empty file, and a half-written
    file return None. Callers that need a theme then use light.
    """
    path = settings_path()
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    name = str(data.get("theme") or "").strip().lower()
    if name in ("light", "dark"):
        return name
    return None


def load_settings() -> dict:
    path = settings_path()
    data = default_settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return data
    if not isinstance(raw, dict):
        return data
    data.update({k: raw[k] for k in data if k in raw})
    if raw.get("layout"):
        data["layout"] = raw["layout"]
    data["pin_on_top"] = _pinned_flag(data.get("pin_on_top"))
    return data


def brain_window_path() -> Path:
    """Size and position of the separate 3D window, beside window.json."""
    return settings_path().with_name(BRAIN_WINDOW_NAME)


def _pinned_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_brain_window() -> dict:
    data = {"x": None, "y": None, "w": None, "h": None, "pinned": False}
    if not persist_enabled():
        return data
    try:
        raw = json.loads(brain_window_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return data
    if not isinstance(raw, dict):
        return data
    for key in data:
        if key not in raw:
            continue
        data[key] = _pinned_flag(raw[key]) if key == "pinned" else raw[key]
    return data


def save_brain_window(geom: dict) -> None:
    if not persist_enabled():
        return
    path = brain_window_path()
    payload = {
        "x": geom.get("x"),
        "y": geom.get("y"),
        "w": geom.get("w"),
        "h": geom.get("h"),
        "pinned": _pinned_flag(geom.get("pinned")),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return


def save_settings(data: dict) -> None:
    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return
