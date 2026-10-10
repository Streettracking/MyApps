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


def install_crash_log():
    """Enable faulthandler for the exe. The handle stays alive on this function."""
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
        faulthandler.enable(all_threads=True)
        return None


install_crash_log.handle = None


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
    }


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
    return data


def save_settings(data: dict) -> None:
    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return
