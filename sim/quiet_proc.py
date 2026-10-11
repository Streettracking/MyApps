"""Start a child with no console and append its output to a log file.

The log sits next to ``trainer.log`` and ``crash.log``. A GUI child still
opens its own window: ``CREATE_NO_WINDOW`` only suppresses a console.
"""

from __future__ import annotations

import os
import subprocess

# Same bits the command server already uses. Present so a Linux test can see them.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000


def log_path(name: str):
    from .doggibot_snap import appdata_root

    return appdata_root() / str(name)


def open_log(name: str):
    """Append-binary handle, or ``DEVNULL`` when the folder cannot be written."""
    path = log_path(name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "ab", buffering=0)
    except OSError:
        return subprocess.DEVNULL


def creationflags() -> int:
    """No console, detached from the trainer's group. Zero off Windows."""
    if os.name != "nt":
        return 0
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW)
    detached = getattr(subprocess, "DETACHED_PROCESS", _DETACHED_PROCESS)
    group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", _CREATE_NEW_PROCESS_GROUP)
    return int(no_window) | int(detached) | int(group)


def popen_quiet(argv, log_name: str, cwd=None):
    """``Popen`` with a hidden console. The caller keeps the log handle alive."""
    handle = open_log(log_name)
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": handle,
        "stderr": subprocess.STDOUT,
    }
    if cwd:
        kwargs["cwd"] = str(cwd)
    if os.name == "nt":
        kwargs["creationflags"] = creationflags()
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([str(part) for part in argv], **kwargs)
    return proc, handle
