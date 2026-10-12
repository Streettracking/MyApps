"""Laptop side of the onboard fly brain.

The window polls ``http://<robot>:8090/status`` and posts buttons. The poll
is read-only. The link stays up only while this window posts ``/heartbeat``
or ``/cmd``. This module does not send UDP commands and does not run
the mushroom body. Stand, walk, and stop go to the robot port. A taken UDP
port is a second command source only when the listener's command line is
``python main.py``.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any

# Shorter than the robot's link hold, so one missed post still leaves the link up.
HEARTBEAT_PERIOD_S = 0.40


def udp_bridge_listening(host: str = "127.0.0.1", port: int = 5451) -> bool:
    """True when something already holds the local command port.

    The probe binds and closes. It does not send a command. A taken port is
    not ``python main.py``: the go2 server uses the same UDP port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, int(port)))
    except OSError:
        return True
    finally:
        sock.close()
    return False


def _udp_inodes(port: int) -> set:
    """Inode numbers of UDP sockets bound to ``port``, from the Linux tables."""
    needle = ":%04X" % int(port)
    found = set()
    for name in ("/proc/net/udp", "/proc/net/udp6"):
        try:
            with open(name, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError:
            continue
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue
            if parts[1].upper().endswith(needle):
                found.add(parts[9])
    return found


def _linux_cmdlines_on_udp(port: int) -> list:
    inodes = _udp_inodes(port)
    if not inodes:
        return []
    owners = []
    try:
        pids = os.listdir("/proc")
    except OSError:
        return []
    for pid in pids:
        if not pid.isdigit():
            continue
        fd_dir = "/proc/%s/fd" % pid
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        hit = False
        for fd in fds:
            try:
                target = os.readlink("%s/%s" % (fd_dir, fd))
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]") and target[8:-1] in inodes:
                hit = True
                break
        if not hit:
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        text = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if text:
            owners.append(text)
    return owners


def _windows_cmdlines_on_udp(port: int) -> list:
    """Command lines holding this UDP port. Empty when the lookup cannot prove one."""
    import subprocess

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        table = subprocess.check_output(["netstat", "-ano", "-p", "UDP"], creationflags=flags, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    pids = set()
    token = ":%s" % int(port)
    for line in table.decode("utf-8", "replace").splitlines():
        if token not in line:
            continue
        parts = line.split()
        if parts and parts[-1].isdigit():
            pids.add(parts[-1])
    owners = []
    for pid in pids:
        try:
            raw = subprocess.check_output(
                ["wmic", "process", "where", "ProcessId=%s" % pid, "get", "CommandLine", "/value"],
                creationflags=flags,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        text = raw.decode("utf-8", "replace")
        for row in text.splitlines():
            if row.startswith("CommandLine="):
                owners.append(row.split("=", 1)[1].strip())
    return owners


def main_py_listening(port: int = 5451) -> bool:
    """True only when a process whose command line is ``main.py`` holds the UDP port.

    A bind failure is not enough: ``go2_wr_server`` listens on the same port.
    """
    if os.name == "nt":
        owners = _windows_cmdlines_on_udp(port)
    else:
        owners = _linux_cmdlines_on_udp(port)
    for text in owners:
        if "main.py" in text.replace("\\", "/"):
            return True
    return False


class OnboardLink:
    def __init__(self, host: str, port: int = 8090):
        self.host = host
        self.port = int(port)
        self.base = f"http://{host}:{int(port)}"
        self.status: dict[str, Any] = {}
        self.status_at = 0.0
        self.ok = False
        self.last_error = ""
        self.last_command = ""

    def _request(self, method: str, path: str, payload: dict | None = None, timeout: float = 0.8) -> dict:
        data = None
        headers = {"User-Agent": "recognize-trainer"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            self.ok = True
            self.last_error = ""
            if not raw:
                return {}
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {}
        except urllib.error.URLError as exc:
            self.ok = False
            self.last_error = str(exc.reason if hasattr(exc, "reason") else exc)
            return {}
        except Exception as exc:
            self.ok = False
            self.last_error = str(exc)
            return {}

    def poll(self) -> dict:
        """GET /status. Read-only: this does not refresh the robot's link."""
        got = self._request("GET", "/status")
        if got:
            self.status = got
            self.status_at = time.monotonic()
        return self.status

    def heartbeat(self) -> dict:
        """POST /heartbeat. This is what keeps the link up while the trainer runs."""
        self.last_command = "heartbeat"
        got = self._request("POST", "/heartbeat", {})
        if got:
            self.status = got
            self.status_at = time.monotonic()
        return self.status

    def post(self, op: str, **extra: Any) -> dict:
        body = {"op": op}
        body.update(extra)
        self.last_command = op
        got = self._request("POST", "/cmd", body)
        if got:
            self.status = got
            self.status_at = time.monotonic()
        return self.status

    def status_age(self, now=None):
        """Seconds since the last non-empty status. None when none has arrived."""
        if not self.status or self.status_at <= 0.0:
            return None
        clock = time.monotonic() if now is None else float(now)
        return clock - self.status_at

    def status_line(self, bridge_open: bool = False) -> str:
        where = f"борт {self.host}:{self.port}"
        if self.ok and bridge_open:
            return f"{where}  ok   два источника: это окно и python main.py (UDP 5451)"
        if self.ok:
            return f"{where}  ok   UDP 5451 не используется"
        if self.last_error:
            return f"{where}  нет связи: {self.last_error[:80]}"
        return f"{where}  нет связи"
