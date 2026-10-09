"""Laptop side of the onboard fly brain.

The window polls ``http://<robot>:8090/status`` and posts buttons. That poll
is the heartbeat. This module does not send UDP commands and does not run
the mushroom body. Stand, walk, and stop go to the robot port. A bind probe
on 5451 only checks whether ``python main.py`` is also listening.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any


def udp_bridge_listening(host: str = "127.0.0.1", port: int = 5451) -> bool:
    """True when something already holds the local command port.

    The probe binds and closes. It does not send a command.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, int(port)))
    except OSError:
        return True
    finally:
        sock.close()
    return False


class OnboardLink:
    def __init__(self, host: str, port: int = 8090):
        self.host = host
        self.port = int(port)
        self.base = f"http://{host}:{int(port)}"
        self.status: dict[str, Any] = {}
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
        """GET /status. The robot treats any request as a laptop heartbeat."""
        got = self._request("GET", "/status")
        if got:
            self.status = got
        return self.status

    def post(self, op: str, **extra: Any) -> dict:
        body = {"op": op}
        body.update(extra)
        self.last_command = op
        got = self._request("POST", "/cmd", body)
        if got:
            self.status = got
        return self.status

    def status_line(self, bridge_open: bool = False) -> str:
        where = f"борт {self.host}:{self.port}"
        if self.ok and bridge_open:
            return f"{where}  ok   два источника: это окно и python main.py (UDP 5451)"
        if self.ok:
            return f"{where}  ok   UDP 5451 не используется"
        if self.last_error:
            return f"{where}  нет связи: {self.last_error[:80]}"
        return f"{where}  нет связи"
