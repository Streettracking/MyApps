"""UDP JSON commands for the local Go2 command server.

The trainer talks to ``python main.py`` in go2_wr_server_v2-v2, which listens
on 127.0.0.1:5451 and forwards to the dog. This module never opens a robot SDK.
"""

from __future__ import annotations

import json
import socket
from typing import Any


def move_params(steer_x: float, steer_z: float) -> dict[str, float]:
    """Arrow mapping used by the existing control panel.

    Forward / back is ``x`` ±0.5. Left is ``z`` +1, right is ``z`` -1.
    ``y`` stays 0 (no strafe on this panel).
    """
    if steer_x > 0:
        x = 0.5
    elif steer_x < 0:
        x = -0.5
    else:
        x = 0.0
    if steer_z > 0:
        z = 1.0
    elif steer_z < 0:
        z = -1.0
    else:
        z = 0.0
    return {"x": x, "y": 0.0, "z": z}


class Go2CommandLink:
    def __init__(self, host: str = "127.0.0.1", port: int = 5451):
        self.host = host
        self.port = int(port)
        self.addr = (self.host, self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_command = ""
        self.last_error = ""
        self.ok = True
        self.sent: list[dict[str, Any]] = []

    def close(self) -> None:
        self.sock.close()

    def send(self, method: str, params: dict | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = params
        raw = json.dumps(payload).encode("utf-8")
        self.last_command = json.dumps(payload, separators=(",", ":"))
        try:
            self.sock.sendto(raw, self.addr)
            self.ok = True
            self.last_error = ""
        except OSError as exc:
            self.ok = False
            self.last_error = str(exc)
        self.sent.append(payload)
        if len(self.sent) > 40:
            self.sent = self.sent[-40:]
        return payload

    def move(self, steer_x: float, steer_z: float) -> dict[str, Any]:
        return self.send("Move", move_params(steer_x, steer_z))

    def stop(self) -> dict[str, Any]:
        return self.send("StopMove")

    def stand_up(self) -> dict[str, Any]:
        return self.send("StandUp")

    def stand_down(self) -> dict[str, Any]:
        return self.send("StandDown")

    def emergency_stop(self) -> dict[str, Any]:
        return self.send("emergency_stop")

    def status_line(self) -> str:
        where = f"UDP {self.host}:{self.port}"
        if self.ok:
            return f"{where}  ok"
        return f"{where}  error: {self.last_error}"
