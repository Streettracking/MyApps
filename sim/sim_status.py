"""Local status and pause for ``--sim``. Same routes as the onboard brain.

The simulator has no robot port. This binds localhost only, so an external
reader can GET metrics and POST pause without opening the process to the
network. The 3D view keeps its own port.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SIM_STATUS_HOST = "127.0.0.1"
SIM_STATUS_PORT = 8090


def sim_status(session) -> dict:
    """The fields an external reader needs. Numpy values become plain floats."""
    mb = getattr(session, "mb", None)
    drift = 0.0
    session_drift = 0.0
    frozen = False
    n_pam = 0
    n_ppl1 = 0
    if mb is not None:
        from .brain_panel import live_drift, session_drift as _session

        drift = float(live_drift(mb))
        session_drift = float(_session(mb))
        frozen = bool(mb.learning_hold)
        n_pam = int(mb.n_pam)
        n_ppl1 = int(mb.n_ppl1)
    watch = getattr(session, "approach", None)
    index = None if watch is None else watch.index()
    marker = getattr(session, "run_marker", None)
    marked = (
        marker.status()
        if marker is not None
        else {"run_stage": "", "run_side": "", "run_label": "", "drive_log": []}
    )
    return {
        "mode": "sim",
        "paused": bool(getattr(session, "paused", False)),
        "learning": bool(getattr(session, "learn", False)),
        "learning_frozen": frozen,
        "t": float(getattr(getattr(session, "world", None), "t", 0.0) or 0.0),
        "drift": drift,
        "session_drift": session_drift,
        "n_pam": n_pam,
        "n_ppl1": n_ppl1,
        "approach_index": None if index is None else float(index),
        "phase": str(getattr(getattr(session, "pilot", None), "phase", "") or ""),
        "run_stage": marked["run_stage"],
        "run_side": marked["run_side"],
        "run_label": marked["run_label"],
        "drive_log": marked["drive_log"],
    }


def apply_sim_command(session, data: dict) -> None:
    """Pause, resume, and the same ops on POST /cmd. ``{}`` is a valid body.

    The clock is the arena time, the same clock ``step`` passes into the hold.
    """
    if not isinstance(data, dict):
        data = {}
    op = str(data.get("op", ""))
    now = float(getattr(getattr(session, "world", None), "t", 0.0) or 0.0)
    mb = getattr(session, "mb", None)
    if op == "pause":
        session.paused = True
        if mb is not None:
            mb.note_motion_hold(True, False, now)
        if hasattr(session, "_log"):
            session._log("пауза: ход ноль, пластичность заморожена")
    elif op == "resume":
        session.paused = False
        if mb is not None:
            mb.note_motion_hold(False, False, now)
        if hasattr(session, "_log"):
            session._log("ход разрешён, обучение заморожено до движения и ещё 1 с")
    elif op == "learn_off":
        session.learn = False
        if mb is not None:
            mb.learn = False
    elif op == "learn_on":
        session.learn = True
        if mb is not None:
            mb.learn = True
    elif op == "run_marker":
        from .run_marker import apply_marker_fields

        apply_marker_fields(session, data.get("stage"), data.get("side"))


class _Handler(BaseHTTPRequestHandler):
    session = None

    def _send(self, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/status":
            self.send_response(404)
            self.end_headers()
            return
        self._send(sim_status(self.session))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        path = self.path.split("?", 1)[0]
        if path == "/pause":
            data["op"] = "pause"
        elif path == "/resume":
            data["op"] = "resume"
        elif path != "/cmd":
            self.send_response(404)
            self.end_headers()
            return
        apply_sim_command(self.session, data)
        self._send(sim_status(self.session))

    def log_message(self, _format, *_args) -> None:
        return


def serve_sim(session, port: int = SIM_STATUS_PORT, host: str = SIM_STATUS_HOST):
    """GET /status, POST /cmd, POST /pause, POST /resume on localhost.

    Returns the server, or None when the port is already taken. The thread
    is a daemon. Pause and resume accept ``{}``.
    """
    handler = type("SimStatusHandler", (_Handler,), {"session": session})
    try:
        server = ThreadingHTTPServer((host, int(port)), handler)
    except OSError:
        return None
    threading.Thread(target=server.serve_forever, name="sim-status", daemon=True).start()
    return server
