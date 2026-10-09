"""Onboard loop: mushroom body, pilot, and the laptop control port.

Autonomy and learning start off. The laptop heartbeat and camera frames
are watched. Either going stale for a second calls StopMove. Arrows do
not arrive here as Move; a takeover command switches to manual, and only
then do posted axes drive. Manual axes older than 300 ms become StopMove
and stay manual.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from sim.frame_sense import features_from_frames
from sim.learn_flash import flash_payload
from sim.map_marks import MarkLayer
from sim.mb_train import MbTrainer, default_npz
from sim.pilot import LINK_HOLD_S, MANUAL_HOLD_S, DriveCommand, Pilot
from sim.pilot import forward_clearance

try:
    from .drive import SportDrive
    from .sensors import cloud_forward
except ImportError:  # python3 /root/flybrain/main.py has no package parent
    from drive import SportDrive
    from sensors import cloud_forward


class BrainLoop:
    def __init__(self, drive: SportDrive, mb: MbTrainer, state_path: str):
        self.drive = drive
        self.mb = mb
        self.state_path = state_path
        self.pilot = Pilot()
        self.learn = False
        self.mb.learn = False
        self.marks = MarkLayer()
        self.log = deque(maxlen=16)
        self.t0 = time.monotonic()
        self.last_hb = self.t0
        self.last_frame_t = 0.0
        self.last_manual_t = 0.0
        self.manual_x = 0.0
        self.manual_z = 0.0
        self.frames_ok = False
        self.recognized = False
        self.sector = None
        self.dist_m = None
        self.forward_m = None
        self.readout = 0.0
        self.confidence = 0.0
        self.confidence_ready = False
        self.cloud_ranges = None
        self.cloud_t = 0.0
        self._teach = None
        self._moving = False
        self._last_send = 0.0
        self._prev_camera = None
        self._prev_near = None
        self._learn_block_log = 0.0
        self._lock = threading.Lock()
        self._log("наблюдение: автономия выключена, обучение выключено")

    def _log(self, text: str) -> None:
        self.log.append(f"{time.monotonic() - self.t0:6.1f}s  {text}")

    def note_heartbeat(self, now: float) -> None:
        self.last_hb = float(now)

    def note_manual(self, x: float, z: float, now: float) -> None:
        self.manual_x = float(x)
        self.manual_z = float(z)
        self.last_manual_t = float(now)

    def set_cloud(self, ranges, now: float) -> None:
        self.cloud_ranges = ranges
        self.cloud_t = float(now)

    def halt(self) -> None:
        self.drive.stop()
        self._moving = False

    def on_estop(self) -> None:
        self.pilot.estop()
        self.halt()
        self._log("E-STOP")

    def on_space(self) -> None:
        self.pilot.space()
        self.halt()
        self._log("стоп")

    def on_takeover(self) -> None:
        self.pilot.takeover()
        self.halt()
        self._log("перехват: ручное, автономия сама не вернётся")

    def handle(self, data: dict, now: float) -> None:
        self.note_heartbeat(now)
        op = str(data.get("op", ""))
        with self._lock:
            if op == "autonomy_on":
                if self.pilot.start_auto(now):
                    self._log("автономия: поиск, пока мозг не скажет «узнаю»")
                else:
                    self._log("E-STOP держит стоп. M переводит в ручное")
            elif op == "autonomy_off":
                self.pilot.stop_auto()
                self.halt()
                self._log("автономия выключена")
            elif op == "learn_on":
                self.learn = True
                self.mb.learn = True
                self._log("обучение включено")
            elif op == "learn_off":
                self.learn = False
                self.mb.learn = False
                self._log("обучение выключено, T/X веса не меняют")
            elif op == "treat":
                self._teach = "pam"
            elif op == "punish":
                self._teach = "ppl1"
            elif op == "estop":
                self.on_estop()
            elif op == "space":
                self.on_space()
            elif op == "takeover":
                self.on_takeover()
            elif op == "save":
                self.mb.lidar_refresh = 1.5
                self.mb.save(self.state_path)
                self._log("сохранено")
            elif op == "manual":
                self.note_manual(float(data.get("x", 0.0)), float(data.get("z", 0.0)), now)
            elif op == "heartbeat":
                pass

    def ingest(self, camera: np.ndarray, lidar: np.ndarray, now: float) -> None:
        """One fresh pair. Empty lidar after a reset must not be passed in."""
        feat, _ego, near = features_from_frames(camera, lidar, self._prev_camera, self._prev_near)
        self._prev_camera = camera
        self._prev_near = near
        self.frames_ok = True
        self.last_frame_t = float(now)
        self.mb.learn = self.learn
        fwd = self.mb.forward(feat)
        value = self.mb.last_readout
        conf = self.mb.observe(feat, value)
        seen = bool(conf.ready and conf.recognized)
        self.marks.consider(self.mb, feat, float(now), recognized=seen, mode="live")
        dist = None
        forward = None
        ranges = self.cloud_ranges
        if ranges is not None and float(now) - self.cloud_t < 0.5:
            if seen and self.marks.aim_sector is not None:
                dist = ranges[int(self.marks.aim_sector)]
            forward = cloud_forward(ranges)
        else:
            if seen:
                dist = self.marks.aim_dist
            forward = forward_clearance(feat)
        kind = self._teach
        self._teach = None
        if kind and not self.learn:
            if float(now) - self._learn_block_log > 1.0:
                self._log("обучение выключено, T/X веса не меняют")
                self._learn_block_log = float(now)
            kind = None
        if kind:
            self.mb.teach(fwd, kind, float(now))
        with self._lock:
            self.recognized = seen
            self.sector = self.marks.aim_sector if seen else None
            self.dist_m = dist
            self.forward_m = forward
            self.readout = float(value)
            self.confidence = float(conf.percent)
            self.confidence_ready = bool(conf.ready)

    def tick(self, now: float) -> DriveCommand:
        now = float(now)
        link_ok = (now - self.last_hb) <= LINK_HOLD_S
        frames_ok = self.frames_ok and (now - self.last_frame_t) <= LINK_HOLD_S
        recent_manual = (now - self.last_manual_t) <= MANUAL_HOLD_S
        if recent_manual:
            manual = (self.manual_x, self.manual_z)
        else:
            manual = (0.0, 0.0)
        with self._lock:
            recognized = self.recognized
            sector = self.sector
            dist_m = self.dist_m
            forward_m = self.forward_m
        if self.pilot.mode != "auto":
            axes = manual
        elif abs(manual[0]) + abs(manual[1]) > 1e-6 and recent_manual:
            axes = manual
        else:
            axes = None
        if not link_ok or not frames_ok:
            cmd = self.pilot.command(
                now,
                (0.0, 0.0),
                focused=True,
                frames_ok=False,
                link_ok=link_ok,
                recognized=recognized,
                sector=sector,
                dist_m=dist_m,
                forward_m=forward_m,
                manual_axes=None,
            )
            # Missing frames in manual still stop. The sport service would
            # otherwise keep the last Move for about a second.
            cmd = DriveCommand(0.0, 0.0, True, "stop", "никто", cmd.hint)
            self.pilot.phase = "stop"
            self.pilot.who = "никто"
            self._emit(cmd, now, force_stop=True)
            return cmd
        cmd = self.pilot.command(
            now,
            (0.0, 0.0),
            focused=True,
            frames_ok=True,
            link_ok=True,
            recognized=recognized,
            sector=sector,
            dist_m=dist_m,
            forward_m=forward_m,
            manual_axes=axes,
        )
        self._emit(cmd, now, force_stop=False)
        return cmd

    def _emit(self, cmd: DriveCommand, now: float, force_stop: bool) -> None:
        if cmd.stop or force_stop or (abs(cmd.x) < 1e-6 and abs(cmd.z) < 1e-6):
            if self._moving or force_stop:
                self.halt()
            return
        if (not self._moving) or (now - self._last_send >= 0.10):
            self.drive.move(cmd.x, cmd.z)
            self._moving = True
            self._last_send = now

    def status(self) -> dict:
        flash = flash_payload(self.mb.flash)
        link_ok = (time.monotonic() - self.last_hb) <= LINK_HOLD_S
        with self._lock:
            return {
                "mode": self.pilot.mode,
                "label": self.pilot.label(),
                "phase": self.pilot.phase,
                "who": self.pilot.who,
                "took_over": bool(self.pilot.took_over),
                "learning": bool(self.learn),
                "autonomy": bool(self.pilot.autonomy),
                "recognized": bool(self.recognized),
                "confidence": self.confidence,
                "confidence_ready": self.confidence_ready,
                "readout": self.readout,
                "sector": self.sector,
                "distance_m": self.dist_m,
                "forward_m": self.forward_m,
                "x": self.drive.moves[-1][0] if self._moving and self.drive.moves else 0.0,
                "z": self.drive.moves[-1][2] if self._moving and self.drive.moves else 0.0,
                "n_pam": int(self.mb.n_pam),
                "n_ppl1": int(self.mb.n_ppl1),
                "frames_ok": bool(self.frames_ok),
                "link_ok": bool(link_ok),
                "hint": self.pilot.hint,
                "kc_on": int(self.mb.last_kc_on),
                "kc_n": int(self.mb.brain.n_kc),
                "drift": float(self.mb.last_drift),
                "flash": flash,
                "log": list(self.log),
            }


def make_brain(drive: SportDrive, state_path: str, npz_path: str | None = None) -> BrainLoop:
    mb = MbTrainer(npz_path or default_npz(), seed=1, eta=0.2, dan="teacher")
    mb.learn = False
    loop = BrainLoop(drive, mb, state_path)
    from pathlib import Path

    path = Path(state_path)
    if path.is_file():
        mb.load(path)
        mb.learn = False
        loop.learn = False
        loop._log("загружены веса KC→MBON")
    return loop


class _Handler(BaseHTTPRequestHandler):
    loop: BrainLoop

    def _send(self, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self.loop.note_heartbeat(time.monotonic())
        if self.path.split("?", 1)[0] != "/status":
            self.send_response(404)
            self.end_headers()
            return
        self._send(self.loop.status())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        now = time.monotonic()
        if self.path.split("?", 1)[0] != "/cmd":
            self.send_response(404)
            self.end_headers()
            return
        self.loop.handle(data, now)
        self._send(self.loop.status())

    def log_message(self, _format, *_args) -> None:
        return


def serve(loop: BrainLoop, port: int = 8090) -> ThreadingHTTPServer:
    _Handler.loop = loop
    server = ThreadingHTTPServer(("0.0.0.0", int(port)), _Handler)
    threading.Thread(target=server.serve_forever, name="flybrain-http", daemon=True).start()
    return server
