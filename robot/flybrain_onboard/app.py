"""Onboard loop: mushroom body, pilot, and the laptop control port.

Autonomy and learning start off. That is watch-only: this process does
not call SportClient at all. StopMove is one edge, and only after this
process itself sent Move (a link or frame loss, or the step from moving
to stopped). E-STOP sends StopMove once and then blocks. StandUp,
StandDown, and RecoveryStand run only when the operator asks.

StandUp locks the Go2. BalanceStand follows about 0.7 s later, from
tick, so the HTTP handler and the main loop do not sleep. The first
Move after a takeover or an autonomy start sends that unlock itself
when the timer has not fired yet.
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
from sim.pilot import forward_clearance, scrub_range

try:
    from .drive import SportDrive
    from .sensors import cloud_forward
except ImportError:  # python3 /root/flybrain/main.py has no package parent
    from drive import SportDrive
    from sensors import cloud_forward


class BrainLoop:
    def __init__(self, drive: SportDrive, mb: MbTrainer, state_path: str, self_radius: float = 0.6):
        self.drive = drive
        self.mb = mb
        self.state_path = state_path
        self.self_radius = float(self_radius)
        self.pilot = Pilot(self_radius=self.self_radius)
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
        self._sent_move = False
        self._last_send = 0.0
        self._move_log_t = -1e9
        self._prev_camera = None
        self._prev_near = None
        self._learn_block_log = 0.0
        self._lock = threading.Lock()
        self._log("наблюдение: SportClient не трогаем, пока нет автономии или перехвата")

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

    def driving(self) -> bool:
        """Autonomy, or manual after an explicit takeover. Watch-only is neither."""
        if self.pilot.mode == "auto":
            return True
        return self.pilot.mode == "manual" and bool(self.pilot.took_over)

    def _stop_once(self) -> None:
        """One StopMove if this process has a Move outstanding. Otherwise silence."""
        if not self._sent_move:
            return
        self.drive.stop()
        self._sent_move = False
        self._moving = False

    def halt(self) -> None:
        self._stop_once()

    def on_estop(self) -> None:
        """One StopMove, then a block. A repeat while the block holds sends nothing."""
        if self.pilot.mode == "estop":
            return
        self.pilot.estop()
        self.drive.cancel_balance_timer()
        self.drive.stop()
        self._sent_move = False
        self._moving = False
        self._log("E-STOP")

    def on_space(self) -> None:
        self._stop_once()
        self.pilot.space()
        self._log("стоп")

    def on_takeover(self) -> None:
        self._stop_once()
        self.pilot.takeover()
        self._log("перехват: ручное, автономия сама не вернётся")

    def handle(self, data: dict, now: float) -> None:
        self.note_heartbeat(now)
        op = str(data.get("op", ""))
        with self._lock:
            if op == "autonomy_on":
                if self.pilot.start_auto(now):
                    self._log("автономия: поиск в одну сторону, z=+0.35, без качки")
                    self._log("сырой сектор — мозг; sector_smooth — медиана контроллера")
                else:
                    self._log("E-STOP держит стоп. M переводит в ручное")
            elif op == "autonomy_off":
                self._stop_once()
                self.pilot.stop_auto()
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
            elif op == "stand_up":
                code = self.drive.stand_up(now)
                self._log(f"StandUp code={code}")
            elif op == "stand_down":
                code = self.drive.stand_down()
                self._log(f"StandDown code={code}")
            elif op == "recovery_stand":
                if self.drive.recovery_stand():
                    self._log("RecoveryStand")
                else:
                    self._log("RecoveryStand нет в SDK")
            elif op == "save":
                self.mb.lidar_refresh = 1.5
                self.mb.save(self.state_path)
                self._log("сохранено")
            elif op == "manual":
                if self.driving():
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
            forward = forward_clearance(feat, self.self_radius)
        dist = scrub_range(dist, self.self_radius)
        forward = scrub_range(forward, self.self_radius)
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
        # Watch-only still finishes a stand the operator already asked for.
        # The HTTP thread only scheduled it.
        self._release_stand_lock(now)
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
        if not self.driving():
            # Watch-only, or the E-STOP block: no Move and no StopMove.
            # A BalanceStand that StandUp already scheduled ran above.
            return DriveCommand(0.0, 0.0, True, "stop", "никто", "")
        if self.pilot.mode != "auto":
            axes = manual
        elif abs(manual[0]) + abs(manual[1]) > 1e-6 and recent_manual:
            axes = manual
        else:
            axes = None
        if not link_ok or not frames_ok:
            # The sport service would keep our last Move for about a second.
            # One StopMove covers that. Further ticks stay silent.
            self._stop_once()
            self.pilot.phase = "stop"
            self.pilot.who = "никто"
            return DriveCommand(0.0, 0.0, True, "stop", "никто", "")
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
        self._emit(cmd, now)
        return cmd

    def _release_stand_lock(self, now: float) -> None:
        if self.pilot.mode == "estop":
            self.drive.cancel_balance_timer()
            return
        due = self.drive.balance_due
        if due is None or now < due:
            return
        self._balance_once()

    def _balance_once(self) -> None:
        with self._lock:
            if self.drive.pose == "balance_unavailable":
                self.drive.balance_due = None
                return
            if not self.drive.needs_balance():
                self.drive.balance_due = None
                return
            code = self.drive.balance_stand()
            missing = self.drive.pose == "balance_unavailable"
        if missing:
            self._log("BalanceStand нет в SDK")
        else:
            self._log(f"BalanceStand code={code}")

    def _log_manual_move(self, x: float, z: float, code, now: float) -> None:
        """One line a second, manual axes only, with the sport-service code."""
        if self.pilot.mode != "manual":
            return
        if float(now) - self._move_log_t < 1.0:
            return
        self._move_log_t = float(now)
        self._log(f"Move x={x:.2f} z={z:.2f} code={code}")

    def _emit(self, cmd: DriveCommand, now: float) -> None:
        if cmd.stop or (abs(cmd.x) < 1e-6 and abs(cmd.z) < 1e-6):
            self._stop_once()
            return
        if self.drive.needs_balance():
            self._balance_once()
        if (not self._sent_move) or (now - self._last_send >= 0.10):
            code = self.drive.move(cmd.x, cmd.z)
            sent = self.drive.moves[-1]
            self._sent_move = True
            self._moving = True
            self._last_send = now
            self._log_manual_move(sent[0], sent[2], code, now)

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
                "sector_smooth": self.pilot.track.sector_smooth,
                "hysteresis": self.pilot.track.as_dict(),
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


def make_brain(drive: SportDrive, state_path: str, npz_path: str | None = None, self_radius: float = 0.6) -> BrainLoop:
    mb = MbTrainer(npz_path or default_npz(), seed=1, eta=0.2, dan="teacher")
    mb.learn = False
    loop = BrainLoop(drive, mb, state_path, self_radius=self_radius)
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
