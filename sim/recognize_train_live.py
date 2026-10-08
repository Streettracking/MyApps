"""Live recognition trainer for one Go2.

Sensors: HTTP JPEG preview (camera.jpg, lidar.jpg).
Motion: UDP JSON to the local command server (go2_wr_server), and only when
the operator holds an arrow key in this focused window.

Learning is in the fly mushroom body on this machine: raw JPEG → PN → KC → MBON,
and only KC→MBON changes. Default DAN is the operator. T injects appetitive PAM,
X injects aversive PPL1. D and N only choose which monitor curve a frame joins.
The mushroom body never sends Move.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

from .frame_sense import decode_image_bytes, features_from_frames
from .go2_udp import Go2CommandLink
from .mb_train import MbTrainer, default_npz
from .recognize import ConspecificRecognizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "logs" / "mb_train_state.npz"
DEFAULT_IP = "192.168.35.213"
DEFAULT_PREVIEW = 8088
DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 5451

LIVE_KEYS = (
    "Arrows drive   T TREAT   X punish   Space stop   -/+ stand   E E-STOP   "
    "D/N monitor only   P R S L F12 Esc"
)


def preview_base(ip: str, port: int) -> str:
    return f"http://{ip}:{port}"


def fetch_bytes(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "recognize-trainer"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise ConnectionError(str(exc.reason if hasattr(exc, "reason") else exc)) from exc


def offline_message(base: str, exc: BaseException) -> str:
    return (
        f"Cannot reach {base}/camera.jpg ({exc}). "
        "Start order: (1) robot preview server on this port, "
        "(2) optional python main.py in go2_wr_server_v2-v2 for the UDP bridge, "
        "(3) this trainer. The trainer does not send Move until a key is held."
    )


class PreviewPull:
    """Background JPEG fetch so a stalled camera does not block teleop."""

    def __init__(self, base: str):
        self.base = base
        self.error = ""
        self.camera = None
        self.lidar = None
        self.frame_id = 0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="preview", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop:
            try:
                cam_b = fetch_bytes(f"{self.base}/camera.jpg", timeout=2.0)
                lid_b = fetch_bytes(f"{self.base}/lidar.jpg", timeout=2.0)
                cam = decode_image_bytes(cam_b)
                lid = decode_image_bytes(lid_b)
            except Exception as exc:
                with self._lock:
                    self.error = offline_message(self.base, exc)
                time.sleep(0.4)
                continue
            with self._lock:
                self.camera = cam
                self.lidar = lid
                self.error = ""
                self.frame_id += 1
            time.sleep(0.05)

    def latest(self):
        with self._lock:
            return self.frame_id, self.camera, self.lidar, self.error


def apply_teleop(link: Go2CommandLink, inp, state: dict, now: float, log) -> None:
    """Keys to UDP. Nothing here is sent unless the window is focused."""
    if not inp.focused:
        if state["moving"]:
            link.stop()
            state["moving"] = False
            log("focus lost, StopMove")
        return
    if inp.estop:
        link.emergency_stop()
        link.stop()
        state["moving"] = False
        state["latched"] = True
        log("E-STOP emergency_stop")
        return
    if state["latched"]:
        if abs(inp.steer_x) + abs(inp.steer_z) == 0:
            state["latched"] = False
        else:
            return
    if inp.stand_up:
        link.stand_up()
        log("StandUp")
    if inp.stand_down:
        link.stand_down()
        log("StandDown")
    idle = inp.stop or (inp.steer_x == 0.0 and inp.steer_z == 0.0)
    if idle:
        if state["moving"]:
            link.stop()
            state["moving"] = False
        return
    if (not state["moving"]) or (now - state["last"] >= 0.10):
        link.move(inp.steer_x, inp.steer_z)
        state["moving"] = True
        state["last"] = now


class LiveSession:
    def __init__(
        self,
        state_path: Path,
        learner: str = "mb",
        dan: str = "teacher",
        eta: float = 0.2,
        npz: Path | None = None,
        seed: int = 1,
    ):
        if learner not in ("mb", "hebb"):
            raise ValueError(learner)
        self.learner_kind = learner
        self.dan = dan
        self.state_path = state_path
        self.learn = True
        self.operator: str | None = None
        self.peer_curve: list[tuple[float, float]] = []
        self.other_curve: list[tuple[float, float]] = []
        self.metric_curve: list[dict] = []
        self.log: deque[str] = deque(maxlen=24)
        self.t0 = time.monotonic()
        self.last_like = 0.0
        self.last_match = 0.0
        self.treat_flash = False
        self._prev_like = 0.0
        self._last_spike = -10.0
        self._last_metric = 0.0
        self._last_ckpt = 0.0
        self._prev_camera = None
        self._prev_near = None
        self.mb: MbTrainer | None = None
        self.recognizer: ConspecificRecognizer | None = None
        if learner == "mb":
            self.mb = MbTrainer(npz or default_npz(), seed=seed, eta=eta, dan=dan)
            self._log(f"MB training  DAN={dan}  KC→MBON only  MB does not drive")
        else:
            self.recognizer = ConspecificRecognizer()
            self._log("Hebbian comparison  no zones  MB does not drive")
        self._record_metric()

    def now(self) -> float:
        return time.monotonic() - self.t0

    def _log(self, text: str) -> None:
        self.log.append(f"{self.now():6.1f}s  {text}")

    def _record_metric(self) -> None:
        if self.mb is not None:
            snap = {"t": self.now(), "drift": self.mb.last_drift, "readout": self.mb.last_readout}
        else:
            assert self.recognizer is not None
            snap = self.recognizer.snapshot()
            snap["t"] = self.now()
        self.metric_curve.append(snap)

    def on_frame(self, camera: np.ndarray, lidar: np.ndarray, teach: str | None = None) -> None:
        feat, ego, near = features_from_frames(camera, lidar, self._prev_camera, self._prev_near)
        self._prev_camera = camera
        self._prev_near = near
        self.treat_flash = False
        # Operator D/N is intentionally not an argument of teaching.
        if self.mb is not None:
            self.mb.learn = self.learn
            fwd = self.mb.forward(feat)
            kind = teach if self.dan == "teacher" else None
            if kind == "pam":
                self.treat_flash = True
            self.mb.teach(fwd, kind, self.now())
            if kind == "pam" and (self.mb.n_pam <= 2 or self.mb.n_pam % 15 == 0):
                self._log(f"DAN treat  PAM  #{self.mb.n_pam}")
            elif kind == "ppl1" and (self.mb.n_ppl1 <= 2 or self.mb.n_ppl1 % 15 == 0):
                self._log(f"DAN punish  PPL1  #{self.mb.n_ppl1}")
            value = self.mb.last_readout
            self.last_like = float(value)
            self.last_match = float(value)
            if self.now() - self._last_metric >= 1.0:
                self.mb.note_drift(self.now())
                self._record_metric()
                self._last_metric = self.now()
        else:
            rec = self.recognizer
            assert rec is not None
            n_before = rec.n_self
            if self.learn:
                like = rec.observe(feat, ego)
            else:
                like = rec.likeness(feat)
            match = rec.learned_match(feat)
            value = float(match)
            self.last_like = float(like)
            self.last_match = value
            if rec.n_self > n_before and (rec.n_self <= 4 or rec.n_self % 8 == 0):
                self._log(f"prototype updated  n={rec.n_self}")
            if like > 0.8 and self._prev_like < 0.45 and self.now() - self._last_spike > 1.5:
                self._log(f"likeness spike  {like:.2f}")
                self._last_spike = self.now()
            self._prev_like = like
            if self.now() - self._last_metric >= 1.0:
                self._record_metric()
                self._last_metric = self.now()
        point = (self.now(), float(value))
        if self.operator == "dog":
            self.peer_curve.append(point)
        elif self.operator == "none":
            self.other_curve.append(point)
        if self.now() - self._last_ckpt >= 30.0:
            self.save()
            self._log(f"checkpoint saved  {self.state_path.name}")
            self._last_ckpt = self.now()

    def save(self) -> None:
        if self.mb is not None:
            self.mb.save(self.state_path)
        elif self.recognizer is not None:
            self.recognizer.save(self.state_path)
        self._log(f"saved  {self.state_path.name}")

    def load(self) -> None:
        if self.mb is not None:
            self.mb.load(self.state_path)
            self.dan = self.mb.dan
            self._log(f"loaded KC→MBON  drift={self.mb.last_drift:.1f}")
        elif self.recognizer is not None:
            self.recognizer.load(self.state_path)
            self._log(f"loaded  n={self.recognizer.n_self}")

    def reset(self) -> None:
        if self.mb is not None:
            self.mb.reset()
            self._log("KC→MBON weights reset")
        elif self.recognizer is not None:
            self.recognizer.reset()
            self._log("prototype reset")
        self._prev_camera = None
        self._prev_near = None

    def label_text(self) -> str:
        if self.operator == "dog":
            return "operator label D — monitor only, not used for learning"
        if self.operator == "none":
            return "operator label N — monitor only, not used for learning"
        return "press D when a dog is in view, N for distractor or empty — monitor only, not a DAN"


def run_live_gui(session: LiveSession, pull: PreviewPull, link: Go2CommandLink, seconds: float = 0.0) -> None:
    from .train_monitor import MonitorView, TrainMonitor

    mon = TrainMonitor("Go2 recognition trainer")
    tele = {"moving": False, "last": 0.0, "latched": False}
    seen_frame = 0
    announced = False
    try:
        while True:
            inp = mon.pump()
            if inp.quit:
                break
            if inp.label == "dog":
                session.operator = "dog"
                session._log("monitor label: dog in view")
            elif inp.label == "none":
                session.operator = "none"
                session._log("monitor label: no dog")
            if inp.pause_learn:
                session.learn = not session.learn
                session._log("learning paused" if not session.learn else "learning resumed")
            if inp.reset:
                session.reset()
            if inp.save:
                session.save()
            if inp.load:
                try:
                    session.load()
                except FileNotFoundError:
                    session._log("no saved state")
            apply_teleop(link, inp, tele, time.monotonic(), session._log)
            frame_id, camera, lidar, error = pull.latest()
            if error and not announced:
                print(error, file=sys.stderr)
                announced = True
            if not error:
                announced = False
            teach = None
            if inp.treat:
                teach = "pam"
            elif inp.punish:
                teach = "ppl1"
            new_frame = frame_id != seen_frame and camera is not None
            teach_now = teach and camera is not None and session.dan == "teacher"
            if new_frame or teach_now:
                session.on_frame(camera, lidar, teach=teach)
                seen_frame = frame_id
            view = _live_view(session, camera, lidar, error, link, inp.focused)
            mon.draw(view)
            if inp.screenshot:
                dest = ROOT / "logs" / "monitor_shot.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                mon.save_screenshot(str(dest))
                session._log(f"screenshot  {dest.name}")
            if seconds > 0 and session.now() >= seconds:
                break
    finally:
        try:
            link.stop()
        finally:
            link.close()
            pull.stop()


def _scaled(session: LiveSession, rows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not rows:
        return []
    vals = [v for _, v in session.peer_curve[-400:]] + [v for _, v in session.other_curve[-400:]]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 1.0
    span = hi - lo if hi > lo else 1.0
    return [(t, (v - lo) / span) for t, v in rows[-400:]]


def _live_view(session: LiveSession, camera, lidar, error: str, link: Go2CommandLink, focused: bool):
    from .train_monitor import MonitorView

    common = dict(
        camera=camera,
        lidar=lidar,
        sensor_error=error,
        likeness=session.last_like,
        learned_match=session.last_match,
        metric_curve=session.metric_curve[-120:],
        log_lines=list(session.log),
        paused=not session.learn,
        udp_status=link.status_line(),
        last_command=link.last_command or "—",
        operator_label=session.label_text(),
        t=session.now(),
        focused=focused,
        keys_hint=LIVE_KEYS if session.learner_kind == "mb" else (
            "Arrows drive   Space stop   -/+ stand   E E-STOP   D/N monitor only   P R S L F12 Esc"
        ),
    )
    if session.mb is not None:
        caption = "appetitive MBON  (approach − avoid)" if session.dan == "teacher" else "familiarity  (− novelty MBON)"
        return MonitorView(
            title="MB training   live dog",
            peer_curve=_scaled(session, session.peer_curve),
            other_curve=_scaled(session, session.other_curve),
            mode_label=f"robot   MB DAN={session.dan}   MB does not drive",
            learner="mb",
            dan_mode=session.dan,
            kc_on=session.mb.last_kc_on,
            kc_n=session.mb.brain.n_kc,
            kc_bins=session.mb.last_kc_bins,
            drift=session.mb.last_drift,
            drift_curve=session.mb.drift_curve[-120:],
            dan_events=session.mb.dan_events[-240:],
            treat_flash=session.treat_flash,
            readout_caption=caption,
            n_pam=session.mb.n_pam,
            n_ppl1=session.mb.n_ppl1,
            **common,
        )
    rec = session.recognizer
    assert rec is not None
    return MonitorView(
        title="recognition training   live dog",
        peer_curve=session.peer_curve[-400:],
        other_curve=session.other_curve[-400:],
        proto_self=rec.proto[0],
        proto_other=rec.proto[1],
        n_self=rec.n_self,
        n_other=rec.n_other,
        mode_label="robot   hebb comparison   MB does not drive",
        learner="hebb",
        **common,
    )


def run_headless(session: LiveSession, base: str, frames: int) -> dict:
    try:
        cam_b = fetch_bytes(f"{base}/camera.jpg", timeout=3.0)
        lid_b = fetch_bytes(f"{base}/lidar.jpg", timeout=3.0)
    except Exception as exc:
        raise SystemExit(offline_message(base, exc)) from exc
    camera = decode_image_bytes(cam_b)
    lidar = decode_image_bytes(lid_b)
    for _ in range(max(1, frames)):
        session.on_frame(camera, lidar)
        cam_b = fetch_bytes(f"{base}/camera.jpg", timeout=3.0)
        lid_b = fetch_bytes(f"{base}/lidar.jpg", timeout=3.0)
        camera = decode_image_bytes(cam_b)
        lidar = decode_image_bytes(lid_b)
    session.save()
    out = {
        "frames": frames,
        "learner": session.learner_kind,
        "dan": session.dan,
        "likeness": session.last_like,
        "state": str(session.state_path),
    }
    if session.mb is not None:
        out.update(
            {
                "drift": session.mb.last_drift,
                "n_pam": session.mb.n_pam,
                "n_ppl1": session.mb.n_ppl1,
                "kc_on": session.mb.last_kc_on,
            }
        )
    elif session.recognizer is not None:
        out.update({"n_self": session.recognizer.n_self, "n_other": session.recognizer.n_other})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live Go2 recognition trainer (sensors + manual UDP teleop)")
    p.add_argument("--robot-ip", default=DEFAULT_IP)
    p.add_argument("--preview-port", type=int, default=DEFAULT_PREVIEW)
    p.add_argument("--udp-host", default=DEFAULT_UDP_HOST)
    p.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    p.add_argument("--state", type=Path, default=DEFAULT_STATE)
    p.add_argument("--load", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=5)
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--reset-lidar", action="store_true", help="GET /lidar/reset once before training")
    p.add_argument("--learner", choices=("mb", "hebb"), default="mb")
    p.add_argument("--dan", choices=("teacher", "familiarity"), default="teacher")
    p.add_argument("--eta", type=float, default=0.2)
    p.add_argument("--npz", type=Path, default=None)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    base = preview_base(args.robot_ip, args.preview_port)
    if args.reset_lidar:
        try:
            fetch_bytes(f"{base}/lidar/reset", timeout=3.0)
        except Exception as exc:
            raise SystemExit(f"Cannot reach {base}/lidar/reset ({exc})") from exc
    session = LiveSession(
        args.state,
        learner=args.learner,
        dan=args.dan,
        eta=args.eta,
        npz=args.npz,
        seed=args.seed,
    )
    if args.load:
        session.load()
    if args.headless:
        summary = run_headless(session, base, args.max_frames)
        print(json.dumps(summary, indent=2))
        return 0
    link = Go2CommandLink(args.udp_host, args.udp_port)
    pull = PreviewPull(base)
    pull.start()
    run_live_gui(session, pull, link, seconds=args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
