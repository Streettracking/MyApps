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
from .lidar_fresh import DEFAULT_LIDAR_REFRESH, FreshWindow, mismatch_warning
from .map_marks import MarkLayer
from .mb_train import MbTrainer, default_npz
from .recognize import ConspecificRecognizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "logs" / "mb_train_state.npz"
DEFAULT_IP = "192.168.35.213"
DEFAULT_PREVIEW = 8088
DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 5451

LIVE_KEYS = (
    "стрелки ход  T лакомство  X наказание  G вспышка  B звук  C/V лидар  "
    "Space  E-STOP  D/N P R S L F12"
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
        self.scan = None
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
            scan = _fetch_scan(self.base)
            with self._lock:
                self.camera = cam
                self.lidar = lid
                self.scan = scan
                self.error = ""
                self.frame_id += 1
            time.sleep(0.05)

    def latest(self):
        with self._lock:
            return self.frame_id, self.camera, self.lidar, self.scan, self.error


def _fetch_scan(base: str):
    """Optional v2 map geometry. A missing endpoint leaves the JPEG cones."""
    try:
        raw = fetch_bytes(f"{base}/lidar/scan.json", timeout=0.35)
        scan = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return scan if isinstance(scan, dict) else None


def request_lidar_reset(base: str, log, fresh: FreshWindow | None = None, quiet: bool = False) -> None:
    """GET /lidar/reset off the teleop thread. Quiet calls are the periodic refresh."""
    if not quiet:
        log("сброс лидара…")

    def work() -> None:
        now = time.monotonic()
        try:
            fetch_bytes(f"{base}/lidar/reset", timeout=2.0)
        except Exception as exc:
            tell = True if fresh is None else fresh.ack(False, now)
            if tell or not quiet:
                log(f"сброс лидара не удался ({exc})")
            return
        if fresh is not None:
            fresh.ack(True, now)
        if not quiet:
            log("лидар сброшен")

    threading.Thread(target=work, name="lidar-reset", daemon=True).start()


def service_lidar(session: "LiveSession", lidar, scan, now: float, base: str):
    """Pick the lidar image for the brain and the one to draw.

    Returns ``(brain_lidar, brain_scan, display, hold_message)``.
    """
    window = session.fresh
    if not session.lidar_fresh_on:
        return lidar, scan, lidar, ""
    if window.phase == "idle":
        window.configure(session.lidar_interval)
        window.start(now)
    window.push(lidar, now, scan)
    if window.poll_reset(now):
        request_lidar_reset(base, session._log, window, quiet=True)
    if window.committed is None:
        return None, None, None, "набор свежего лидара, мозг ждёт первый полный кадр"
    return window.committed, window.committed_scan, window.committed, ""


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
        lidar_refresh: float = DEFAULT_LIDAR_REFRESH,
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
        self._was_rec = False
        self._last_rec_log = -10.0
        self._last_tick = time.monotonic()
        self._prev_like = 0.0
        self._last_spike = -10.0
        self._last_metric = 0.0
        self._last_ckpt = 0.0
        self._prev_camera = None
        self._prev_near = None
        self._last_feat = None
        self.pending_teach: str | None = None
        self.lidar_interval = float(lidar_refresh) if float(lidar_refresh) > 0 else DEFAULT_LIDAR_REFRESH
        self.lidar_fresh_on = float(lidar_refresh) > 0
        self.fresh = FreshWindow(self.lidar_interval if self.lidar_fresh_on else 0.0)
        self.lidar_warning = ""
        self.lidar_hold = ""
        self._lidar_warn_logged = ""
        self._weights_from_disk = False
        self._lidar_saved: float | None = None
        self.marks = MarkLayer()
        self.mb: MbTrainer | None = None
        self.recognizer: ConspecificRecognizer | None = None
        if learner == "mb":
            self.mb = MbTrainer(npz or default_npz(), seed=seed, eta=eta, dan=dan)
            self._log("грибовидное тело  учитель — клавиша T  мозг собаку не ведёт")
            self._log(self._lidar_intro())
        else:
            self.recognizer = ConspecificRecognizer()
            self._log("слой сравнения  мозг собаку не ведёт")
            self._log(self._lidar_intro())
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

    def teach_current(self, kind: str | None) -> None:
        """PAM/PPL1 on the frame already shown. Does not move the confidence baseline."""
        if self.mb is None or self.mb.last_fwd is None or kind is None:
            return
        self.treat_flash = kind == "pam"
        self.mb.teach(self.mb.last_fwd, kind if self.dan == "teacher" else None, self.now())
        self._log_teach(kind)

    def _log_teach(self, kind: str | None) -> None:
        if self.mb is None:
            return
        if kind == "pam" and (self.mb.n_pam <= 2 or self.mb.n_pam % 15 == 0):
            self._log(f"лакомство PAM  #{self.mb.n_pam}")
        elif kind == "ppl1" and (self.mb.n_ppl1 <= 2 or self.mb.n_ppl1 % 15 == 0):
            self._log(f"наказание PPL1  #{self.mb.n_ppl1}")

    def on_frame(self, camera: np.ndarray, lidar: np.ndarray, teach: str | None = None, scan: dict | None = None) -> None:
        feat, ego, near = features_from_frames(camera, lidar, self._prev_camera, self._prev_near)
        self._prev_camera = camera
        self._prev_near = near
        self._last_feat = feat
        self._last_scan = scan
        self.treat_flash = False
        # Operator D/N is intentionally not an argument of teaching or confidence.
        if self.mb is not None:
            self.mb.learn = self.learn
            fwd = self.mb.forward(feat)
            value = self.mb.last_readout
            conf = self.mb.observe(feat, value)
            now_m = time.monotonic()
            self.mb.progress.tick(now_m - self._last_tick, self.learn)
            self._last_tick = now_m
            self.mb.progress.note_label(self.operator, value, conf.recognized, conf.ready)
            kind = teach if self.dan == "teacher" else None
            if kind == "pam":
                self.treat_flash = True
            self.mb.teach(fwd, kind, self.now())
            self._log_teach(kind)
            if (
                conf.ready
                and conf.recognized
                and not self._was_rec
                and self.now() - self._last_rec_log > 2.0
            ):
                self._log(f"узнаю сородича  {conf.percent:.0f}%")
                self._last_rec_log = self.now()
            self._was_rec = bool(conf.recognized) if conf.ready else False
            self.last_like = float(value)
            self.last_match = float(value)
            self.marks.consider(
                self.mb,
                feat,
                self.now(),
                recognized=bool(conf.ready and conf.recognized),
                mode="live",
                scan=scan,
            )
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
            self._log(f"сохранено  {self.state_path.name}")
            self._last_ckpt = self.now()

    def _effective_refresh(self) -> float:
        return self.lidar_interval if self.lidar_fresh_on else 0.0

    def _lidar_intro(self) -> str:
        if self.lidar_fresh_on:
            return (
                f"свежий лидар каждые {self.lidar_interval:.1f} с: "
                "в мозг идёт кадр перед сбросом, пустые кадры после сброса не идут"
            )
        return "лидар копится на карте сервера"

    def _lidar_caption(self) -> str:
        if self.lidar_fresh_on:
            return f"свежий лидар {self.lidar_interval:.1f} с"
        return "лидар копится"

    def _sync_lidar_warning(self) -> None:
        if not self._weights_from_disk:
            self.lidar_warning = ""
            return
        full = mismatch_warning(self._lidar_saved, self._effective_refresh())
        self.lidar_warning = "веса с другой карты лидара — R сброс, файл не стираю" if full else ""
        if full and full != self._lidar_warn_logged:
            self._log(full)
            self._lidar_warn_logged = full

    def toggle_lidar_fresh(self, now: float) -> None:
        self.lidar_fresh_on = not self.lidar_fresh_on
        if self.lidar_fresh_on and self.lidar_interval <= 0:
            self.lidar_interval = DEFAULT_LIDAR_REFRESH
        if self.lidar_fresh_on:
            self.fresh.configure(self.lidar_interval)
            self.fresh.start(now)
        else:
            self.fresh.configure(0.0)
        self._log(self._lidar_intro())
        self._sync_lidar_warning()

    def save(self) -> None:
        if self.mb is not None:
            self.mb.lidar_refresh = self._effective_refresh()
            self.mb.save(self.state_path)
        elif self.recognizer is not None:
            self.recognizer.save(self.state_path)
        self._log(f"сохранено  {self.state_path.name}")

    def load(self) -> None:
        if not Path(self.state_path).is_file():
            raise FileNotFoundError(self.state_path)
        if self.mb is not None:
            self.mb.load(self.state_path)
            self.dan = self.mb.dan
            self._was_rec = bool(self.mb.conf.recognized)
            self._weights_from_disk = True
            self._lidar_saved = self.mb.saved_lidar_refresh
            self._sync_lidar_warning()
            self._log(f"загружены веса KC→MBON  дрейф {self.mb.last_drift:.1f}  лакомств всего {self.mb.progress.base_pam}")
        elif self.recognizer is not None:
            self.recognizer.load(self.state_path)
            self._log(f"загружен прототип  n={self.recognizer.n_self}")

    def reset(self) -> None:
        if self.mb is not None:
            self.mb.reset()
            self._was_rec = False
            self.marks.alive.clear()
            self._weights_from_disk = False
            self.lidar_warning = ""
            self._log("веса и счётчики обучения сброшены")
        elif self.recognizer is not None:
            self.recognizer.reset()
            self._log("прототип сброшен")
        self._prev_camera = None
        self._prev_near = None

    def label_text(self) -> str:
        if self.operator == "dog":
            return "метка D держится — только панель точности, не DAN"
        if self.operator == "none":
            return "метка N держится — только панель точности, не DAN"
        return "D — собака в кадре, N — нет. Метка только для панели."


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
                session._log("метка D — только для панели, в обучение не входит")
            elif inp.label == "none":
                session.operator = "none"
                session._log("метка N — только для панели, в обучение не входит")
            if inp.beep_toggle:
                session._log("звук включён" if mon.beep_on else "звук выключен")
            if inp.pause_learn:
                session.learn = not session.learn
                session._log("обучение на паузе" if not session.learn else "обучение продолжается")
            if inp.reset:
                session.reset()
            if inp.lidar_reset:
                if session.lidar_fresh_on:
                    session.fresh.manual_clear()
                request_lidar_reset(
                    pull.base,
                    session._log,
                    session.fresh if session.lidar_fresh_on else None,
                    quiet=False,
                )
            if inp.lidar_toggle:
                session.toggle_lidar_fresh(time.monotonic())
            if inp.flash_toggle:
                if session.learner_kind != "mb":
                    mon.flash_open = False
                    session._log("вспышка обучения только у грибовидного тела")
                else:
                    session._log("схема обучения открыта" if mon.flash_open else "схема обучения скрыта")
            if inp.save:
                session.save()
            if inp.load:
                try:
                    session.load()
                except FileNotFoundError:
                    session._log(f"нет файла {session.state_path.name}, продолжаем с текущими весами")
            apply_teleop(link, inp, tele, time.monotonic(), session._log)
            frame_id, camera, lidar, scan, error = pull.latest()
            brain_lidar, brain_scan, disp_lidar, hold = service_lidar(
                session, lidar, scan, time.monotonic(), pull.base
            )
            session.lidar_hold = hold
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
            waiting = session.lidar_fresh_on and brain_lidar is None
            if new_frame:
                if waiting:
                    if teach:
                        session.pending_teach = teach
                else:
                    teach_frame = teach or session.pending_teach
                    session.pending_teach = None
                    session.on_frame(
                        camera,
                        brain_lidar if brain_lidar is not None else lidar,
                        teach=teach_frame,
                        scan=brain_scan if session.lidar_fresh_on else scan,
                    )
                seen_frame = frame_id
            elif teach_now and not waiting:
                session.teach_current(teach)
            elif teach_now and waiting:
                session.pending_teach = teach
            view = _live_view(session, camera, disp_lidar, error, link, inp.focused)
            mon.draw(view)
            if inp.screenshot:
                dest = ROOT / "logs" / "monitor_shot.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                mon.save_screenshot(str(dest))
                session._log(f"снимок  {dest.name}")
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
            "Arrows drive   C reset   V fresh   Space stop   -/+ stand   E E-STOP   D/N   P R S L F12 Esc"
        ),
        lidar_mode=session._lidar_caption(),
        lidar_hold=session.lidar_hold,
        lidar_warning=session.lidar_warning,
        lidar_fresh_on=session.lidar_fresh_on,
        learn_flash=None if session.mb is None else session.mb.flash,
        mb_layout=None if session.mb is None else session.mb.layout,
    )
    if session.mb is not None:
        caption = "сырой выход: подход − избегание" if session.dan == "teacher" else "сырой выход: минус новизна"
        prog = session.mb.progress
        conf = session.mb.conf
        mode = "робот · лакомство T · мозг не рулит" if session.dan == "teacher" else "робот · знакомство · мозг не рулит"
        return MonitorView(
            title="тренировка узнавания   живая собака",
            peer_curve=_scaled(session, session.peer_curve),
            other_curve=_scaled(session, session.other_curve),
            mode_label=mode,
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
            recognized=conf.recognized,
            confidence=conf.percent,
            confidence_ready=conf.ready,
            session_time=prog.session_time,
            total_time=prog.base_time + prog.session_time,
            total_pam=prog.base_pam + session.mb.n_pam,
            total_ppl1=prog.base_ppl1 + session.mb.n_ppl1,
            session_sep=prog.session_sep(),
            total_sep=prog.total_sep(),
            session_acc=prog.session_acc(),
            total_acc=prog.total_acc(),
            session_labeled=prog.ses_scored,
            total_labeled=prog.base_scored + prog.ses_scored,
            session_novelty=session.mb.n_novelty,
            total_novelty=prog.base_novelty + session.mb.n_novelty,
            marks=session.marks.visible(session.now()),
            **common,
        )
    rec = session.recognizer
    assert rec is not None
    return MonitorView(
        title="тренировка узнавания   живая собака",
        peer_curve=session.peer_curve[-400:],
        other_curve=session.other_curve[-400:],
        proto_self=rec.proto[0],
        proto_other=rec.proto[1],
        n_self=rec.n_self,
        n_other=rec.n_other,
        mode_label="робот · слой сравнения · мозг не рулит",
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
        brain_lidar, brain_scan, _, _ = service_lidar(session, lidar, None, time.monotonic(), base)
        if not (session.lidar_fresh_on and brain_lidar is None):
            session.on_frame(camera, brain_lidar if brain_lidar is not None else lidar, scan=brain_scan)
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


def try_load_live(session: LiveSession) -> None:
    path = Path(session.state_path)
    if not path.is_file():
        session._log(f"нет сохранённого состояния, старт с нуля ({path.name})")
        return
    try:
        session.load()
    except FileNotFoundError:
        session._log(f"нет сохранённого состояния, старт с нуля ({path.name})")


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
    p.add_argument(
        "--lidar-refresh",
        type=float,
        default=DEFAULT_LIDAR_REFRESH,
        help="Seconds between GET /lidar/reset. 0 feeds the accumulating map.",
    )
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
        lidar_refresh=args.lidar_refresh,
    )
    if args.load or Path(args.state).is_file():
        try_load_live(session)
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
