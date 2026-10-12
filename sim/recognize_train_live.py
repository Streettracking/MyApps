"""Live recognition trainer for one Go2.

Sensors: HTTP JPEG preview (camera.jpg, lidar.jpg).
Motion: UDP JSON to the local command server, or, with ``--onboard``,
buttons to the fly brain already running on the dog.

Learning is KC→MBON. T is PAM, X is PPL1. D and N only pick a monitor curve.
Autonomy (A) lets that same readout search and approach. Arrows do not take
over: M or the ПЕРЕХВАТ button does, and A is the only way back.
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

from .approach_watch import ApproachWatch
from .frame_sense import decode_image_bytes, features_from_frames
from .raw_sense import add_recog_camera_only_arg, recog_label
from .go2_udp import Go2CommandLink
from .learn_flash import flash_from_payload
from .lidar_fresh import DEFAULT_LIDAR_REFRESH, FreshWindow, mismatch_warning
from .map_marks import MarkLayer
from .mb_train import MbTrainer, default_npz
from .onboard_link import OnboardLink
from .hemifield import DEFAULT_OVERLAP, format_fly_line
from .pilot import format_eyes_line, phase_label
from .pilot import Pilot, TeachRepeater, clamp_velocity, format_range_line, forward_clearance, scrub_range
from .recognize import ConspecificRecognizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "logs" / "mb_train_state.npz"
DEFAULT_IP = "192.168.35.213"
DEFAULT_PREVIEW = 8088
DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 5451

LIVE_KEYS = (
    "A авто  M перехват  K руль  Y учитель  H рамки  J мозг  F11 экран  "
    "U запись  T/X  P  G  V/C  Space E-STOP  -/+  B D/N R S L"
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


def camera_offline_caption(base: str) -> str:
    """One grey line on the camera card. The long reason goes to the journal."""
    from urllib.parse import urlparse

    text = str(base or "")
    parsed = urlparse(text if "://" in text else "http://%s" % text)
    port = parsed.port or DEFAULT_PREVIEW
    return "камера борта недоступна (%s)" % port


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
        self.camera_jpeg = b""
        self.scan = None
        self.frame_id = 0
        self._cam_jpeg = b""
        self._lid_jpeg = b""
        self._decoded_id = -1
        self._decoded_cam = None
        self._decoded_lid = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="preview", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        """Fetch JPEG bytes only. pygame stays on the thread that owns the window."""
        while not self._stop:
            try:
                cam_b = fetch_bytes(f"{self.base}/camera.jpg", timeout=2.0)
                lid_b = fetch_bytes(f"{self.base}/lidar.jpg", timeout=2.0)
            except Exception as exc:
                with self._lock:
                    self.error = offline_message(self.base, exc)
                time.sleep(0.4)
                continue
            scan = _fetch_scan(self.base)
            with self._lock:
                self._cam_jpeg = bytes(cam_b)
                self._lid_jpeg = bytes(lid_b)
                self.camera_jpeg = bytes(cam_b)
                self.scan = scan
                self.error = ""
                self.frame_id += 1
            time.sleep(0.05)

    def latest(self):
        from .sdl_thread import require_main_thread

        require_main_thread()
        with self._lock:
            frame_id = self.frame_id
            cam_b = bytes(self._cam_jpeg)
            lid_b = bytes(self._lid_jpeg)
            scan = None if self.scan is None else dict(self.scan)
            error = self.error
            jpeg = bytes(self.camera_jpeg)
        if error or not cam_b or not lid_b:
            return frame_id, None, None, scan, error, jpeg
        if frame_id != self._decoded_id:
            try:
                cam = decode_image_bytes(cam_b)
                lid = decode_image_bytes(lid_b)
            except Exception as exc:
                message = offline_message(self.base, exc)
                with self._lock:
                    if self.frame_id == frame_id:
                        self.error = message
                return frame_id, None, None, scan, message, jpeg
            self._decoded_id = frame_id
            self._decoded_cam = cam
            self._decoded_lid = lid
        camera = np.array(self._decoded_cam, copy=True)
        lidar = np.array(self._decoded_lid, copy=True)
        return frame_id, camera, lidar, scan, error, jpeg


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
        return_auto_s: float = 0.0,
        onboard: bool = False,
        steer: str = "bilateral",
        overlap: float = DEFAULT_OVERLAP,
        camera_only: bool = True,
    ):
        if learner not in ("mb", "hebb"):
            raise ValueError(learner)
        self.learner_kind = learner
        self.dan = dan
        self.state_path = state_path
        self.learn = True
        self.searching = False
        self.target_in_view = False
        self.operator: str | None = None
        self.peer_curve: list[tuple[float, float]] = []
        self.other_curve: list[tuple[float, float]] = []
        self.metric_curve: list[dict] = []
        self.log: deque[str] = deque(maxlen=200)
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
        self.approach = ApproachWatch()
        self._approach_t = 0.0
        self.pilot = Pilot(return_auto_s)
        self.pilot.set_steer(steer if steer in ("bilateral", "sectors") else "bilateral")
        self.overlap = float(overlap)
        self.camera_only = bool(camera_only)
        self.teach_pulse = TeachRepeater()
        self.aim_sector: int | None = None
        self.aim_dist: float | None = None
        self.forward_m: float | None = None
        self.recognized_now = False
        self._hint_logged = ""
        self._learn_block_log = -10.0
        self.eye_l_recognized = False
        self.eye_r_recognized = False
        self.eye_l_confidence = 0.0
        self.eye_r_confidence = 0.0
        self.eye_l_ready = False
        self.eye_r_ready = False
        self.phase_ru = ""
        self.onboard = bool(onboard)
        self.bridge_open = False
        self._bridge_warned = False
        self._server_down_logged = False
        self.remote: dict = {}
        self.remote_flash = None
        self.remote_flash_r = None
        self.layout = None
        self.mb: MbTrainer | None = None
        self.recognizer: ConspecificRecognizer | None = None
        if onboard:
            self.learner_kind = "mb"
            self.learn = False
            self._log("мозг на роботе. Стойка и ход идут на порт 8090, не через python main.py.")
            self._log("автономия и обучение на борту выключены, пока не нажаты A и P")
            return
        if learner == "mb":
            self.mb = MbTrainer(
                npz or default_npz(),
                seed=seed,
                eta=eta,
                dan=dan,
                overlap=overlap,
                camera_only=self.camera_only,
            )
            self._log("грибовидное тело  T учит  A включает поиск сородича")
            self._log(recog_label(self.camera_only))
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

    def _note_aim(self, feat: np.ndarray, recognized: bool) -> None:
        self.recognized_now = bool(recognized)
        radius = self.pilot.self_radius
        self.forward_m = forward_clearance(feat, radius)
        if recognized:
            self.aim_sector = self.marks.aim_sector
            self.aim_dist = scrub_range(self.marks.aim_dist, radius)
        else:
            self.aim_sector = None
            self.aim_dist = None

    def ensure_layout(self) -> None:
        if self.layout is not None or not self.onboard:
            return
        holder = MbTrainer(default_npz(), seed=1, eta=0.2, dan="teacher")
        self.layout = holder.layout
        self._layout_holder = holder

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
            from .kc_pathways import yolo_vector

            boxes = getattr(self, "teacher_boxes", None) or ()
            overlap = float(getattr(self, "overlap", 0.4) or 0.4)
            fwd = self.mb.forward(
                feat,
                yolo_l=yolo_vector(boxes, "L", overlap),
                yolo_r=yolo_vector(boxes, "R", overlap),
            )
            value = self.mb.last_readout
            conf = self.mb.observe(feat, value)
            now_m = time.monotonic()
            dt = now_m - self._last_tick
            self.mb.progress.tick(dt, self.learn)
            self._last_tick = now_m
            self.mb.progress.note_label(self.operator, value, conf.recognized, conf.ready)
            kind = teach if self.dan == "teacher" else None
            stopped = self.pilot.mode == "estop" or bool(self.pilot.held_stop)
            moving = (not stopped) and (
                abs(float(getattr(self, "cmd_x", 0.0))) + abs(float(getattr(self, "cmd_z", 0.0))) > 1e-6
            )
            self.mb.note_motion_hold(stopped, moving, self.now())
            boxes = getattr(self, "teacher_boxes", None) or ()
            self.searching = bool(self.pilot.autonomy) and not boxes
            self.target_in_view = bool(boxes)
            if self.mb.learning_hold or self.searching:
                kind = None
            if kind == "pam":
                self.treat_flash = True
            self.mb.teach(fwd, kind, self.now())
            pulsed = kind in ("pam", "ppl1")
            if not self.searching:
                self.mb.settle(pulsed, pulsed)
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
            seen = bool(conf.ready and conf.recognized)
            self.marks.consider(
                self.mb,
                feat,
                self.now(),
                recognized=seen,
                mode="live",
                scan=scan,
            )
            self._note_aim(feat, seen)
            self._note_approach(dt)
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
            self.recognized_now = False
            self.aim_sector = None
            self.aim_dist = None
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
            self._last_ckpt = self.now()

    def _effective_refresh(self) -> float:
        return self.lidar_interval if self.lidar_fresh_on else 0.0

    def _lidar_intro(self) -> str:
        if self.lidar_fresh_on:
            return f"свежий лидар {self.lidar_interval:.1f} с, пустой кадр после сброса не идёт в мозг"
        return "лидар копится на карте сервера"

    def _lidar_caption(self) -> str:
        from .tabnum import format_lidar_caption

        return format_lidar_caption(self.lidar_fresh_on, self.lidar_interval)

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
        from .tabnum import note_repeated_save

        note_repeated_save(self.log, "%6.1fs" % self.now(), self.state_path.name)

    def _note_approach(self, dt: float) -> None:
        """Time near a conspecific box and near a cube box. Display only."""
        watch = getattr(self, "approach", None)
        if watch is None:
            return
        dog = False
        cube = False
        for box in getattr(self, "teacher_boxes", None) or ():
            kind = str(getattr(box, "kind", "") or "")
            if kind in ("cube", "control", "куб"):
                cube = True
            else:
                dog = True
        drive = None
        if self.mb is not None:
            drive = 0.5 * (float(self.mb.r_l) + float(self.mb.r_r))
        watch.note(float(dt), dog, cube, drive)

    def load(self) -> None:
        if not Path(self.state_path).is_file():
            raise FileNotFoundError(self.state_path)
        if self.mb is not None:
            if not self.mb.load(self.state_path):
                self._weights_from_disk = False
                self._log(self.mb.load_note)
                return
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


def _apply_live_keys(session: LiveSession, inp, onboard: OnboardLink | None) -> None:
    if inp.estop:
        session.pilot.estop()
        if onboard is not None:
            onboard.post("estop")
        else:
            session._log("E-STOP")
    elif inp.stop:
        session.pilot.space()
        if onboard is not None:
            onboard.post("space")
        session._log("стоп")
    if inp.takeover and not inp.estop:
        session.pilot.takeover()
        if onboard is not None:
            onboard.post("takeover")
        session._log("перехват: ручное, автономия сама не вернётся")
    if inp.autonomy_toggle and not inp.estop:
        if session.learner_kind != "mb" and not session.onboard:
            session._log("автономия только у грибовидного тела")
        elif onboard is not None:
            if session.remote.get("autonomy"):
                onboard.post("autonomy_off")
            else:
                onboard.post("autonomy_on")
        elif session.pilot.autonomy:
            session.pilot.stop_auto()
            session._log("автономия выключена")
        elif not session.pilot.start_auto(session.now()):
            session._log("E-STOP держит стоп. M переводит в ручное")
        else:
            session._log("автономия: медленный поиск, пока в кадре пусто, руль %s" % ("билатерально" if session.pilot.steer == "bilateral" else "секторы"))
    if inp.steer_toggle and session.learner_kind == "mb":
        if onboard is not None:
            current = str(session.remote.get("steer") or session.pilot.steer)
            mode = "sectors" if current == "bilateral" else "bilateral"
            onboard.post("steer", mode=mode)
            session._log("руль: %s" % ("билатерально" if mode == "bilateral" else "секторы"))
        else:
            mode = session.pilot.toggle_steer()
            session._log("руль: %s" % ("билатерально" if mode == "bilateral" else "секторы"))
    if inp.pause_learn:
        if onboard is not None:
            onboard.post("learn_off" if session.remote.get("learning") else "learn_on")
        else:
            session.learn = not session.learn
            session._log("обучение выключено, T/X веса не меняют" if not session.learn else "обучение включено")


def _mirror_remote(session: LiveSession, status: dict) -> None:
    if not status:
        return
    session.remote = status
    session.learn = bool(status.get("learning", False))
    session.pilot.mode = str(status.get("mode", session.pilot.mode))
    session.pilot.took_over = bool(status.get("took_over", False))
    session.pilot.phase = str(status.get("phase", "stop"))
    session.pilot.who = str(status.get("who", "никто"))
    session.pilot.hint = str(status.get("hint", ""))
    session.pilot.held_stop = status.get("label") == "СТОП" and status.get("mode") != "estop"
    session.recognized_now = bool(status.get("recognized", False))
    session.last_like = float(status.get("readout") or 0.0)
    session.last_match = session.last_like
    session.remote_flash = flash_from_payload(status.get("flash"))
    session.remote_flash_r = flash_from_payload(status.get("flash_r"))
    if status.get("steer") in ("bilateral", "sectors"):
        session.pilot.steer = str(status["steer"])
    if status.get("overlap") is not None:
        session.overlap = float(status["overlap"])
    side = status.get("last_seen_side") or ""
    if side in ("L", "R"):
        session.pilot.last_seen_side = str(side)
    elif status.get("last_seen_side") is None and "last_seen_side" in status:
        session.pilot.last_seen_side = ""
    if status.get("search_sign") is not None:
        session.pilot.search_sign = 1.0 if float(status["search_sign"]) >= 0 else -1.0
    session.eye_l_recognized = bool(status.get("recognized_L", False))
    session.eye_r_recognized = bool(status.get("recognized_R", False))
    session.eye_l_confidence = float(status.get("confidence_L") or 0.0)
    session.eye_r_confidence = float(status.get("confidence_R") or 0.0)
    session.eye_l_ready = bool(status.get("confidence_L_ready", False))
    session.eye_r_ready = bool(status.get("confidence_R_ready", False))
    session.phase_ru = str(status.get("phase_ru") or "")
    for line in status.get("log") or []:
        if line not in session.log:
            session.log.append(str(line))


def take_teacher_weight(session, inp) -> None:
    """Slider motion. None means the pointer is not on the track."""
    value = getattr(inp, "teacher_weight", None)
    if value is None:
        return
    from .teacher_lead import clamp_weight

    session.teacher_weight = clamp_weight(value)


def _lead_of(session):
    from .teacher_lead import TeacherLead

    lead = getattr(session, "lead", None)
    if lead is None:
        lead = TeacherLead()
        session.lead = lead
    from .teacher_lead import clamp_weight

    lead.weight = clamp_weight(getattr(session, "teacher_weight", 0.0))
    return lead


def _yolo_list(boxes, side: str, overlap: float) -> list:
    from .kc_pathways import yolo_vector

    return [float(v) for v in yolo_vector(boxes, side, overlap)]


def _lead_boxes(session):
    teacher = getattr(session, "teacher", None)
    if teacher is not None:
        boxes = getattr(teacher, "boxes", None)
        if boxes:
            return list(boxes)
    return list(getattr(session, "teacher_boxes", ()) or ())


def refresh_teacher_wish(session, now: float, dist_m, forward_m):
    """Teacher command from the latest boxes. Does not move the dog."""
    lead = _lead_of(session)
    boxes = _lead_boxes(session)
    lead.update_from_boxes(now, boxes, dist_m, forward_m)
    teacher = getattr(session, "teacher", None)
    overlap = float(getattr(session, "overlap", 0.4) or 0.4)
    if teacher is not None and boxes:
        from .yolo_teacher import hemisphere_hits

        hit_l, hit_r = hemisphere_hits(boxes, overlap, conf_min=float(getattr(teacher, "conf", 0.5)), center_only=True)
    else:
        hit_l, hit_r = False, False
    lead.set_hits(hit_l, hit_r)
    return lead


def publish_teacher_aim(session, onboard, now: float, dist_m, forward_m) -> None:
    """Send the teacher's wish to the dog. The dog mixes it with the fly."""
    if onboard is None:
        return
    if float(now) - float(getattr(session, "_aim_post_t", -1.0)) < 0.10:
        return
    lead = refresh_teacher_wish(session, now, dist_m, forward_m)
    teacher = getattr(session, "teacher", None)
    learning = bool(getattr(session, "learn", False))
    if getattr(session, "onboard", False):
        learning = bool((getattr(session, "remote", {}) or {}).get("learning", False))
    teacher_on = bool(teacher is not None and teacher.on and learning)
    session._aim_post_t = float(now)
    boxes = _lead_boxes(session)
    overlap = float(getattr(session, "overlap", 0.4) or 0.4)
    onboard.post(
        "teacher_aim",
        x=float(lead.teacher_x),
        z=float(lead.teacher_z),
        weight=float(lead.weight),
        hit_l=bool(lead.hit_l) and teacher_on,
        hit_r=bool(lead.hit_r) and teacher_on,
        centered=bool(lead.centered) and teacher_on,
        approaching=bool(lead.approaching) and teacher_on,
        lost=bool(lead.lost) and teacher_on,
        too_close=bool(lead.too_close) and teacher_on,
        last_l=bool(lead.last_hit_l) and teacher_on,
        last_r=bool(lead.last_hit_r) and teacher_on,
        yolo_l=_yolo_list(boxes if teacher_on else (), "L", overlap),
        yolo_r=_yolo_list(boxes if teacher_on else (), "R", overlap),
    )


def blend_local_command(session, cmd, now: float, estop: bool, autonomy: bool, dist_m, forward_m):
    """Replace an autonomy command when the teacher weight is above zero.

    Weight 0 returns the fly command unchanged. Outcome pulses are applied
    here for a local brain. The onboard brain does that itself. A command
    match is recorded for status and does not teach.
    """
    from .pilot import DriveCommand
    from .teacher_lead import apply_guards, attach_recognition_stimuli, commands_match, mix_velocity, observe_recognition

    lead = refresh_teacher_wish(session, now, dist_m, forward_m)
    lead.fly_x = float(cmd.x)
    lead.fly_z = float(cmd.z)
    matched = commands_match(cmd.x, cmd.z, lead.teacher_x, lead.teacher_z)
    teacher = getattr(session, "teacher", None)
    teacher_on = bool(teacher is not None and getattr(teacher, "on", False))
    learning = bool(getattr(session, "learn", False))
    session._outcome_l = None
    session._outcome_r = None
    stopped = bool(estop) or session.pilot.mode == "estop" or bool(session.pilot.held_stop)
    moving = (not stopped) and (abs(float(cmd.x)) + abs(float(cmd.z)) > 1e-6)
    if autonomy and lead.weight > 0.005 and not stopped:
        mixed_x, mixed_z = mix_velocity(cmd.x, cmd.z, lead.teacher_x, lead.teacher_z, lead.weight)
        out_x, out_z, _held = apply_guards(mixed_x, mixed_z, False, forward_m)
        moving = abs(float(out_x)) + abs(float(out_z)) > 1e-6
    mb = getattr(session, "mb", None)
    if mb is not None:
        mb.note_motion_hold(stopped, moving, float(now))
    if autonomy and not estop:
        lead.note(now, matched)
        mb = getattr(session, "mb", None)
        if mb is not None:
            attach_recognition_stimuli(mb.stimuli)
            observe_recognition(mb.stimuli, lead, teacher_on)
            kind_l, kind_r = mb.reinforce_stimuli(float(now), learning=learning)
            if kind_l or kind_r:
                session._outcome_l = kind_l
                session._outcome_r = kind_r
    session.lead_match = lead.fraction(now)
    if not autonomy or lead.weight <= 0.005:
        return cmd
    mixed_x, mixed_z = mix_velocity(cmd.x, cmd.z, lead.teacher_x, lead.teacher_z, lead.weight)
    out_x, out_z, stopped = apply_guards(mixed_x, mixed_z, bool(estop), forward_m)
    who = "учитель" if lead.weight >= 0.995 else "смешанно"
    return DriveCommand(out_x, out_z, stopped, cmd.phase, who, cmd.hint)


def _drive_udp(session: LiveSession, link: Go2CommandLink, inp, state: dict, now: float, frames_ok: bool) -> str:
    if session.mb is None:
        eye_l, eye_r = None, None
    else:
        eye_l, eye_r = session.mb.eye_recognized()
    cmd = session.pilot.command(
        session.now(),
        (inp.steer_x, inp.steer_z),
        focused=inp.focused,
        frames_ok=frames_ok,
        link_ok=True,
        recognized=session.recognized_now and session.learner_kind == "mb",
        sector=session.aim_sector,
        dist_m=session.aim_dist,
        forward_m=session.forward_m,
        r_l=0.0 if session.mb is None else float(session.mb.r_l),
        r_r=0.0 if session.mb is None else float(session.mb.r_r),
        recognized_l=eye_l,
        recognized_r=eye_r,
    )
    from .search_reflex import cover_command, yolo_boxes_seen

    if session.pilot.autonomy and session.learner_kind == "mb" and cmd.phase != "stop":
        seen = yolo_boxes_seen(getattr(session, "teacher_boxes", None))
        cmd, session.searching = cover_command(
            cmd,
            seen,
            0.0 if session.mb is None else float(session.mb.r_l),
            0.0 if session.mb is None else float(session.mb.r_r),
            session.aim_dist,
            session.forward_m,
        )
        session.pilot.phase = cmd.phase
        session.target_in_view = bool(seen)
    else:
        session.searching = False
    cmd = blend_local_command(
        session,
        cmd,
        session.now(),
        bool(inp.estop or session.pilot.mode == "estop"),
        bool(session.pilot.autonomy),
        session.aim_dist,
        session.forward_m,
    )
    if cmd.hint and cmd.hint != session._hint_logged:
        session._log(cmd.hint)
        session._hint_logged = cmd.hint
    elif not cmd.hint:
        session._hint_logged = ""
    if inp.estop:
        link.emergency_stop()
    force = bool(inp.estop or inp.stop or inp.takeover or not inp.focused)
    if cmd.stop:
        if state["moving"] or force:
            link.stop()
            state["moving"] = False
    elif (not state["moving"]) or (now - state["last"] >= 0.10):
        link.move_axes(cmd.x, cmd.z)
        state["moving"] = True
        state["last"] = now
    session.cmd_x = float(cmd.x)
    session.cmd_z = float(cmd.z)
    if inp.focused and not inp.estop:
        if inp.stand_up:
            link.stand_up()
            session._log("StandUp")
        if inp.stand_down:
            link.stand_down()
            session._log("StandDown")
    return link.last_command or _phase_name(cmd.phase, session.pilot.steer, session.pilot.search_sign)


def _note_robot_server(session: LiveSession, link: Go2CommandLink, state: dict, now: float) -> None:
    """Say once when the robot window's UDP port has nobody listening.

    Onboard mode uses port 8090 and does not call this. Sim does not either.
    """
    if now - state.get("server_note", 0.0) < 1.0:
        return
    state["server_note"] = now
    from .go2_control import robot_server_message
    from .onboard_link import udp_bridge_listening

    listening = bool(udp_bridge_listening(link.host, int(link.port)))
    text, told = robot_server_message(
        listening,
        bool(getattr(session, "_server_down_logged", False)),
        link.host,
        int(link.port),
    )
    session.bridge_open = listening
    session._server_down_logged = told
    if text:
        session._log(text)


def _note_bridge(session: LiveSession, now: float, state: dict, link_ok: bool = False) -> None:
    """Warn only when this window is linked and ``python main.py`` holds UDP 5451.

    A busy port is not that process. The go2 server, or a window with no
    onboard link, does not get the two-source line.
    """
    if now - state.get("bridge", 0.0) < 1.0:
        return
    state["bridge"] = now
    from .onboard_link import main_py_listening

    proven = bool(link_ok) and main_py_listening()
    session.bridge_open = proven
    if proven and not session._bridge_warned:
        session._log("два источника команд: это окно и python main.py на UDP 5451")
        session._bridge_warned = True
    elif not proven:
        session._bridge_warned = False


def _drive_onboard(session: LiveSession, onboard: OnboardLink, inp, state: dict, now: float) -> None:
    """Buttons and, only after takeover, manual axes. Never UDP."""
    if now - state["poll"] >= 0.10:
        onboard.poll()
        state["poll"] = now
    # post() already wrote onboard.status. Mirror it on this frame, including
    # when the poll interval has not elapsed, so M sends op=manual from
    # took_over and mode=manual without waiting for the next GET.
    _mirror_remote(session, onboard.status)
    _note_bridge(session, now, state, bool(onboard.ok))
    remote = session.remote or {}
    publish_teacher_aim(
        session,
        onboard,
        now,
        remote.get("distance_m"),
        remote.get("forward_m"),
    )
    if inp.stand_up:
        onboard.post("stand_up")
        session._log("StandUp")
    if inp.stand_down:
        onboard.post("stand_down")
        session._log("StandDown")
    if inp.recovery:
        onboard.post("recovery_stand")
        session._log("RecoveryStand")
    autonomy = bool(session.remote.get("autonomy"))
    grabbed = bool(session.remote.get("took_over")) and session.remote.get("mode") == "manual"
    if inp.focused and grabbed and not autonomy:
        ax = float(inp.steer_x)
        az = float(inp.steer_z)
        if abs(ax) + abs(az) > 0:
            if now - state["manual"] >= 0.10:
                x, z = clamp_velocity(0.4 * ax, az if az else 0.0)
                if az > 0:
                    z = 1.0
                elif az < 0:
                    z = -1.0
                x, z = clamp_velocity(x, z)
                onboard.post("manual", x=x, z=z)
                state["manual"] = now
                state["axes"] = True
        elif state["axes"]:
            onboard.post("manual", x=0.0, z=0.0)
            state["manual"] = now
            state["axes"] = False
    elif state["axes"]:
        state["axes"] = False
    if autonomy and inp.focused and abs(float(inp.steer_x)) + abs(float(inp.steer_z)) > 0:
        session.pilot.hint = "нажми ПЕРЕХВАТ"
        if session._hint_logged != session.pilot.hint:
            session._log(session.pilot.hint)
            session._hint_logged = session.pilot.hint
    elif not autonomy:
        session._hint_logged = ""


def _phase_name(phase: str, steer: str = "bilateral", search_sign: float = 1.0) -> str:
    return phase_label(phase, steer, search_sign)


def _offer_live_frame(session: LiveSession, rec, marks, camera, camera_jpeg: bytes, onboard: OnboardLink | None) -> None:
    """Queue the preview JPEG. Onboard still reads :8088 on the laptop."""
    from .frame_record import fingerprint, make_meta

    if onboard is not None:
        remote = session.remote or {}
        source = "onboard"
        mode = "auto" if remote.get("autonomy") else "manual"
        speed = abs(float(remote.get("x") or 0.0))
        yaw = float(remote.get("z") or 0.0)
        readout = remote.get("readout")
        confidence = remote.get("confidence")
        ready = bool(remote.get("confidence_ready"))
        recognized = bool(remote.get("recognized"))
        sector = remote.get("sector")
        r_l = remote.get("r_l")
        r_r = remote.get("r_r")
    else:
        source = "robot"
        mode = "auto" if session.pilot.autonomy else "manual"
        speed = abs(float(getattr(session, "cmd_x", 0.0)))
        yaw = float(getattr(session, "cmd_z", 0.0))
        mb = session.mb
        conf = None if mb is None else mb.conf
        readout = None if mb is None else float(mb.last_readout)
        confidence = None if conf is None else float(conf.percent)
        ready = False if conf is None else bool(conf.ready)
        recognized = bool(session.recognized_now)
        sector = session.aim_sector
        r_l = None if mb is None else float(mb.r_l)
        r_r = None if mb is None else float(mb.r_r)
    rec.offer(
        camera_jpeg,
        make_meta(
            marks,
            session.now(),
            source=source,
            mode=mode,
            speed_m_s=speed,
            yaw_rad_s=yaw,
            readout=None if readout is None else float(readout),
            confidence=None if confidence is None else float(confidence),
            confidence_ready=ready,
            recognized=recognized,
            sector=None if sector is None else int(sector),
            r_l=None if r_l is None else float(r_l),
            r_r=None if r_r is None else float(r_r),
        ),
        fingerprint(camera),
    )


def _learning_now(session) -> bool:
    if getattr(session, "onboard", False):
        return bool(getattr(session, "remote", {}).get("learning", False))
    return bool(getattr(session, "learn", False))


def _teacher_counts(session: LiveSession, teacher) -> str:
    """Local teacher tallies, unless this window is the onboard brain.

    The simulator and the robot window do not carry ``onboard``. Only a
    session that actually has the flag set reads the dog's counters.
    """
    from .yolo_teacher import counts_line

    remote = {}
    if getattr(session, "onboard", False):
        remote = getattr(session, "remote", {}) or {}
    if isinstance(remote, dict) and "teacher_pam_l" in remote:
        return counts_line(
            int(remote.get("teacher_pam_l") or 0),
            int(remote.get("teacher_pam_r") or 0),
            int(remote.get("teacher_ppl1_l") or 0),
            int(remote.get("teacher_ppl1_r") or 0),
        )
    return counts_line(teacher.pam_l, teacher.pam_r, teacher.ppl1_l, teacher.ppl1_r)


def _flash_show(session, side: str, now: float) -> str:
    kind = getattr(session, "teacher_flash_" + side, "")
    until = float(getattr(session, "teacher_flash_%s_until" % side, -1.0) or -1.0)
    if kind in ("pam", "ppl1") and float(now) <= until + 1e-9:
        return str(kind)
    return ""


def _drive_teacher(session: LiveSession, jpeg: bytes | None, frame_id, state: dict, operator_busy: bool, onboard: OnboardLink | None) -> None:
    """Look when learning is on. Teach only while Y is on. Never sends Move.

    ``recognized`` is read here, on this tick, from the same bits the plaques
    paint. It is not the bit that was true when the JPEG left for YOLO.
    """
    teacher = getattr(session, "teacher", None)
    if teacher is None or session.learner_kind != "mb":
        return
    from .yolo_teacher import FLASH_S, plaque_recognized, send_teach, teach_phrase

    learning = _learning_now(session)
    teacher.note_learning(learning)
    if learning and jpeg and teacher.wants_frame(learning) and state.get("frame") != frame_id:
        teacher.offer_frame(jpeg)
        state["frame"] = frame_id
    rec_l, rec_r, overlap = plaque_recognized(
        getattr(session, "mb", None),
        getattr(session, "eye_l_recognized", False),
        getattr(session, "eye_r_recognized", False),
        getattr(session, "overlap", 0.4),
    )
    now = session.now() if hasattr(session, "now") else float(session.world.t)
    kind_l, kind_r = teacher.collect(now, overlap, rec_l, rec_r, learning, operator_busy)
    sent_l, sent_r = kind_l, kind_r
    if sent_l and sent_l == getattr(session, "_outcome_l", None):
        sent_l = None
    if sent_r and sent_r == getattr(session, "_outcome_r", None):
        sent_r = None
    session.teacher_boxes = teacher.visible_boxes(learning)
    session.yolo_state = teacher.status_text(learning)
    session.teacher_counts = _teacher_counts(session, teacher)
    session.teacher_skips = teacher.skips_line()
    if sent_l in ("pam", "ppl1") or sent_r in ("pam", "ppl1"):
        if onboard is not None:
            send_teach(onboard, sent_l, sent_r)
        elif session.mb is not None:
            session.mb.teach_sides(sent_l, sent_r, now)
    if kind_l in ("pam", "ppl1") or kind_r in ("pam", "ppl1"):
        for side, kind in (("L", kind_l), ("R", kind_r)):
            phrase = teach_phrase(side, kind)
            if not phrase:
                continue
            session._log(phrase)
            attr = "l" if side == "L" else "r"
            setattr(session, "teacher_flash_" + attr, kind)
            setattr(session, "teacher_flash_%s_until" % attr, now + FLASH_S)
    session.teacher_flash_l_show = _flash_show(session, "l", now)
    session.teacher_flash_r_show = _flash_show(session, "r", now)


def run_live_gui(
    session: LiveSession,
    pull: PreviewPull,
    link: Go2CommandLink | None,
    seconds: float = 0.0,
    onboard: OnboardLink | None = None,
) -> None:
    from .train_monitor import MonitorView, TrainMonitor

    from .frame_record import FrameRecorder, OperatorMarks, note_operator, weak_label_now

    from .yolo_teacher import TeacherRuntime

    mon = TrainMonitor("Go2 recognition trainer", fullscreen=bool(getattr(session, "start_fullscreen", False)))
    if session.learner_kind == "mb":
        session.teacher = TeacherRuntime(
            getattr(session, "teacher_url", "http://127.0.0.1:8091"),
            conf=float(getattr(session, "teacher_conf", 0.5)),
            rate=float(getattr(session, "teacher_rate", 2.0)),
        )
    else:
        session.teacher = None
    from .yolo_teacher import OwnedTeacher

    session._owned_teacher = OwnedTeacher()
    if session.teacher is not None:
        note = session._owned_teacher.start(getattr(session, "teacher_url", "http://127.0.0.1:8091"))
        if note:
            session._log(note)
    rec = FrameRecorder(ROOT / "logs" / "yolo_frames", fps=float(getattr(session, "rec_fps", 2.0)))
    marks = OperatorMarks()
    tele = {
        "moving": False,
        "last": 0.0,
        "latched": False,
        "poll": 0.0,
        "manual": 0.0,
        "axes": False,
        "bridge": 0.0,
    }
    seen_frame = 0
    teacher_state = {"frame": None, "log": "", "log_t": -10.0}
    announced = False
    command_name = "—"
    try:
        while True:
            inp = mon.pump()
            if inp.quit:
                break
            if inp.record_toggle:
                on = rec.toggle()
                _path = rec.stats()[3]
                session._log("запись кадров %s  (ноутбук, :8088)" % (_path if on else "выключена"))
            if inp.label == "dog":
                session.operator = "dog"
                session._log("метка D — только для панели, в обучение не входит")
            elif inp.label == "none":
                session.operator = "none"
                session._log("метка N — только для панели, в обучение не входит")
            note_operator(marks, session.now(), inp)
            if inp.beep_toggle:
                session._log("звук включён" if mon.beep_on else "звук выключен")
            if inp.fullscreen_toggle:
                session._log("полный экран" if mon.fullscreen else "окно")
            if inp.teacher_toggle and session.teacher is not None:
                session._log(session.teacher.toggle_teacher())
            if inp.boxes_toggle and session.teacher is not None:
                hidden = session.teacher.toggle_boxes(_learning_now(session))
                if hidden:
                    session._log(hidden)
            if inp.reset and not session.onboard:
                session.reset()
            if getattr(inp, "approach_reset", False):
                session.approach.reset()
                session._log("новый заход")
            if inp.lidar_reset and not session.onboard:
                if session.lidar_fresh_on:
                    session.fresh.manual_clear()
                request_lidar_reset(
                    pull.base,
                    session._log,
                    session.fresh if session.lidar_fresh_on else None,
                    quiet=False,
                )
            elif inp.lidar_reset and session.onboard:
                session._log("свежий лидар ведёт борт, это окно его не сбрасывает")
            if inp.lidar_toggle and not session.onboard:
                session.toggle_lidar_fresh(time.monotonic())
            if inp.flash_toggle:
                if session.learner_kind != "mb":
                    mon.flash_open = False
                    session._log("вспышка обучения только у грибовидного тела")
                elif session.onboard and mon.flash_open:
                    session.ensure_layout()
                    session._log("схема обучения открыта")
                else:
                    session._log("схема обучения открыта" if mon.flash_open else "схема обучения скрыта")
            if inp.save:
                if onboard is not None:
                    onboard.post("save")
                else:
                    session.save()
            if inp.load and not session.onboard:
                try:
                    session.load()
                except FileNotFoundError:
                    session._log(f"нет файла {session.state_path.name}, продолжаем с текущими весами")
            now = time.monotonic()
            take_teacher_weight(session, inp)
            frame_id, camera, lidar, scan, error, camera_jpeg = pull.latest()
            if session.onboard:
                disp_lidar = lidar
                session.lidar_hold = ""
                brain_lidar = None
                brain_scan = None
            else:
                brain_lidar, brain_scan, disp_lidar, hold = service_lidar(
                    session, lidar, scan, now, pull.base
                )
                session.lidar_hold = hold
            frames_ok = camera is not None and lidar is not None and not error
            if error and not announced:
                session._log(error)
                announced = True
            if error:
                error = camera_offline_caption(pull.base)
            if not error:
                announced = False
            _apply_live_keys(session, inp, onboard)
            if onboard is not None:
                _drive_onboard(session, onboard, inp, tele, now)
                command_name = str(session.remote.get("phase_ru") or _phase_name(str(session.remote.get("phase") or "—"), session.pilot.steer, session.pilot.search_sign))
            elif link is not None:
                _note_robot_server(session, link, tele, now)
                command_name = _drive_udp(session, link, inp, tele, now, frames_ok)
            teach = session.teach_pulse.poll(
                session.now(),
                inp.treat,
                inp.punish,
                inp.treat_down,
                inp.punish_down,
            )
            operator_busy = bool(teach)
            if teach and onboard is not None:
                onboard.post("treat" if teach == "pam" else "punish")
                teach = None
            elif teach and not session.learn and session.now() - session._learn_block_log > 1.0:
                session._log("обучение выключено, T/X веса не меняют")
                session._learn_block_log = session.now()
                teach = None
            elif teach and not session.learn:
                teach = None
            new_frame = frame_id != seen_frame and camera is not None and not session.onboard
            teach_now = teach and camera is not None and session.dan == "teacher" and not session.onboard
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
            _drive_teacher(session, camera_jpeg, frame_id, teacher_state, operator_busy, onboard)
            if session.onboard:
                now_a = session.now()
                prev_a = float(getattr(session, "_approach_t", now_a))
                session._approach_t = now_a
                session._note_approach(now_a - prev_a)
            snap = None
            if inp.doggibot:
                from .doggibot_snap import reserve_folder

                snap = reserve_folder()
                session._log(snap.line)
                mon.show_toast("снимок DoggiBOT")
            view = _live_view(session, camera, disp_lidar, error, link, inp.focused, command_name, onboard)
            rec_on, rec_n, rec_bytes, _rec_path = rec.stats()
            labelled = weak_label_now(marks, session.now()) is not None
            view.record_on = rec_on
            view.record_saved = rec_n
            view.record_bytes = rec_bytes
            view.record_idle = rec_on and not labelled
            if rec_on and labelled and camera is not None and camera_jpeg and rec.due():
                _offer_live_frame(session, rec, marks, camera, camera_jpeg, onboard)
            from .go2_control import drive_go2
            from .mb_view3d import drive_brain

            drive_brain(session, inp, session.now(), mon)
            drive_go2(session, inp, mon)
            mon.draw(view)
            if snap is not None:
                from .doggibot_snap import commit

                if onboard is not None:
                    mode, link_ok = "onboard", bool(onboard.ok)
                elif link is not None:
                    mode, link_ok = "robot", bool(link.ok)
                else:
                    mode, link_ok = "robot", False
                commit(
                    mon,
                    view,
                    snap,
                    mode=mode,
                    link_ok=link_ok,
                    teacher=getattr(session, "teacher", None),
                    learning=bool(session.learn),
                )
            if inp.screenshot:
                dest = ROOT / "logs" / "monitor_shot.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                mon.save_screenshot(str(dest))
                session._log(f"снимок  {dest.name}")
            if seconds > 0 and session.now() >= seconds:
                break
    finally:
        from .mb_view3d import close_brain

        close_brain(session)
        teacher = getattr(session, "teacher", None)
        if teacher is not None:
            teacher.close()
        owned = getattr(session, "_owned_teacher", None)
        if owned is not None:
            owned.close()
        rec.close()
        try:
            if link is not None:
                link.stop()
        finally:
            if link is not None:
                link.close()
            pull.stop()


def _live_lead_plaque(session) -> str:
    from .teacher_lead import lead_plaque

    return lead_plaque(float(getattr(session, "teacher_weight", 0.0) or 0.0))


def _opt_float(payload: dict, key: str) -> float | None:
    """Missing status fields stay blank. An old onboard build has no lifetime totals."""
    if key not in payload or payload.get(key) is None:
        return None
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        return None


def _scaled(session: LiveSession, rows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not rows:
        return []
    from .brain_panel import scale_readout_curves

    dog, none = scale_readout_curves(session.peer_curve, session.other_curve)
    if rows is session.peer_curve:
        return dog
    return none


def _recog_line(session: LiveSession, remote: dict) -> str:
    """What the onboard brain actually feeds PN. Missing key means an old build."""
    if "recog_camera_only" not in remote:
        return "узнавание: борт без опции"
    return recog_label(bool(remote.get("recog_camera_only")))


def _fly_line(session: LiveSession, onboard: OnboardLink | None) -> str:
    if onboard is not None:
        remote = onboard.status or {}
        return format_fly_line(
            remote.get("r_l"),
            remote.get("r_r"),
            remote.get("r_diff"),
            remote.get("z_fly"),
            str(remote.get("steer") or "bilateral"),
        )
    return format_fly_line(
        session.pilot.track.r_l,
        session.pilot.track.r_r,
        session.pilot.track.r_diff,
        session.pilot.track.yaw_z,
        session.pilot.steer,
    )


def _range_line(session: LiveSession, onboard: OnboardLink | None) -> str:
    if onboard is not None:
        remote = onboard.status or {}
        return format_range_line(
            remote.get("distance_m"),
            remote.get("forward_m"),
            remote.get("sector"),
            remote.get("sector_smooth"),
            remote.get("hysteresis"),
        )
    return format_range_line(
        session.aim_dist,
        session.forward_m,
        session.aim_sector,
        session.pilot.track.sector_smooth,
        session.pilot.track.as_dict(),
    )


def _live_view(
    session: LiveSession,
    camera,
    lidar,
    error: str,
    link: Go2CommandLink | None,
    focused: bool,
    command_name: str = "—",
    onboard: OnboardLink | None = None,
):
    from .brain_panel import panel_readout
    from .train_monitor import MonitorView

    if onboard is not None:
        udp_status = onboard.status_line(session.bridge_open)
    elif link is not None:
        udp_status = link.status_line()
    else:
        udp_status = ""
    common = dict(
        camera=camera,
        lidar=lidar,
        sensor_error=error,
        likeness=session.last_like,
        learned_match=session.last_match,
        metric_curve=session.metric_curve[-120:],
        log_lines=list(session.log),
        paused=not session.learn,
        udp_status=udp_status,
        last_command=command_name,
        operator_label=session.label_text(),
        t=session.now(),
        focused=focused,
        keys_hint=LIVE_KEYS if session.learner_kind == "mb" else (
            "Arrows drive   U record   C reset   V fresh   Space stop   -/+ stand   E E-STOP   D/N   P R S L F12 Esc"
        ),
        pilot_mode=session.pilot.label(),
        pilot_phase=session.pilot.phase,
        pilot_who=session.pilot.who,
        pilot_hint=session.pilot.hint,
        learning_on=session.learn,
        autonomy_on=session.pilot.autonomy,
        lidar_mode=session._lidar_caption(),
        lidar_hold=session.lidar_hold,
        lidar_warning=session.lidar_warning,
        lidar_fresh_on=session.lidar_fresh_on,
        learn_flash=session.remote_flash if session.onboard else (None if session.mb is None else session.mb.flash),
        learn_flash_r=session.remote_flash_r if session.onboard else (None if session.mb is None else session.mb.flash_r),
        mb_layout=session.layout if session.onboard else (None if session.mb is None else session.mb.layout),
        range_line=_range_line(session, onboard),
        fly_line=_fly_line(session, onboard),
        steer=str(session.remote.get("steer") or session.pilot.steer) if session.onboard else session.pilot.steer,
        last_seen_side=str(session.pilot.last_seen_side or ""),
        yolo_state=str(getattr(session, "yolo_state", "") or ""),
        teacher_counts=str(getattr(session, "teacher_counts", "") or ""),
        teacher_skips=str(getattr(session, "teacher_skips", "") or ""),
        teacher_flash_l=str(getattr(session, "teacher_flash_l_show", "") or ""),
        teacher_flash_r=str(getattr(session, "teacher_flash_r_show", "") or ""),
        teacher_boxes=list(getattr(session, "teacher_boxes", ()) or ()),
        search_lamp=not bool(getattr(session, "teacher_boxes", None)),
        teacher_weight=float(getattr(session, "teacher_weight", 0.0) or 0.0),
        lead_plaque=_live_lead_plaque(session),
    )
    if session.mb is not None:
        caption = "сырой выход: подход − избегание" if session.dan == "teacher" else "сырой выход: минус новизна"
        prog = session.mb.progress
        conf = session.mb.conf
        conf_l = session.mb.conf_l
        conf_r = session.mb.conf_r
        mode = f"робот · {session.pilot.label()}"
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
            hemi_l=float(session.pilot.track.r_l),
            hemi_r=float(session.pilot.track.r_r),
            hemi_z=float(session.pilot.track.yaw_z),
            overlap=float(session.mb.overlap),
            eye_l_recognized=bool(conf_l.ready and conf_l.recognized),
            eye_r_recognized=bool(conf_r.ready and conf_r.recognized),
            eye_l_confidence=float(conf_l.percent),
            eye_r_confidence=float(conf_r.percent),
            eye_l_ready=bool(conf_l.ready),
            eye_r_ready=bool(conf_r.ready),
            phase_ru=phase_label(session.pilot.phase, session.pilot.steer, session.pilot.search_sign),
            recog_line=recog_label(bool(getattr(session, "camera_only", True))),
            eyes_line=format_eyes_line(
                session.pilot.track.r_l,
                session.pilot.track.r_r,
                bool(conf_l.ready and conf_l.recognized),
                bool(conf_r.ready and conf_r.recognized),
            ),
            session_sep=prog.session_sep(),
            total_sep=prog.total_sep(),
            session_acc=prog.session_acc(),
            total_acc=prog.total_acc(),
            session_labeled=prog.ses_scored,
            total_labeled=prog.base_scored + prog.ses_scored,
            session_novelty=session.mb.n_novelty,
            total_novelty=prog.base_novelty + session.mb.n_novelty,
            marks=session.marks.visible(session.now()),
            **panel_readout(session.mb, session.approach),
            **common,
        )
    if session.onboard:
        from .brain_panel import brain_panel_from_status

        remote = session.remote if isinstance(session.remote, dict) else {}
        age = onboard.status_age() if onboard is not None else None
        panel_ok, panel_fields = brain_panel_from_status(remote, age)
        view_kw = dict(
            title="тренировка узнавания   мозг на роботе",
            mode_label=f"борт · {session.pilot.label()}",
            learner="mb",
            dan_mode="teacher",
            readout_caption="сырой выход с борта",
            hemi_l=_opt_float(remote, "r_l"),
            hemi_r=_opt_float(remote, "r_r"),
            hemi_z=_opt_float(remote, "z_fly"),
            recognized=bool(remote.get("recognized")),
            confidence=float(remote.get("confidence") or 0.0),
            confidence_ready=bool(remote.get("confidence_ready")),
            overlap=float(session.overlap),
            eye_l_recognized=bool(session.eye_l_recognized),
            eye_r_recognized=bool(session.eye_r_recognized),
            eye_l_confidence=float(session.eye_l_confidence),
            eye_r_confidence=float(session.eye_r_confidence),
            eye_l_ready=bool(session.eye_l_ready),
            eye_r_ready=bool(session.eye_r_ready),
            phase_ru=session.phase_ru or phase_label(session.pilot.phase, session.pilot.steer, session.pilot.search_sign),
            recog_line=_recog_line(session, remote),
            eyes_line=format_eyes_line(
                remote.get("r_l") if "r_l" in remote else None,
                remote.get("r_r") if "r_r" in remote else None,
                bool(session.eye_l_recognized),
                bool(session.eye_r_recognized),
                str(remote.get("eyes_ru") or "") or None,
            ),
            onboard=True,
            brain_offline=not panel_ok,
        )
        view_kw.update(common)
        if panel_ok:
            view_kw.update(panel_fields)
        else:
            view_kw["teacher_counts"] = ""
            view_kw["brain_offline"] = True
        watch = getattr(session, "approach", None)
        if watch is not None:
            if watch.dog is not None and "stim_dog" not in view_kw:
                view_kw["stim_dog"] = float(watch.dog)
            if watch.cube is not None and "stim_cube" not in view_kw:
                view_kw["stim_cube"] = float(watch.cube)
            view_kw["pref_index"] = watch.index()
            view_kw["pref_seconds"] = float(watch.total)
        return MonitorView(**view_kw)
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
    from .ui_settings import ensure_stdio

    ensure_stdio()
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
    p.add_argument("--onboard", default="", help="Robot IP with the fly brain on port 8090. No local Move and no UDP.")
    p.add_argument("--onboard-port", type=int, default=8090)
    p.add_argument("--return-auto", type=float, default=0.0, help="Seconds of idle takeover before autonomy returns. 0 stays manual.")
    p.add_argument("--steer", choices=("bilateral", "sectors"), default="bilateral")
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="Shared fraction of the field, 0..0.5. 0 is the hard midline.")
    add_recog_camera_only_arg(p)
    p.add_argument("--rec-fps", type=float, default=2.0, help="Max camera.jpg frames per second saved on this laptop.")
    p.add_argument("--fullscreen", action="store_true", help="Open the trainer fullscreen. F11 toggles it.")
    p.add_argument("--teacher-url", default="http://127.0.0.1:8091", help="YOLO service on this laptop.")
    p.add_argument("--teacher-conf", type=float, default=0.5, help="Box confidence for a hemisphere hit.")
    p.add_argument("--teacher-rate", type=float, default=2.0, help="Max DAN pulses per second on each hemisphere.")
    from .go2_control import add_go2_control_arg

    add_go2_control_arg(p)
    args = p.parse_args(argv)
    if args.rec_fps <= 0:
        print("--rec-fps must be positive", file=sys.stderr)
        return 2
    if args.onboard and args.learner != "mb":
        print("Onboard mode is the mushroom body. Ignoring --learner hebb.", file=sys.stderr)
        args.learner = "mb"
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
        return_auto_s=args.return_auto,
        onboard=bool(args.onboard),
        steer=args.steer,
        overlap=args.overlap,
        camera_only=bool(args.recog_camera_only),
    )
    session.rec_fps = float(args.rec_fps)
    session.start_fullscreen = bool(args.fullscreen)
    session.teacher_url = str(args.teacher_url)
    session.teacher_conf = float(args.teacher_conf)
    session.teacher_rate = float(args.teacher_rate)
    from .go2_control import apply_go2_control_arg

    apply_go2_control_arg(session, args)
    if args.load or Path(args.state).is_file():
        try_load_live(session)
    if args.headless:
        summary = run_headless(session, base, args.max_frames)
        print(json.dumps(summary, indent=2))
        return 0
    pull = PreviewPull(base)
    pull.start()
    if args.onboard:
        board = OnboardLink(args.onboard, args.onboard_port)
        board.post("steer", mode=args.steer)
        run_live_gui(session, pull, None, seconds=args.seconds, onboard=board)
        return 0
    link = Go2CommandLink(args.udp_host, args.udp_port)
    run_live_gui(session, pull, link, seconds=args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
