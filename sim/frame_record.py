"""Save raw camera JPEGs for a later YOLO set. The robot does not write these.

Live and onboard modes pass the exact ``/camera.jpg`` bytes from the preview
server. The simulator encodes its camera picture, the one with no monitor
overlay. A background thread does the disk write. The main loop only queues.
"""

from __future__ import annotations

import hashlib
import io
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

LABEL_WINDOW_S = 0.5
SIMILAR_MEAN = 4.0
_DOG = ("T", "D")
_NONE = ("X", "N")


def _meta_has_label(meta: dict) -> bool:
    """True only for a dog or no_dog weak label. Unlabelled frames are not files."""
    operator = meta.get("operator") if isinstance(meta, dict) else None
    if not isinstance(operator, dict):
        return False
    return operator.get("weak_label") in ("dog", "no_dog")


def jpeg_complete(blob: bytes) -> bool:
    """A finished JPEG starts with FFD8 and ends with FFD9."""
    return len(blob) >= 4 and blob[:2] == b"\xff\xd8" and blob[-2:] == b"\xff\xd9"


def fingerprint(rgb: np.ndarray) -> np.ndarray:
    """24×32 gray sample. Used only to skip near-duplicate frames."""
    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[0] < 2 or img.shape[1] < 2:
        return np.zeros((24, 32), dtype=np.uint8)
    gray = img[:, :, :3].mean(axis=2)
    h, w = gray.shape
    y_idx = np.linspace(0, h - 1, 24).astype(np.int32)
    x_idx = np.linspace(0, w - 1, 32).astype(np.int32)
    return np.ascontiguousarray(gray[y_idx][:, x_idx], dtype=np.uint8)


def too_similar(prev: np.ndarray | None, cur: np.ndarray | None, limit: float = SIMILAR_MEAN) -> bool:
    if prev is None or cur is None:
        return False
    if prev.shape != cur.shape:
        return False
    return float(np.mean(np.abs(prev.astype(np.int16) - cur.astype(np.int16)))) < float(limit)


def encode_camera_jpeg(rgb: np.ndarray) -> bytes:
    """JPEG of the simulator camera. No window chrome, no lidar marks.

    Called from the GUI thread before the frame is queued. The writer thread
    only stores the bytes.
    """
    import pygame

    from .sdl_thread import require_main_thread

    require_main_thread()
    arr = np.ascontiguousarray(np.transpose(np.asarray(rgb), (1, 0, 2)))
    surf = pygame.surfarray.make_surface(arr)
    buf = io.BytesIO()
    pygame.image.save(surf, buf, "jpg")
    data = buf.getvalue()
    if not jpeg_complete(data):
        raise ValueError("simulator camera did not encode as a JPEG")
    return data


def format_disk(n_bytes: int) -> str:
    n = int(n_bytes)
    if n < 1024 * 1024:
        return "%.0f КБ" % (n / 1024.0)
    return "%.1f МБ" % (n / (1024.0 * 1024.0))


class OperatorMarks:
    """T/X/D/N presses in the last half-second. Not a training input."""

    def __init__(self, window_s: float = LABEL_WINDOW_S) -> None:
        self.window_s = float(window_s)
        self.events: list[tuple[float, str]] = []

    def note(self, key: str, t: float) -> None:
        if key not in _DOG and key not in _NONE:
            return
        self.events.append((float(t), key))
        self._trim(float(t))

    def _trim(self, t: float) -> None:
        window = self.window_s
        self.events = [(ts, key) for ts, key in self.events if t - ts <= window]

    def export(self, t: float) -> tuple[list[dict], str | None]:
        """Events in the window, and dog / no_dog from the latest key."""
        t = float(t)
        self._trim(t)
        best: dict[str, float] = {}
        for ts, key in self.events:
            age = t - ts
            if age < 0 or age > self.window_s:
                continue
            if key not in best or age < best[key]:
                best[key] = age
        if not best:
            return [], None
        latest_key = None
        latest_age = None
        for ts, key in self.events:
            age = t - ts
            if age < 0 or age > self.window_s:
                continue
            if latest_age is None or age <= latest_age:
                latest_age = age
                latest_key = key
        events = [{"key": key, "age_s": round(age, 3)} for key, age in sorted(best.items(), key=lambda item: item[1])]
        if latest_key in _DOG:
            return events, "dog"
        if latest_key in _NONE:
            return events, "no_dog"
        return events, None


def note_operator(marks: OperatorMarks, t: float, inp) -> None:
    """Record a key that is down on this frame. Held T/X stay inside the window."""
    if inp.treat or inp.treat_down:
        marks.note("T", t)
    if inp.punish or inp.punish_down:
        marks.note("X", t)
    if inp.label == "dog":
        marks.note("D", t)
    elif inp.label == "none":
        marks.note("N", t)


def weak_label_now(marks: OperatorMarks, t: float) -> str | None:
    """dog / no_dog when T/D or X/N is inside the label window, else None."""
    _events, weak = marks.export(t)
    return weak


def make_meta(
    marks: OperatorMarks,
    t_s: float,
    *,
    source: str,
    mode: str,
    speed_m_s: float,
    yaw_rad_s: float,
    readout: float | None,
    confidence: float | None,
    confidence_ready: bool,
    recognized: bool,
    sector: int | None,
    r_l: float | None,
    r_r: float | None,
) -> dict:
    events, weak = marks.export(t_s)
    diff = None
    if r_l is not None and r_r is not None:
        diff = float(r_l) - float(r_r)
    drive = "auto" if mode == "auto" else "manual"
    return {
        "time": datetime.now().isoformat(timespec="milliseconds"),
        "t_s": round(float(t_s), 3),
        "source": source,
        "operator": {"window_s": marks.window_s, "events": events, "weak_label": weak},
        "mb": {
            "readout": None if readout is None else round(float(readout), 3),
            "confidence": None if confidence is None else round(float(confidence), 3),
            "confidence_ready": bool(confidence_ready),
            "recognized": bool(recognized),
            "sector": None if sector is None else int(sector),
            "r_l": None if r_l is None else round(float(r_l), 3),
            "r_r": None if r_r is None else round(float(r_r), 3),
            "r_diff": None if diff is None else round(diff, 3),
        },
        "mode": drive,
        "speed_m_s": round(float(speed_m_s), 4),
        "yaw_rad_s": round(float(yaw_rad_s), 4),
    }


class FrameRecorder:
    """Queue JPEGs and write them off the teleop thread.

    ``offer`` returns immediately. A frame with no T/X/D/N label in the
    window is not queued and does not use a rate-limit slot. Identical bytes
    and near-duplicate thumbnails are dropped in the writer. At most ``fps``
    labelled frames are queued per second.
    """

    def __init__(self, root: Path | str, fps: float = 2.0) -> None:
        if fps <= 0:
            raise ValueError("rec fps must be positive")
        self.root = Path(root)
        self.fps = float(fps)
        self.min_interval = 1.0 / float(fps)
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="yolo-frames", daemon=True)
        self._thread.start()
        self._lock = threading.Lock()
        self.enabled = False
        self.session_dir: Path | None = None
        self.saved = 0
        self.nbytes = 0
        self.skipped = 0
        self._last_offer = -1e9
        self._last_digest: bytes | None = None
        self._last_fp: np.ndarray | None = None
        self._next_index = 1

    def toggle(self) -> bool:
        with self._lock:
            self.enabled = not self.enabled
            if self.enabled:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = self.root / ("session_%s" % stamp)
                n = 2
                while dest.exists():
                    dest = self.root / ("session_%s_%s" % (stamp, n))
                    n += 1
                dest.mkdir(parents=True, exist_ok=False)
                self.session_dir = dest
                self.saved = 0
                self.nbytes = 0
                self.skipped = 0
                self._next_index = 1
                self._last_digest = None
                self._last_fp = None
                self._last_offer = -1e9
            return self.enabled

    def stats(self) -> tuple[bool, int, int, str | None]:
        with self._lock:
            path = None if self.session_dir is None else str(self.session_dir)
            return self.enabled, self.saved, self.nbytes, path

    def due(self, now: float | None = None) -> bool:
        """True when another frame may be queued. Does not reserve the slot."""
        with self._lock:
            if not self.enabled:
                return False
        clock = time.monotonic() if now is None else float(now)
        return clock - self._last_offer >= self.min_interval

    def offer(self, jpeg: bytes, meta: dict, fp: np.ndarray | None, now: float | None = None) -> bool:
        """Queue one labelled frame. False when there is no label, recording is off, rate-limited, or the queue is full."""
        if not jpeg_complete(jpeg):
            return False
        if not _meta_has_label(meta):
            return False
        with self._lock:
            if not self.enabled or self.session_dir is None:
                return False
        clock = time.monotonic() if now is None else float(now)
        if clock - self._last_offer < self.min_interval:
            return False
        self._last_offer = clock
        try:
            self._queue.put_nowait((bytes(jpeg), dict(meta), None if fp is None else np.array(fp, copy=True)))
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        self._stop = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            self._write(item[0], item[1], item[2])

    def _write(self, jpeg: bytes, meta: dict, fp: np.ndarray | None) -> None:
        digest = hashlib.sha1(jpeg).digest()
        with self._lock:
            if digest == self._last_digest or too_similar(self._last_fp, fp):
                self._last_digest = digest
                self._last_fp = fp
                self.skipped += 1
                return
            dest = self.session_dir
            index = self._next_index
            self._next_index += 1
            self._last_digest = digest
            self._last_fp = fp
        if dest is None:
            return
        stem = "frame_%06d" % index
        jpg_path = dest / ("%s.jpg" % stem)
        json_path = dest / ("%s.json" % stem)
        jpg_path.write_bytes(jpeg)
        text = json.dumps(meta, ensure_ascii=False, indent=2)
        json_path.write_text(text, encoding="utf-8")
        with self._lock:
            self.saved += 1
            self.nbytes += jpg_path.stat().st_size + json_path.stat().st_size
