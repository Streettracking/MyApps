"""Laptop YOLO teacher. The trainer never imports torch.

A separate process (``tools/yolo_teacher/serve.py``) reads JPEGs and returns
boxes. This module decides which hemisphere gets PAM or PPL1 and draws
nothing itself. It does not send Move.

Learning off means no requests and no boxes. Learning on looks. The Y key
is what starts handing out DAN pulses. H only hides the boxes.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .hemifield import overlap_bands

DEFAULT_URL = "http://127.0.0.1:8091"
DEFAULT_CONF = 0.5
DEFAULT_RATE = 2.0
AREA_MIN = 0.20
TEACHER_KEY = "Y"
BOXES_KEY = "H"
FULLSCREEN_KEY = "F11"
STEER_KEY = "K"


@dataclass
class DetBox:
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float
    zone: str = ""


def _span(box: DetBox) -> tuple[float, float, float]:
    x0 = float(box.x0)
    x1 = float(box.x1)
    if x1 < x0:
        x0, x1 = x1, x0
    return x0, x1, 0.5 * (x0 + x1)


def _hits(box: DetBox, left: float, right: float, area_min: float = AREA_MIN) -> bool:
    """True when the box centre sits in ``[left, right]`` or enough of its width does."""
    x0, x1, center = _span(box)
    if center >= float(left) - 1e-9 and center <= float(right) + 1e-9:
        return True
    width = x1 - x0
    if width <= 1e-9:
        return False
    lo = x0 if x0 > float(left) else float(left)
    hi = x1 if x1 < float(right) else float(right)
    if hi <= lo:
        return False
    return (hi - lo) / width >= float(area_min)


def zone_of(box: DetBox, overlap: float, area_min: float = AREA_MIN) -> str:
    """«Л», «П», or «Л+П». Image left is П (MB_R). Image right is Л (MB_L)."""
    lo, hi = overlap_bands(overlap)
    hit_l = _hits(box, lo, 1.0, area_min)
    hit_r = _hits(box, 0.0, hi, area_min)
    if hit_l and hit_r:
        return "Л+П"
    if hit_l:
        return "Л"
    if hit_r:
        return "П"
    return ""


def _center_in(box: DetBox, left: float, right: float) -> bool:
    _x0, _x1, center = _span(box)
    return center >= float(left) - 1e-9 and center <= float(right) + 1e-9


def hemisphere_hits(
    boxes,
    overlap: float,
    conf_min: float = DEFAULT_CONF,
    area_min: float = AREA_MIN,
    center_only: bool = False,
) -> tuple[bool, bool]:
    """``(hit_L, hit_R)`` for boxes at or above ``conf_min``.

    Teaching passes ``center_only``. A box whose centre sits in the other eye
    does not count, even when the overlap lets part of its width cross.
    A centre in the shared band hits both eyes. Screen labels stay on area.
    """
    hit_l = False
    hit_r = False
    lo, hi = overlap_bands(overlap)
    for box in boxes:
        if float(box.conf) < float(conf_min):
            continue
        if center_only:
            hit_l = hit_l or _center_in(box, lo, 1.0)
            hit_r = hit_r or _center_in(box, 0.0, hi)
        else:
            hit_l = hit_l or _hits(box, lo, 1.0, area_min)
            hit_r = hit_r or _hits(box, 0.0, hi, area_min)
    return hit_l, hit_r


def dan_for_eyes(
    hit_l: bool,
    hit_r: bool,
    recognized_l: bool,
    recognized_r: bool,
) -> tuple[str | None, str | None]:
    """PAM where the box lands. PPL1 only on a false «узнаю». Nothing otherwise."""

    def one(hit: bool, recognized: bool) -> str | None:
        if hit:
            return "pam"
        if recognized:
            return "ppl1"
        return None

    return one(hit_l, recognized_l), one(hit_r, recognized_r)


FLASH_S = 0.45


def teach_phrase(side: str, kind: str | None) -> str:
    """One journal line per pulse. Empty when this side got nothing."""
    if side == "L" and kind == "ppl1":
        return "учитель: PPL1 Л — ложное узнавание"
    if side == "R" and kind == "ppl1":
        return "учитель: PPL1 П — ложное узнавание"
    if side == "L" and kind == "pam":
        return "учитель: PAM Л — собака в поле"
    if side == "R" and kind == "pam":
        return "учитель: PAM П — собака в поле"
    return ""


def plaque_recognized(mb, eye_l: bool, eye_r: bool, overlap: float) -> tuple[bool, bool, float]:
    """The bits painted on the plaques. Onboard has no local brain, so the status flags are used."""
    if mb is not None:
        rec_l, rec_r = mb.eye_recognized()
        return bool(rec_l), bool(rec_r), float(mb.overlap)
    return bool(eye_l), bool(eye_r), float(overlap)


def annotate_boxes(boxes, overlap: float, conf_min: float = DEFAULT_CONF) -> list[DetBox]:
    out: list[DetBox] = []
    for box in boxes:
        if float(box.conf) < float(conf_min):
            continue
        zone = zone_of(box, overlap)
        if not zone:
            continue
        out.append(DetBox(float(box.x0), float(box.y0), float(box.x1), float(box.y1), float(box.conf), zone))
    return out


class TeachGate:
    """At most ``rate`` pulses per second for each eye and each kind.

    PAM and PPL1 keep separate clocks, so a stream of treats cannot eat the
    punishment slot.
    """

    def __init__(self, rate: float = DEFAULT_RATE):
        gap = 0.5 if float(rate) <= 0 else 1.0 / float(rate)
        self.min_gap_s = gap
        self.last: dict[tuple[str, str], float] = {}

    def filter(self, now: float, kind_l: str | None, kind_r: str | None) -> tuple[str | None, str | None]:
        return self._one(now, "L", kind_l), self._one(now, "R", kind_r)

    def _one(self, now: float, side: str, kind: str | None) -> str | None:
        if kind not in ("pam", "ppl1"):
            return None
        key = (side, kind)
        prev = self.last.get(key, -1e9)
        if float(now) - prev < self.min_gap_s - 1e-9:
            return None
        self.last[key] = float(now)
        return kind


def yolo_status(learning: bool, teacher_on: bool, service_down: bool) -> str:
    """Panel word. Learning off wins, then a dead service, then учит / смотрит."""
    if not learning:
        return "выкл (нет обучения)"
    if service_down:
        return "учитель не запущен"
    if teacher_on:
        return "учит"
    return "смотрит"


def should_query(learning: bool, teacher_on: bool, boxes_hidden: bool, service_down: bool, since_fail: float) -> bool:
    """Whether this frame may be posted to the YOLO service."""
    if not learning:
        return False
    if service_down and since_fail < 2.0:
        return False
    if teacher_on:
        return True
    if boxes_hidden:
        return False
    return True


def send_teach(link, kind_l: str | None, kind_r: str | None) -> None:
    """Same ``/cmd`` channel as T/X. Does not send Move."""
    if kind_l not in ("pam", "ppl1") and kind_r not in ("pam", "ppl1"):
        return
    link.post("teach_sides", left=kind_l or "", right=kind_r or "")


def counts_line(pam_l: int, pam_r: int, ppl1_l: int, ppl1_r: int) -> str:
    return "учитель: PAM_L %d / PAM_R %d / PPL1_L %d / PPL1_R %d" % (
        int(pam_l),
        int(pam_r),
        int(ppl1_l),
        int(ppl1_r),
    )


class YoloClient:
    """HTTP client. A dead service returns None and never raises into the trainer."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 0.8):
        self.url = str(url).rstrip("/")
        self.timeout = float(timeout)

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=min(0.4, self.timeout)) as resp:
                return int(resp.status) == 200
        except Exception:
            return False

    def detect(self, jpeg: bytes) -> list[DetBox] | None:
        if not jpeg:
            return []
        req = urllib.request.Request(
            self.url + "/detect",
            data=jpeg,
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None
        except Exception:
            return None
        boxes: list[DetBox] = []
        for row in payload.get("boxes") or []:
            try:
                boxes.append(
                    DetBox(
                        float(row["x0"]),
                        float(row["y0"]),
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row.get("conf", 0.0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return boxes


class TeacherRuntime:
    """Background detect. The pygame loop only reads the latest boxes."""

    def __init__(self, url: str = DEFAULT_URL, conf: float = DEFAULT_CONF, rate: float = DEFAULT_RATE):
        self.client = YoloClient(url)
        self.conf = float(conf)
        self.gate = TeachGate(rate)
        self.on = False
        self.boxes_hidden = False
        self.service_down = False
        self.boxes: list[DetBox] = []
        self.pam_l = 0
        self.pam_r = 0
        self.ppl1_l = 0
        self.ppl1_r = 0
        self.skip_n = 0
        self.skip_reason = ""
        self.answered = False
        self._teach_boxes: list[DetBox] = []
        self._skip_at = {"L": -1e9, "R": -1e9}
        self._last_fail = -1e9
        self._gen = 0
        self._pending: tuple[int, bytes] | None = None
        self._result: tuple[int, list[DetBox] | None] | None = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._learning = False

    def close(self) -> None:
        self._stop = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)

    def toggle_teacher(self) -> str:
        if self.on:
            self.on = False
            return "авто-учитель выключен"
        if not self.client.health():
            self.service_down = True
            self._last_fail = time.monotonic()
            self.on = False
            return "учитель не запущен"
        self.service_down = False
        self.on = True
        return "авто-учитель включен"

    def toggle_boxes(self, learning: bool) -> str | None:
        if not learning:
            return None
        self.boxes_hidden = not self.boxes_hidden
        if self.boxes_hidden and not self.on:
            self.boxes = []
        return "рамки скрыты" if self.boxes_hidden else "рамки видны"

    def note_learning(self, learning: bool) -> None:
        learning = bool(learning)
        if learning == self._learning:
            if not learning:
                self.boxes = []
            return
        self._learning = learning
        if learning:
            return
        self._gen += 1
        self.boxes = []
        self._teach_boxes = []
        self.answered = False
        with self._lock:
            self._pending = None
            self._result = None

    def status_text(self, learning: bool) -> str:
        return yolo_status(learning, self.on, self.service_down)

    def wants_frame(self, learning: bool) -> bool:
        since = time.monotonic() - self._last_fail
        return should_query(learning, self.on, self.boxes_hidden, self.service_down, since)

    def offer_frame(self, jpeg: bytes) -> None:
        if not jpeg:
            return
        self._ensure()
        with self._lock:
            self._pending = (self._gen, jpeg)

    def skips_line(self) -> str:
        if self.skip_n <= 0:
            return "ложных узнаваний без наказания: 0"
        return "ложных узнаваний без наказания: %d (%s)" % (int(self.skip_n), self.skip_reason or "—")

    def collect(
        self,
        now: float,
        overlap: float,
        recognized_l: bool,
        recognized_r: bool,
        learning: bool,
        operator_busy: bool,
    ) -> tuple[str | None, str | None]:
        """DAN from the latest YOLO answer and the plaque bits passed in.

        An empty answer is «no dog» and can be PPL1. No answer yet is not a
        guess. ``recognized_*`` must be the same bits the plaques show.
        """
        if not learning:
            self.note_learning(False)
            return None, None
        self._take_result(overlap)
        if self.boxes_hidden and not self.on:
            self.boxes = []
        if not self.on or operator_busy:
            return None, None
        if not self.answered or self.service_down:
            self._note_skips(now, recognized_l, recognized_r, "нет ответа YOLO", "нет ответа YOLO")
            return None, None
        hit_l, hit_r = hemisphere_hits(
            self._teach_boxes,
            overlap,
            conf_min=self.conf,
            center_only=True,
        )
        want_l, want_r = dan_for_eyes(hit_l, hit_r, recognized_l, recognized_r)
        kind_l, kind_r = self.gate.filter(now, want_l, want_r)
        self._count(kind_l, kind_r)
        self._note_skips(
            now,
            recognized_l,
            recognized_r,
            self._miss_reason(recognized_l, hit_l, want_l, kind_l),
            self._miss_reason(recognized_r, hit_r, want_r, kind_r),
        )
        return kind_l, kind_r

    def visible_boxes(self, learning: bool) -> list[DetBox]:
        if not learning or self.boxes_hidden:
            return []
        return list(self.boxes)

    def _miss_reason(self, recognized: bool, hit: bool, want: str | None, sent: str | None) -> str:
        if not recognized or sent == "ppl1":
            return ""
        if hit:
            return "бокс в поле"
        if want == "ppl1":
            return "лимит"
        return ""

    def _note_skips(self, now: float, recognized_l: bool, recognized_r: bool, reason_l: str, reason_r: str) -> None:
        parts = []
        if recognized_l and reason_l:
            parts.append(("L", "Л", reason_l))
        if recognized_r and reason_r:
            parts.append(("R", "П", reason_r))
        if not parts:
            return
        reasons = {reason for _side, _label, reason in parts}
        if len(reasons) == 1:
            self.skip_reason = parts[0][2]
        else:
            self.skip_reason = ", ".join("%s: %s" % (label, reason) for _side, label, reason in parts)
        gap = self.gate.min_gap_s
        for side, _label, _reason in parts:
            prev = self._skip_at.get(side, -1e9)
            if float(now) - prev < gap - 1e-9:
                continue
            self._skip_at[side] = float(now)
            self.skip_n += 1

    def _count(self, kind_l: str | None, kind_r: str | None) -> None:
        if kind_l == "pam":
            self.pam_l += 1
        elif kind_l == "ppl1":
            self.ppl1_l += 1
        if kind_r == "pam":
            self.pam_r += 1
        elif kind_r == "ppl1":
            self.ppl1_r += 1

    def _take_result(self, overlap: float) -> None:
        with self._lock:
            result = self._result
            self._result = None
        if result is None:
            return
        gen, boxes = result
        if gen != self._gen:
            return
        if boxes is None:
            self.service_down = True
            self._last_fail = time.monotonic()
            self.answered = False
            self._teach_boxes = []
            self.boxes = []
            return
        self.service_down = False
        self.answered = True
        self._teach_boxes = annotate_boxes(boxes, overlap, self.conf)
        if self.boxes_hidden and not self.on:
            self.boxes = []
        else:
            self.boxes = list(self._teach_boxes)

    def _ensure(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="yolo-teacher", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                item = self._pending
                self._pending = None
            if item is None:
                time.sleep(0.01)
                continue
            gen, jpeg = item
            boxes = self.client.detect(jpeg)
            with self._lock:
                self._result = (gen, boxes)
