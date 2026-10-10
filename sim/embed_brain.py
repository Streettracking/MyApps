"""Embedded 3D in its own process.

The trainer never creates a moderngl context. The child renders a fixed
640×360 frame (GL, or pygame lines if GL fails) and publishes RGB bytes
through a seqlock in shared memory. The window only blits a copy, scaled
into the dock block, so a GL crash or a dock resize cannot take the trainer
with it. Spawn, not fork: a forked child would inherit the parent's SDL window.
"""

from __future__ import annotations

import os
import struct
import time
import traceback
from multiprocessing import shared_memory
from queue import Empty

import numpy as np

FRAME_W = 640
FRAME_H = 360
HEADER = 16
PAYLOAD = FRAME_W * FRAME_H * 3
SHM_SIZE = HEADER + PAYLOAD
SUBMIT_HZ = 15.0

STARTS = 0


def plain_packet(packet) -> dict:
    """Scalars and short int lists. Nothing that aliases a live KC buffer."""
    if not isinstance(packet, dict):
        packet = {}

    def ids(key: str, cap: int = 400) -> list:
        out = []
        for item in list(packet.get(key) or [])[:cap]:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out

    edges = []
    for item in list(packet.get("edges") or [])[:80]:
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        try:
            edges.append([int(item[0]), int(item[1]), float(item[2]), int(item[3])])
        except (TypeError, ValueError):
            continue
    return {
        "t": float(packet.get("t") or 0.0),
        "r_l": float(packet.get("r_l") or 0.0),
        "r_r": float(packet.get("r_r") or 0.0),
        "kc_l": ids("kc_l"),
        "kc_r": ids("kc_r"),
        "flash_l": str(packet.get("flash_l") or ""),
        "flash_r": str(packet.get("flash_r") or ""),
        "rec_l": bool(packet.get("rec_l")),
        "rec_r": bool(packet.get("rec_r")),
        "edges": edges,
    }


def _publish(shm, image: np.ndarray, seq: int) -> int:
    """Odd sequence while the bytes are in flight, even when the frame is stable."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        return int(seq)
    h, w = int(image.shape[0]), int(image.shape[1])
    raw = image.tobytes()
    n = len(raw)
    if n != w * h * 3 or n > PAYLOAD or HEADER + n > shm.size:
        return int(seq)
    odd = int(seq) + 1
    if odd % 2 == 0:
        odd += 1
    buf = shm.buf
    struct.pack_into("<I", buf, 0, odd)
    struct.pack_into("<III", buf, 4, w, h, n)
    buf[HEADER : HEADER + n] = raw
    even = odd + 1
    struct.pack_into("<I", buf, 0, even)
    return even


def _read_frame(shm):
    """Copy one stable frame out. None while the writer is mid-update or empty."""
    buf = shm.buf
    seq1 = struct.unpack_from("<I", buf, 0)[0]
    if seq1 == 0 or seq1 & 1:
        return None
    w, h, n = struct.unpack_from("<III", buf, 4)
    if w <= 0 or h <= 0 or n != w * h * 3 or n > PAYLOAD:
        return None
    raw = bytes(buf[HEADER : HEADER + n])
    seq2 = struct.unpack_from("<I", buf, 0)[0]
    if seq2 != seq1:
        return None
    image = np.frombuffer(raw, dtype=np.uint8).copy().reshape(h, w, 3)
    return int(seq1), image


def _rgb_surface(image: np.ndarray):
    import pygame

    owned = np.ascontiguousarray(np.transpose(image, (1, 0, 2)))
    return pygame.surfarray.make_surface(owned)


def _render(surface, cloud, cmd, soft: bool) -> bool:
    from .mb_view3d import paint

    packet = cmd.get("packet") if isinstance(cmd.get("packet"), dict) else {}
    try:
        paint(
            surface,
            cloud,
            packet,
            float(cmd.get("yaw") or 0.0),
            float(cmd.get("pitch") or 0.0),
            float(cmd.get("dist") or 4.0),
            auto=False,
            pan_x=float(cmd.get("pan_x") or 0.0),
            pan_y=float(cmd.get("pan_y") or 0.0),
            frames=bool(cmd.get("frames")),
            fast=True,
            hint="ЛКМ обзор   ПКМ/СКМ/Shift сдвиг   колёсико зум   кнопка «каркас»",
            software_only=soft,
        )
        return soft
    except Exception:
        traceback.print_exc()
        if soft:
            return True
        paint(
            surface,
            cloud,
            packet,
            float(cmd.get("yaw") or 0.0),
            float(cmd.get("pitch") or 0.0),
            float(cmd.get("dist") or 4.0),
            auto=False,
            pan_x=float(cmd.get("pan_x") or 0.0),
            pan_y=float(cmd.get("pan_y") or 0.0),
            frames=bool(cmd.get("frames")),
            fast=True,
            software_only=True,
        )
        return True


def _serve(queue, shm_name: str, software: bool = False) -> None:
    """Child entry. SDL stays dummy so this process never opens a window."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    try:
        _serve_body(queue, shm_name, bool(software))
    except Exception:
        traceback.print_exc()


def _serve_body(queue, shm_name: str, software: bool = False) -> None:
    import pygame

    pygame.init()
    from .mb_view3d import BrainCloud

    shm = shared_memory.SharedMemory(name=shm_name)
    surface = pygame.Surface((FRAME_W, FRAME_H))
    cloud = BrainCloud()
    seq = 0
    latest = None
    soft = bool(software)
    try:
        while True:
            drained = False
            while True:
                try:
                    if latest is None and not drained:
                        item = queue.get(timeout=0.25)
                    else:
                        item = queue.get_nowait()
                except Empty:
                    break
                drained = True
                if isinstance(item, dict) and item.get("stop"):
                    return
                if isinstance(item, dict):
                    latest = item
            if latest is None:
                continue
            soft = _render(surface, cloud, latest, soft)
            rgb = pygame.surfarray.array3d(surface)
            image = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
            seq = _publish(shm, image, seq)
            time.sleep(0.03)
    finally:
        try:
            shm.close()
        except Exception:
            pass


class EmbedSession:
    """Parent side. submit never blocks. take a copied surface, or None."""

    def __init__(self):
        self.proc = None
        self.queue = None
        self.shm = None
        self.dead = False
        self._stopping = False
        self._restarted = False
        self._pending = None
        self._last_submit = 0.0
        self._seq = 0
        self._base = None
        self._scaled: dict = {}
        self._software = False
        self._started = 0.0

    def start(self) -> None:
        global STARTS
        if self.dead:
            return
        if self.proc is not None and self.proc.is_alive():
            return
        if self.proc is not None or self.shm is not None:
            self._release()
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE)
        queue = ctx.Queue(maxsize=2)
        proc = ctx.Process(
            target=_serve,
            args=(queue, shm.name, bool(self._software)),
            daemon=True,
            name="embed-brain",
        )
        try:
            proc.start()
        except Exception:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
            try:
                queue.close()
            except Exception:
                pass
            raise
        self.shm = shm
        self.queue = queue
        self.proc = proc
        self._stopping = False
        self._started = time.monotonic()
        self._seq = 0
        STARTS += 1

    def poll(self) -> None:
        proc = self.proc
        if self._stopping or proc is None:
            return
        if proc.is_alive():
            if (
                not self._software
                and not self._restarted
                and self._seq == 0
                and self._started
                and time.monotonic() - self._started > 5.0
            ):
                self._software = True
                self._restarted = True
                self._release()
                try:
                    self.start()
                except Exception:
                    self.dead = True
            return
        if self._restarted:
            self.dead = True
            self._release()
            return
        self._restarted = True
        self._software = True
        self._release()
        try:
            self.start()
        except Exception:
            self.dead = True

    def submit(self, yaw, pitch, dist, pan_x, pan_y, frames, packet) -> None:
        self.poll()
        if self.dead or self.queue is None:
            return
        self._pending = {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "dist": float(dist),
            "pan_x": float(pan_x),
            "pan_y": float(pan_y),
            "frames": bool(frames),
            "packet": plain_packet(packet),
            "stop": False,
        }
        self._flush()

    def _flush(self) -> None:
        if self._pending is None or self.queue is None or self.proc is None or not self.proc.is_alive():
            return
        now = time.monotonic()
        if now - self._last_submit < (1.0 / SUBMIT_HZ):
            return
        cmd = self._pending
        try:
            self.queue.put_nowait(cmd)
        except Exception:
            try:
                self.queue.get_nowait()
            except Exception:
                pass
            try:
                self.queue.put_nowait(cmd)
            except Exception:
                return
        self._pending = None
        self._last_submit = now

    def frame_for(self, w: int, h: int):
        self.poll()
        self._flush()
        if self.dead or self.shm is None:
            return None
        got = None
        try:
            got = _read_frame(self.shm)
        except Exception:
            got = None
        if got is not None:
            seq, image = got
            if seq != self._seq:
                self._base = _rgb_surface(image)
                self._scaled.clear()
                self._seq = seq
        if self._base is None:
            return None
        w, h = int(w), int(h)
        if w < 2 or h < 2:
            return None
        hit = self._scaled.get((w, h))
        if hit is not None:
            return hit
        import pygame

        scaled = pygame.transform.smoothscale(self._base, (w, h))
        if len(self._scaled) > 4:
            self._scaled.clear()
        self._scaled[(w, h)] = scaled
        return scaled

    def _release(self) -> None:
        proc = self.proc
        queue = self.queue
        self.proc = None
        self.queue = None
        if queue is not None and proc is not None and proc.is_alive():
            try:
                queue.put_nowait({"stop": True})
            except Exception:
                pass
        if proc is not None and proc.is_alive():
            proc.join(0.4)
            if proc.is_alive():
                proc.terminate()
                proc.join(0.3)
            if proc.is_alive():
                proc.kill()
                proc.join(0.3)
        if queue is not None:
            try:
                queue.cancel_join_thread()
                queue.close()
            except Exception:
                pass
        shm = self.shm
        self.shm = None
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        self._base = None
        self._scaled = {}
        self._pending = None

    def stop(self) -> None:
        self._stopping = True
        self.dead = False
        self._release()
