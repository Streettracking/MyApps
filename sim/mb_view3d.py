"""Separate 3D window for the two mushroom bodies.

``connectome_mb_v1.npz`` has FlyWire root ids, types, and synapses, not soma
or skeleton coordinates. Positions come from
``artifacts/connectome_mb_v1_neurons.csv.gz`` (``soma_x/y/z``, else ``pos_*``).
A neuron missing from that table is placed on a small schematic lobe so the
cloud stays complete.

The trainer process only sends a UDP datagram to 127.0.0.1:5473. This module
is the receiver and the window. B stays the beep; the window is J.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .npz_compat import open_npz

HOST = "127.0.0.1"
VIEW_PORT = 5473
RATE_HZ = 12.0
KC_CAP = 400
EDGE_CAP = 80

_GAMMA = ("KCg-d", "KCg-s1", "KCg-s2", "KCg-s3", "KCg-m")
_AB = ("KCab-p", "KCab")

_BG_TOP = (236, 234, 230)
_BG_BOT = (244, 242, 239)
_LABEL = (107, 107, 107)
_VALUE = (63, 63, 63)
_PEACH = (201, 120, 91)
_GREEN = (141, 181, 150)
_RED = (217, 136, 128)
_KC = ((196, 176, 160), (168, 186, 176), (176, 180, 196))


def artifact_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "artifacts"
    return Path(__file__).resolve().parents[1] / "artifacts"


def connectome_npz() -> Path:
    folder = artifact_dir()
    plain = folder / "connectome_mb_v1_np1.npz"
    if plain.is_file():
        return plain
    return folder / "connectome_mb_v1.npz"


def neurons_csv() -> Path:
    return artifact_dir() / "connectome_mb_v1_neurons.csv.gz"


def compact_ids(vec, cap: int = KC_CAP) -> list:
    """Local KC indices that fired. Capped so one datagram stays small."""
    if vec is None:
        return []
    arr = np.asarray(vec)
    if arr.size == 0:
        return []
    idx = np.flatnonzero(arr > 0)
    if idx.size > cap:
        step = int(np.ceil(idx.size / float(cap)))
        idx = idx[::step][:cap]
    return [int(i) for i in idx]


def _lobe(name: str) -> int:
    if name in _GAMMA:
        return 0
    if name in _AB:
        return 1
    return 2


def _load_somas(path: Path) -> dict:
    import csv
    import gzip

    out = {}
    if not path.is_file():
        return out
    with gzip.open(str(path), "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                root = int(row.get("root_id") or 0)
            except ValueError:
                continue
            raw = []
            for a, b in (("soma_x", "pos_x"), ("soma_y", "pos_y"), ("soma_z", "pos_z")):
                text = row.get(a) or row.get(b) or ""
                try:
                    raw.append(float(text))
                except ValueError:
                    raw.append(float("nan"))
            if not all(np.isfinite(raw)):
                continue
            out[root] = (raw[0], raw[1], raw[2])
    return out


def _schematic(i: int, left: bool, lobe: int) -> tuple:
    x = -0.62 if left else 0.62
    y = (0.05, 0.38, 0.22)[lobe]
    ang = (i * 0.61803398875) % 1.0 * 6.28318530718
    rad = 0.05 + (i % 17) * 0.01
    return (
        x + rad * np.cos(ang) * 0.45,
        y + ((i % 23) - 11) * 0.01,
        rad * np.sin(ang) * 0.45,
    )


class BrainCloud:
    """Unit-space points. Anatomical left is FlyWire ``side=left`` (Л)."""

    def __init__(self, npz_path: Path | None = None, csv_path: Path | None = None):
        z = open_npz(npz_path or connectome_npz())
        try:
            root_ids = z["root_ids"].astype(np.int64)
            types = np.array([str(t) for t in z["cell_types"]])
            if "sides" in z.files:
                sides = np.array([str(s) for s in z["sides"]])
            else:
                sides = np.array(["right"] * len(root_ids))
            kc_idx = z["kc_idx"].astype(np.int32)
            mbon_idx = z["mbon_idx"].astype(np.int32)
            app = z["dan_appetitive_idx"].astype(np.int32) if "dan_appetitive_idx" in z.files else np.zeros(0, np.int32)
            av = z["dan_aversive_idx"].astype(np.int32) if "dan_aversive_idx" in z.files else np.zeros(0, np.int32)
            pre = z["kc_mbon_pre"].astype(np.int32)
            post = z["kc_mbon_post"].astype(np.int32)
        finally:
            z.close()
        somas = _load_somas(csv_path or neurons_csv())
        n = len(root_ids)
        raw = np.full((n, 3), np.nan, dtype=np.float64)
        for i, root in enumerate(root_ids):
            xyz = somas.get(int(root))
            if xyz is not None:
                raw[i] = xyz
        finite = np.isfinite(raw).all(axis=1)
        self.n_soma = int(finite.sum())
        self.n_schematic = int((~finite).sum())
        if self.n_soma:
            center = np.median(raw[finite], axis=0)
            shifted = raw - center
            dist = np.linalg.norm(shifted[finite], axis=1)
            scale = float(np.percentile(dist, 95)) or 1.0
            shifted /= scale
        else:
            shifted = raw
        unit = np.zeros((n, 3), dtype=np.float32)
        if self.n_soma:
            unit[finite] = shifted[finite].astype(np.float32)
        for i in range(n):
            if finite[i]:
                continue
            left = str(sides[i]) == "left"
            unit[i] = _schematic(i, left, _lobe(str(types[i])))
        self.source = "flywire-soma" if self.n_schematic == 0 and self.n_soma else (
            "schematic" if self.n_soma == 0 else "flywire-soma+schematic"
        )
        left = np.array([str(s) == "left" for s in sides])

        def take(idx):
            idx = np.asarray(idx, dtype=np.int32)
            return unit[idx], left[idx].astype(np.uint8)

        self.kc_pos, kc_left = take(kc_idx)
        self.kc_side = kc_left
        self.kc_lobe = np.array([_lobe(str(types[int(g)])) for g in kc_idx], dtype=np.uint8)
        self.mbon_pos, self.mbon_side = take(mbon_idx)
        self.pam_pos, self.pam_side = take(app) if len(app) else (np.zeros((0, 3), np.float32), np.zeros(0, np.uint8))
        self.ppl_pos, self.ppl_side = take(av) if len(av) else (np.zeros((0, 3), np.float32), np.zeros(0, np.uint8))
        g2kc = {int(g): i for i, g in enumerate(kc_idx)}
        g2mbon = {int(g): i for i, g in enumerate(mbon_idx)}
        lp, lq = [], []
        for i in range(len(pre)):
            a = g2kc.get(int(pre[i]))
            b = g2mbon.get(int(post[i]))
            if a is None or b is None:
                continue
            lp.append(a)
            lq.append(b)
        self.edge_pre = np.asarray(lp, dtype=np.int32)
        self.edge_post = np.asarray(lq, dtype=np.int32)

    @property
    def n_kc(self) -> int:
        return int(len(self.kc_pos))


def _gain(value: float) -> float:
    mag = abs(float(value))
    return 0.28 + 0.72 * (mag / (mag + 40.0))


def _mix(a, b, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _project(pos, yaw: float, pitch: float, dist: float, cx: float, cy: float, fov: float):
    if len(pos) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    x = pos[:, 0].astype(np.float64)
    y = pos[:, 1].astype(np.float64)
    z = pos[:, 2].astype(np.float64)
    cyaw, syaw = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x1 = x * cyaw - z * syaw
    z1 = x * syaw + z * cyaw
    y2 = y * cp - z1 * sp
    z2 = y * sp + z1 * cp + dist
    scale = fov / np.maximum(z2, 0.15)
    return cx + x1 * scale, cy - y2 * scale, z2


def _flags(n: int, ids) -> np.ndarray:
    on = np.zeros(n, dtype=bool)
    if not ids:
        return on
    arr = np.asarray(list(ids), dtype=np.int64)
    arr = arr[(arr >= 0) & (arr < n)]
    on[arr] = True
    return on


def paint(surface, cloud: BrainCloud, packet: dict, yaw: float, pitch: float, dist: float) -> None:
    """Software projection. No OpenGL, so the same pygame build as the trainer is enough."""
    import pygame

    w, h = surface.get_size()
    bg = pygame.Surface((w, h))
    for row in range(h):
        tone = _mix(_BG_TOP, _BG_BOT, row / max(h - 1, 1))
        pygame.draw.line(bg, tone, (0, row), (w, row))
    surface.blit(bg, (0, 0))
    font = pygame.font.Font(None, 20)
    small = pygame.font.Font(None, 17)
    title = "грибовидное тело  ·  soma FlyWire" if cloud.source.startswith("flywire") else "грибовидное тело  ·  схема лобов"
    surface.blit(font.render(title, True, _VALUE), (16, 12))
    note = "Л/П — стороны FlyWire. Персиковый KC — левый глаз, зелёный — правый. MBON: R_L слева, R_R справа. Мышь — поворот, колесо — зум"
    surface.blit(small.render(note, True, _LABEL), (16, 34))
    cx, cy, fov = w * 0.5, h * 0.52, min(w, h) * 0.72
    kc_l = _flags(cloud.n_kc, packet.get("kc_l"))
    kc_r = _flags(cloud.n_kc, packet.get("kc_r"))
    r_l = float(packet.get("r_l") or 0.0)
    r_r = float(packet.get("r_r") or 0.0)
    flash_l = str(packet.get("flash_l") or "")
    flash_r = str(packet.get("flash_r") or "")
    sx, sy, depth = _project(cloud.kc_pos, yaw, pitch, dist, cx, cy, fov)
    order = np.argsort(depth)
    for i in order:
        if depth[i] <= 0.2:
            continue
        x, y = int(sx[i]), int(sy[i])
        if x < -4 or y < -4 or x >= w + 4 or y >= h + 4:
            continue
        left_on = bool(kc_l[i])
        right_on = bool(kc_r[i])
        if left_on and right_on:
            color, radius = _mix(_PEACH, _GREEN, 0.5), 3
        elif left_on:
            color, radius = _PEACH, 3
        elif right_on:
            color, radius = _GREEN, 3
        else:
            color, radius = _KC[int(cloud.kc_lobe[i])], 1
        pygame.draw.circle(surface, color, (x, y), radius)

    def group(pos, side, kind: str):
        if len(pos) == 0:
            return
        px, py, dep = _project(pos, yaw, pitch, dist, cx, cy, fov)
        for i in np.argsort(dep):
            if dep[i] <= 0.2:
                continue
            x, y = int(px[i]), int(py[i])
            anatomical_left = int(side[i]) == 1
            readout = r_l if anatomical_left else r_r
            flash = flash_l if anatomical_left else flash_r
            if kind == "mbon":
                base = _GREEN if readout >= 0 else _RED
                color = _mix((186, 176, 170), base, _gain(readout))
                radius = 3 + int(3 * _gain(readout))
            elif kind == "pam":
                hot = flash == "pam"
                color = _GREEN if hot else (186, 204, 192)
                radius = 5 if hot else 3
            else:
                hot = flash == "ppl1"
                color = _RED if hot else (214, 186, 182)
                radius = 5 if hot else 3
            pygame.draw.circle(surface, color, (x, y), radius)

    group(cloud.mbon_pos, cloud.mbon_side, "mbon")
    group(cloud.pam_pos, cloud.pam_side, "pam")
    group(cloud.ppl_pos, cloud.ppl_side, "ppl1")
    _draw_edges(surface, cloud, packet, yaw, pitch, dist, cx, cy, fov)
    _draw_side_labels(surface, cloud, yaw, pitch, dist, cx, cy, fov, font)


def _draw_edges(surface, cloud, packet, yaw, pitch, dist, cx, cy, fov) -> None:
    import pygame

    edges = packet.get("edges") or []
    if not edges:
        return
    for item in edges[:EDGE_CAP]:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        pre, post, weight = int(item[0]), int(item[1]), float(item[2])
        if pre < 0 or pre >= cloud.n_kc or post < 0 or post >= len(cloud.mbon_pos):
            continue
        pair = np.vstack((cloud.kc_pos[pre], cloud.mbon_pos[post]))
        px, py, dep = _project(pair, yaw, pitch, dist, cx, cy, fov)
        if dep[0] <= 0.2 or dep[1] <= 0.2:
            continue
        color = _GREEN if weight >= 0 else _RED
        pygame.draw.line(surface, color, (int(px[0]), int(py[0])), (int(px[1]), int(py[1])), 1)


def _draw_side_labels(surface, cloud, yaw, pitch, dist, cx, cy, fov, font) -> None:
    import pygame

    for side, text in ((1, "Л"), (0, "П")):
        mask = cloud.kc_side == side
        if not np.any(mask):
            continue
        mean = cloud.kc_pos[mask].mean(axis=0)
        px, py, dep = _project(mean.reshape(1, 3), yaw, pitch, dist, cx, cy, fov)
        if dep[0] <= 0.2:
            continue
        color = _PEACH if side == 1 else _GREEN
        big = pygame.font.Font(None, 34)
        label = big.render(text, True, color)
        surface.blit(label, (int(px[0]) - label.get_width() // 2, int(py[0]) - 42))


def render_frame(cloud: BrainCloud, packet: dict, size=(960, 700), yaw: float = 0.55, pitch: float = 0.42, dist: float = 2.6):
    import pygame

    pygame.font.init()
    surface = pygame.Surface(size)
    paint(surface, cloud, packet, yaw, pitch, dist)
    return surface


def demo_packet(cloud: BrainCloud) -> dict:
    left = np.flatnonzero(cloud.kc_side == 1)
    right = np.flatnonzero(cloud.kc_side == 0)
    return {
        "t": 0.0,
        "r_l": 80.0,
        "r_r": -30.0,
        "kc_l": [int(i) for i in left[::40][:KC_CAP]],
        "kc_r": [int(i) for i in right[::40][:KC_CAP]],
        "flash_l": "pam",
        "flash_r": "ppl1",
        "edges": [],
    }


def viewer_command(port: int = VIEW_PORT) -> list:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--mb-view3d", "--port", str(int(port))]
    return [sys.executable, "-m", "sim.mb_view3d", "--port", str(int(port))]


class BrainView:
    """Owned by the trainer. Spawn and send never wait on a frame."""

    def __init__(self, port: int = VIEW_PORT):
        self.port = int(port)
        self.proc = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self._next = 0.0

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _ensure_sock(self) -> None:
        if self.sock is not None:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def _stop_proc(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            proc.kill()

    def toggle(self) -> str:
        if self.alive:
            self._stop_proc()
            return "окно мозга закрыто"
        self.spawn()
        return "окно мозга открыто"

    def spawn(self) -> None:
        self._stop_proc()
        self._ensure_sock()
        self.proc = subprocess.Popen(viewer_command(self.port), stdin=subprocess.DEVNULL)
        self._next = 0.0

    def send(self, packet: dict, now: float | None = None, force: bool = False) -> None:
        if self.sock is None or (not force and not self.alive):
            return
        now = time.monotonic() if now is None else float(now)
        if not force and now < self._next:
            return
        self._next = now + (1.0 / RATE_HZ)
        raw = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        if len(raw) > 60000:
            return
        try:
            self.sock.sendto(raw, (HOST, self.port))
        except OSError:
            pass

    def close(self) -> None:
        self._stop_proc()
        sock = self.sock
        self.sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _sample_edges(brain, active, side: int) -> list:
    if brain is None or not active:
        return []
    ids = np.asarray(active, dtype=np.int32)
    mask = np.isin(brain.kc_mbon_pre, ids)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    if idx.size > EDGE_CAP:
        idx = idx[:EDGE_CAP]
    out = []
    for i in idx:
        out.append([int(brain.kc_mbon_pre[i]), int(brain.kc_mbon_post[i]), round(float(brain.kc_mbon_w[i]), 4), int(side)])
    return out


def _flash_kind(session, side: str, now: float) -> str:
    show = str(getattr(session, "teacher_flash_%s_show" % side, "") or "")
    if show in ("pam", "ppl1"):
        return show
    mb = getattr(session, "mb", None)
    flash = None
    if mb is not None:
        flash = mb.flash if side == "l" else getattr(mb, "flash_r", None)
    if flash is not None and str(getattr(flash, "kind", "")) in ("pam", "ppl1"):
        if now - float(getattr(flash, "t", -1e9)) <= 0.5:
            return str(flash.kind)
    if not getattr(session, "onboard", False):
        return ""
    remote = getattr(session, "remote", None)
    if not isinstance(remote, dict):
        return ""
    data = remote.get("flash" if side == "l" else "flash_r")
    if not isinstance(data, dict):
        return ""
    kind = str(data.get("kind") or "")
    if kind not in ("pam", "ppl1"):
        return ""
    token = (kind, data.get("t"), data.get("n_syn"))
    seen = getattr(session, "_brain_flash_seen", None)
    if seen is None:
        seen = {}
        session._brain_flash_seen = seen
    untils = getattr(session, "_brain_flash_until", None)
    if untils is None:
        untils = {}
        session._brain_flash_until = untils
    if seen.get(side) != token:
        seen[side] = token
        untils[side] = now + 0.5
    if now <= float(untils.get(side) or 0.0):
        return kind
    return ""


def _as_ids(value) -> list:
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value[:KC_CAP]:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def packet_from_session(session, now: float) -> dict:
    """KC lists, readouts, compartment flashes. Weights only from a local brain."""
    remote = getattr(session, "remote", None) if getattr(session, "onboard", False) else None
    mb = getattr(session, "mb", None)
    edges: list = []
    if isinstance(remote, dict) and remote:
        kc_l = _as_ids(remote.get("kc_l"))
        kc_r = _as_ids(remote.get("kc_r"))
        r_l = float(remote.get("r_l") or 0.0)
        r_r = float(remote.get("r_r") or 0.0)
    elif mb is not None:
        kc_l = compact_ids(None if mb.last_fwd is None else mb.last_fwd.kc)
        kc_r = compact_ids(None if mb.last_fwd_r is None else mb.last_fwd_r.kc)
        r_l = float(mb.r_l)
        r_r = float(mb.r_r)
        edges = _sample_edges(mb.brain, kc_l, 0) + _sample_edges(mb.brain_r, kc_r, 1)
        edges = edges[:EDGE_CAP]
    else:
        kc_l, kc_r, r_l, r_r = [], [], 0.0, 0.0
    return {
        "t": float(now),
        "r_l": r_l,
        "r_r": r_r,
        "kc_l": kc_l,
        "kc_r": kc_r,
        "flash_l": _flash_kind(session, "l", now),
        "flash_r": _flash_kind(session, "r", now),
        "edges": edges,
    }


def drive_brain(session, inp, now: float, mon=None) -> None:
    if getattr(session, "learner_kind", "") != "mb":
        return
    link = getattr(session, "_brain_view", None)
    if link is None:
        link = BrainView()
        session._brain_view = link
    if getattr(inp, "brain_toggle", False):
        session._log(link.toggle())
    if link.alive:
        link.send(packet_from_session(session, now), now)
    if mon is not None:
        mon.brain_open = bool(link.alive)


def close_brain(session) -> None:
    link = getattr(session, "_brain_view", None)
    if link is not None:
        link.close()


def _recv_latest(sock) -> dict | None:
    latest = None
    while True:
        try:
            raw, _addr = sock.recvfrom(65535)
        except BlockingIOError:
            break
        except OSError:
            break
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            latest = data
    return latest


def run_viewer(port: int = VIEW_PORT) -> int:
    import pygame

    pygame.init()
    pygame.display.set_caption("Грибовидное тело")
    window = pygame.display.set_mode((960, 700), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    window.fill(_BG_TOP)
    window.blit(font.render("загрузка координат FlyWire…", True, _VALUE), (24, 24))
    pygame.display.flip()
    cloud = BrainCloud()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((HOST, int(port)))
    except OSError as exc:
        print("окно мозга: порт %s занят (%s)" % (port, exc), file=sys.stderr)
        return 1
    sock.setblocking(False)
    packet = {"r_l": 0.0, "r_r": 0.0, "kc_l": [], "kc_r": [], "flash_l": "", "flash_r": "", "edges": []}
    yaw, pitch, dist = 0.55, 0.42, 2.6
    drag = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_j):
                return 0
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                drag = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drag = None
            elif event.type == pygame.MOUSEMOTION and drag is not None and event.buttons[0]:
                dx = event.pos[0] - drag[0]
                dy = event.pos[1] - drag[1]
                drag = event.pos
                yaw += dx * 0.008
                pitch = max(-1.2, min(1.2, pitch + dy * 0.008))
            elif event.type == pygame.MOUSEWHEEL:
                dist = max(1.2, min(8.0, dist - event.y * 0.18))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
                dist = max(1.2, min(8.0, dist + (-0.18 if event.button == 4 else 0.18)))
            elif event.type == pygame.VIDEORESIZE:
                window = pygame.display.set_mode((max(320, event.w), max(240, event.h)), pygame.RESIZABLE)
        fresh = _recv_latest(sock)
        if fresh is not None:
            packet = fresh
        paint(window, cloud, packet, yaw, pitch, dist)
        pygame.display.flip()
        clock.tick(30)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="3D mushroom body window")
    parser.add_argument("--mb-view3d", action="store_true")
    parser.add_argument("--port", type=int, default=VIEW_PORT)
    parser.add_argument("--shot", default="")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)
    if args.shot:
        import pygame

        pygame.init()
        cloud = BrainCloud()
        packet = demo_packet(cloud)
        frame = render_frame(cloud, packet)
        dest = Path(args.shot)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(frame, str(dest))
        return 0
    return run_viewer(args.port)


if __name__ == "__main__":
    sys.exit(main())
