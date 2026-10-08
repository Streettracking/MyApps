"""Unsupervised conspecific recognition between raw sensors and PN.

The layer never sees zone reward or a class label. It segments blobs from the
camera/lidar vector, and a self-similarity gate — own walking speed and own
body width — decides which blobs may update the conspecific prototype.
Appearance (which image row lights up) is learned by Hebbian update.
A second prototype collects the blobs that fail the gate.

The scalar conspecific-likeness is what replaces a fixed detector.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .raw_sense import (
    FOV,
    N_AZ,
    N_RAW,
    OFF_BODY,
    OFF_CLOSE,
    OFF_FLOOR,
    OFF_LIDAR,
    OFF_MOTION,
    SEE_M,
    _paint_object,
)

OWN_RADIUS = 0.28
SECTOR = FOV / N_AZ
FEAT_DIM = 7


def _predict_bins(dist: float) -> float:
    half = int(np.floor((OWN_RADIUS / max(dist, 0.35)) / SECTOR))
    return float(2 * half + 1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_blobs(feat: np.ndarray) -> list[dict]:
    """Contiguous active sectors. Zone-wide floor color is subtracted first."""
    body = feat[OFF_BODY : OFF_BODY + 3 * N_AZ].reshape(N_AZ, 3)
    floor = feat[OFF_FLOOR : OFF_FLOOR + 3 * N_AZ].reshape(N_AZ, 3)
    floor_res = np.clip(floor - np.median(floor, axis=0), 0.0, None)
    body_e = body.sum(axis=1)
    floor_e = floor_res.sum(axis=1)
    motion = feat[OFF_MOTION : OFF_MOTION + N_AZ]
    near = feat[OFF_LIDAR : OFF_LIDAR + N_AZ]
    closing = feat[OFF_CLOSE : OFF_CLOSE + N_AZ]
    active = (body_e > 0.08) | (floor_e > 0.08) | (motion > 0.08)
    idxs = np.flatnonzero(active)
    if len(idxs) == 0:
        return []
    groups: list[list[int]] = [[int(idxs[0])]]
    for b in idxs[1:]:
        if int(b) == groups[-1][-1] + 1:
            groups[-1].append(int(b))
        else:
            groups.append([int(b)])
    blobs = []
    for group in groups:
        g = np.array(group)
        # Lit bins dominate. Uniform sim blobs are unchanged because every bin matches.
        w = body_e[g] + floor_e[g] + motion[g]
        if float(w.sum()) < 1e-4:
            w = np.ones(len(g), dtype=np.float32)
        bsum = float(np.average(body_e[g], weights=w))
        fsum = float(np.average(floor_e[g], weights=w))
        near_m = float(np.average(near[g], weights=w))
        mot = float(np.average(motion[g], weights=w))
        dist = float(np.clip((1.0 - near_m) * SEE_M, 0.35, SEE_M))
        est_speed = (mot / max(near_m, 0.15)) * 0.8
        desc = np.array(
            [
                bsum,
                fsum,
                bsum / (bsum + fsum + 1e-3),
                mot,
                near_m,
                len(group) / N_AZ,
                float(np.average(closing[g], weights=w)),
            ],
            dtype=np.float32,
        )
        blobs.append(
            {
                "az": int(np.mean(group)),
                "n_bins": len(group),
                "est_speed": est_speed,
                "dist": dist,
                "desc": desc,
            }
        )
    return blobs


def probe_view(kind: str) -> np.ndarray:
    """Canonical unlabeled views. Speed and row differ; there is no class bit."""
    specs = {
        "dog": dict(radius=0.28, color=(0.45, 0.62, 0.95), speed=0.45, body=True),
        "dog_still": dict(radius=0.28, color=(0.45, 0.62, 0.95), speed=0.02, body=True),
        "dist": dict(radius=0.12, color=(0.55, 0.38, 0.16), speed=0.12, body=False),
        "dist_fast": dict(radius=0.12, color=(0.55, 0.38, 0.16), speed=0.45, body=False),
        "static": dict(radius=0.12, color=(0.42, 0.42, 0.40), speed=0.0, body=False),
    }
    feat = np.zeros(N_RAW, dtype=np.float32)
    spec = specs[kind]
    _paint_object(feat, bx=N_AZ // 2, dist=1.4, closing=0.05, **spec)
    return feat


class ConspecificRecognizer:
    def __init__(self, lr: float = 0.2):
        self.lr = lr
        self.self_speed = 0.40
        self.proto = np.zeros((2, FEAT_DIM), dtype=np.float32)
        self.n_self = 0
        self.n_other = 0
        self.prev_az: int | None = None
        self.prev_desc: np.ndarray | None = None
        self.last_likeness = 0.0
        self.last_gate = 0.0

    def _gate(self, blob: dict) -> float:
        expected = _predict_bins(blob["dist"])
        size_z = (blob["n_bins"] - expected) / 1.25
        speed_z = (blob["est_speed"] - self.self_speed) / 0.18
        return float(np.exp(-0.5 * (size_z * size_z + speed_z * speed_z)))

    def _likeness_desc(self, desc: np.ndarray, gate: float) -> float:
        if self.n_self < 4:
            return gate
        return float(np.clip(_cosine(desc, self.proto[0]), 0.0, 1.0))

    def reset(self) -> None:
        """Drop learned prototypes. Gait prior returns to the walking default."""
        self.proto[:] = 0.0
        self.n_self = 0
        self.n_other = 0
        self.prev_az = None
        self.prev_desc = None
        self.last_likeness = 0.0
        self.last_gate = 0.0
        self.self_speed = 0.40

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "lr": self.lr,
            "self_speed": self.self_speed,
            "proto": self.proto.astype(float).tolist(),
            "n_self": int(self.n_self),
            "n_other": int(self.n_other),
        }

    def load_state_dict(self, data: dict) -> None:
        proto = np.asarray(data["proto"], dtype=np.float32)
        if proto.shape != (2, FEAT_DIM):
            raise ValueError(f"recognizer proto shape {proto.shape}, expected (2, {FEAT_DIM})")
        self.lr = float(data.get("lr", self.lr))
        self.self_speed = float(data.get("self_speed", self.self_speed))
        self.proto = proto
        self.n_self = int(data.get("n_self", 0))
        self.n_other = int(data.get("n_other", 0))
        self.prev_az = None
        self.prev_desc = None

    def save(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")
        return dest

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.load_state_dict(data)

    def learned_match(self, feat: np.ndarray) -> float:
        """Cosine to the conspecific prototype. Zero until that prototype exists.

        This is the score the monitor plots. It is not a class label.
        """
        if float(np.linalg.norm(self.proto[0])) < 1e-6:
            return 0.0
        blobs = extract_blobs(feat)
        if not blobs:
            return 0.0
        return max(float(np.clip(_cosine(b["desc"], self.proto[0]), 0.0, 1.0)) for b in blobs)

    def note_speed(self, speed: float) -> None:
        """Proprioception. Only walking updates the gait prior, not sitting still."""
        if speed > 0.15:
            self.self_speed = 0.98 * self.self_speed + 0.02 * float(speed)

    def observe(self, feat: np.ndarray, speed: float) -> float:
        self.note_speed(speed)
        blobs = extract_blobs(feat)
        if not blobs:
            self.last_likeness = 0.0
            self.last_gate = 0.0
            return 0.0
        best_l = 0.0
        best_g = 0.0
        for blob in blobs:
            gate = self._gate(blob)
            desc = blob["desc"]
            coh = 1.0
            if (
                self.prev_az is not None
                and abs(blob["az"] - self.prev_az) <= 1
                and self.prev_desc is not None
                and _cosine(desc, self.prev_desc) > 0.75
            ):
                coh = 1.5
            if gate > 0.55:
                a = min(0.6, self.lr * gate * coh)
                self.proto[0] = (1.0 - a) * self.proto[0] + a * desc
                self.n_self += 1
            elif gate < 0.30:
                a = min(0.4, 0.08 * coh)
                self.proto[1] = (1.0 - a) * self.proto[1] + a * desc
                self.n_other += 1
            like = self._likeness_desc(desc, gate)
            if like > best_l:
                best_l = like
                best_g = gate
                self.prev_az = blob["az"]
                self.prev_desc = desc.copy()
        self.last_likeness = best_l
        self.last_gate = best_g
        return best_l

    def likeness(self, feat: np.ndarray) -> float:
        """Score a view without learning."""
        blobs = extract_blobs(feat)
        if not blobs:
            return 0.0
        return max(self._likeness_desc(b["desc"], self._gate(b)) for b in blobs)

    def snapshot(self) -> dict:
        """Separability of held-out views. Labels are used only here, as a metric."""
        views = {k: probe_view(k) for k in ("dog", "dog_still", "dist", "dist_fast", "static")}
        scores = {k: self.likeness(v) for k, v in views.items()}

        def nearest_self(feat: np.ndarray) -> bool:
            blobs = extract_blobs(feat)
            if not blobs:
                return False
            desc = max(blobs, key=lambda b: _cosine(b["desc"], self.proto[0]))["desc"]
            # An empty "other" prototype has cosine 0, so every blob would look like self.
            if self.n_other < 4 or float(np.linalg.norm(self.proto[1])) < 1e-3:
                return _cosine(desc, self.proto[0]) > 0.55
            return _cosine(desc, self.proto[0]) >= _cosine(desc, self.proto[1])

        if self.n_self < 4:
            dog_self = scores["dog"] > 0.55
            dist_other = scores["dist"] <= 0.55
            static_other = scores["static"] <= 0.55
        else:
            dog_self = nearest_self(views["dog"])
            dist_other = not nearest_self(views["dist"])
            static_other = not nearest_self(views["static"])
        return {
            "like_dog": scores["dog"],
            "like_dist": scores["dist"],
            "like_static": scores["static"],
            "like_dog_still": scores["dog_still"],
            "like_dist_fast": scores["dist_fast"],
            "sep_moving": scores["dog"] - scores["dist"],
            "sep_static": scores["dog"] - scores["static"],
            "invariance": scores["dog_still"] - scores["dist_fast"],
            "purity": float(dog_self) + float(dist_other) + float(static_other),
            "n_self": self.n_self,
        }
