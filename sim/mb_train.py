"""Mushroom-body recognition training: raw features → PN → KC → MBON.

Only KC→MBON weights change, and only when a DAN teaching signal is present.
Default DAN is the operator: T injects appetitive PAM, X injects aversive PPL1.
Familiarity is the other DAN: repetition depresses KC→novelty-MBON.
Monitor labels never enter either rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .learn_flash import LearnFlash, MbLayout, build_layout, record_teacher_step
from .mb_confidence import Confidence, ConfidenceCalibrator, TrainProgress
from .mb_runtime import MBForward, MushroomBodyRuntime
from .npz_compat import open_npz
from .raw_sense import RawProjector, probe_features

ROOT = Path(__file__).resolve().parents[1]


def default_npz() -> Path:
    if getattr(sys, "frozen", False):
        folder = Path(getattr(sys, "_MEIPASS")) / "artifacts"
    else:
        folder = ROOT / "artifacts"
    plain = folder / "connectome_mb_v1_np1.npz"
    if plain.is_file():
        return plain
    return folder / "connectome_mb_v1.npz"


def bin_kc(kc: np.ndarray, n: int = 48) -> np.ndarray:
    acc = np.zeros(n, dtype=np.float32)
    if len(kc) == 0:
        return acc
    idx = (np.arange(len(kc)) * n) // len(kc)
    np.add.at(acc, idx, kc.astype(np.float32, copy=False))
    counts = np.bincount(idx, minlength=n).astype(np.float32)
    acc /= np.maximum(counts, 1.0)
    return acc


class MbTrainer:
    def __init__(self, npz: Path, seed: int = 1, eta: float = 0.2, dan: str = "teacher"):
        if dan not in ("teacher", "familiarity"):
            raise ValueError(dan)
        self.dan = dan
        self.seed = int(seed)
        self.brain = MushroomBodyRuntime(npz, eta=eta, seed=seed)
        self.proj = RawProjector(self.brain.n_pn, seed=seed + 17)
        self.learn = True
        self.drift_curve: list[tuple[float, float]] = []
        self.dan_events: list[tuple[float, str]] = []
        self.last_drift = 0.0
        self.last_readout = 0.0
        self.last_kc_on = 0
        self.last_kc_bins = np.zeros(48, dtype=np.float32)
        self.last_fwd: MBForward | None = None
        self.probe_init = self.probe()
        self.n_pam = 0
        self.n_ppl1 = 0
        self.n_novelty = 0
        self.cal = ConfidenceCalibrator()
        self.progress = TrainProgress()
        self.conf = Confidence()
        self.lidar_refresh = 0.0
        self.saved_lidar_refresh: float | None = None
        self.layout: MbLayout = build_layout(self.brain)
        self.flash: LearnFlash | None = None

    def forward(self, feat: np.ndarray) -> MBForward:
        pn = self.proj.project(feat)
        pn_o, kc, mbon, scores = self.brain.readout({}, pn)
        fwd = MBForward(pn=pn_o, kc=kc, mbon=mbon, action="Freeze", action_scores=scores)
        self.last_fwd = fwd
        self.last_kc_on = int(np.count_nonzero(kc))
        self.last_kc_bins = bin_kc(kc)
        self.last_readout = self.readout_of(fwd)
        self.brain.last_forward = fwd
        return fwd

    def readout_of(self, fwd: MBForward) -> float:
        if self.dan == "familiarity":
            # Quiet novelty MBON means the pattern has been depressed by repetition.
            return -self.brain.novelty_drive(fwd.mbon)
        return self.brain.appetitive_drive(fwd.action_scores)

    def score_feature(self, feat: np.ndarray) -> float:
        """Readout only. Does not teach and does not replace the full-frame state."""
        pn = self.proj.project(feat)
        pn_o, kc, mbon, scores = self.brain.readout({}, pn)
        fwd = MBForward(pn=pn_o, kc=kc, mbon=mbon, action="Freeze", action_scores=scores)
        return self.readout_of(fwd)

    def teach(self, fwd: MBForward, kind: str | None, t: float) -> None:
        """Apply a DAN. ``kind`` is pam / ppl1 / None. Never a dog/no-dog label."""
        if not self.learn:
            return
        if self.dan == "teacher":
            if kind == "pam":
                self.flash = record_teacher_step(self.brain, fwd, "pam", t)
                self.dan_events.append((t, "PAM"))
                self.n_pam += 1
            elif kind == "ppl1":
                self.flash = record_teacher_step(self.brain, fwd, "ppl1", t)
                self.dan_events.append((t, "PPL1"))
                self.n_ppl1 += 1
        elif fwd.kc.sum() > 0:
            novelty = self.brain.plasticity_familiarity(fwd.kc)
            if novelty < 0.85:
                self.n_novelty += 1
                if self.n_novelty % 4 == 0:
                    self.dan_events.append((t, "novelty"))
        self.last_drift = self.brain.weight_drift()
        self.last_readout = self.readout_of(fwd)

    def observe(self, feat: np.ndarray, readout: float) -> Confidence:
        """Confidence from the readout and the frame's own energy. No labels."""
        energy = float(np.mean(np.abs(np.asarray(feat, dtype=np.float32))))
        self.conf = self.cal.update(readout, energy)
        return self.conf

    def note_drift(self, t: float) -> None:
        self.drift_curve.append((t, self.last_drift))

    def probe(self) -> dict[str, float]:
        """Canonical views. Does not teach."""
        out = {}
        for name in ("empty", "dog", "distractor"):
            fwd = self.forward(probe_features(name))
            out[name] = self.readout_of(fwd)
        return out

    def separation(self, final: dict[str, float] | None = None) -> float:
        """How far the dog-minus-distractor readout moved from its own start."""
        end = final if final is not None else self.probe()
        init_gap = self.probe_init["dog"] - self.probe_init["distractor"]
        end_gap = end["dog"] - end["distractor"]
        return float(end_gap - init_gap)

    def reset(self) -> None:
        self.brain.reset_plastic()
        self.flash = None
        self.last_drift = 0.0
        self.n_pam = self.n_ppl1 = self.n_novelty = 0
        self.cal.reset()
        self.progress.reset()
        self.conf = Confidence()
        self.probe_init = self.probe()

    def save(self, path: Path) -> None:
        extra = self.progress.export_arrays(self.n_pam, self.n_ppl1, self.n_novelty)
        extra.update(self.cal.export_arrays())
        extra["lidar_refresh"] = np.float32(self.lidar_refresh)
        self.brain.save_mb(path, self.seed, self.dan, extra=extra)

    def load(self, path: Path) -> None:
        seed, dan = self.brain.load_mb(path)
        self.seed = seed
        self.proj = RawProjector(self.brain.n_pn, seed=seed + 17)
        if dan in ("teacher", "familiarity"):
            self.dan = dan
        self.last_drift = self.brain.weight_drift()
        self.flash = None
        with open_npz(path) as z:
            self.progress.load_arrays(z)
            self.cal.load_arrays(z)
            if "lidar_refresh" in z.files:
                self.saved_lidar_refresh = float(np.asarray(z["lidar_refresh"]).ravel()[0])
            else:
                self.saved_lidar_refresh = None
        self.conf = self.cal.last
        self.n_pam = self.n_ppl1 = self.n_novelty = 0
        self.probe_init = self.probe()
