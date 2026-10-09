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

from .hemifield import hemifield
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
        # Same connectome, own KC→MBON weights. MB_L is ``brain``, MB_R is ``brain_r``.
        self.brain_r = MushroomBodyRuntime(npz, eta=eta, seed=seed)
        self.proj = RawProjector(self.brain.n_pn, seed=seed + 17)
        self.learn = True
        self.drift_curve: list[tuple[float, float]] = []
        self.dan_events: list[tuple[float, str]] = []
        self.last_drift = 0.0
        self.last_readout = 0.0
        self.last_kc_on = 0
        self.last_kc_bins = np.zeros(48, dtype=np.float32)
        self.last_fwd: MBForward | None = None
        self.last_fwd_r: MBForward | None = None
        self.r_l = 0.0
        self.r_r = 0.0
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
        self.flash_r: LearnFlash | None = None

    def _drive(self, brain: MushroomBodyRuntime, feat: np.ndarray) -> tuple[MBForward | None, float]:
        """One hemisphere. An empty half is a zero readout and does not run KC."""
        if not np.any(feat):
            return None, 0.0
        pn = self.proj.project(feat)
        pn_o, kc, mbon, scores = brain.readout({}, pn)
        fwd = MBForward(pn=pn_o, kc=kc, mbon=mbon, action="Freeze", action_scores=scores)
        return fwd, self._value(brain, fwd)

    def _value(self, brain: MushroomBodyRuntime, fwd: MBForward) -> float:
        if self.dan == "familiarity":
            return -brain.novelty_drive(fwd.mbon)
        return brain.appetitive_drive(fwd.action_scores)

    def forward(self, feat: np.ndarray) -> MBForward:
        """Both halves. The stored readout is their sum, which is what «УЗНАЮ» uses."""
        fwd_l, self.r_l = self._drive(self.brain, hemifield(feat, "L"))
        fwd_r, self.r_r = self._drive(self.brain_r, hemifield(feat, "R"))
        if fwd_l is None:
            fwd_l = self._silent(self.brain)
        if fwd_r is None:
            fwd_r = self._silent(self.brain_r)
        self.last_fwd = fwd_l
        self.last_fwd_r = fwd_r
        self.last_kc_on = int(np.count_nonzero(fwd_l.kc) + np.count_nonzero(fwd_r.kc))
        self.last_kc_bins = bin_kc(fwd_l.kc)
        self.last_readout = float(self.r_l + self.r_r)
        self.brain.last_forward = fwd_l
        self.brain_r.last_forward = fwd_r
        return fwd_l

    def _silent(self, brain: MushroomBodyRuntime) -> MBForward:
        scores = np.zeros(6, dtype=np.float32)
        return MBForward(
            pn=np.zeros(brain.n_pn, dtype=np.float32),
            kc=np.zeros(brain.n_kc, dtype=np.float32),
            mbon=np.zeros(brain.n_mbon, dtype=np.float32),
            action="Freeze",
            action_scores=scores,
        )

    def readout_of(self, fwd: MBForward) -> float:
        if self.dan == "familiarity":
            # Quiet novelty MBON means the pattern has been depressed by repetition.
            return -self.brain.novelty_drive(fwd.mbon)
        return self.brain.appetitive_drive(fwd.action_scores)

    def score_feature(self, feat: np.ndarray) -> float:
        """Readout only. Does not teach and does not replace the full-frame state.

        A sector window is split the same way as a full frame. The mark uses the sum.
        """
        _fwd_l, left = self._drive(self.brain, hemifield(feat, "L"))
        _fwd_r, right = self._drive(self.brain_r, hemifield(feat, "R"))
        return float(left + right)

    def teach(self, fwd: MBForward, kind: str | None, t: float) -> None:
        """Apply a DAN. ``kind`` is pam / ppl1 / None. Never a dog/no-dog label."""
        if not self.learn:
            return
        if self.dan == "teacher":
            if kind in ("pam", "ppl1"):
                # One dopamine pulse, both mushroom bodies, each on its own KC.
                left = self.last_fwd if self.last_fwd is not None else fwd
                right = self.last_fwd_r if self.last_fwd_r is not None else left
                self.flash = record_teacher_step(self.brain, left, kind, t)
                self.flash_r = record_teacher_step(self.brain_r, right, kind, t)
                if kind == "pam":
                    self.dan_events.append((t, "PAM"))
                    # One pulse trains both halves. stat_tot stores this shared count.
                    self.n_pam += 1
                else:
                    self.dan_events.append((t, "PPL1"))
                    self.n_ppl1 += 1
        else:
            fired = False
            for brain, half in ((self.brain, fwd), (self.brain_r, self.last_fwd_r)):
                if half is None or half.kc.sum() <= 0:
                    continue
                novelty = brain.plasticity_familiarity(half.kc)
                fired = fired or novelty < 0.85
            if fired:
                self.n_novelty += 1
                if self.n_novelty % 4 == 0:
                    self.dan_events.append((t, "novelty"))
        self.last_drift = self.brain.weight_drift() + self.brain_r.weight_drift()
        if self.last_fwd is not None and self.last_fwd_r is not None:
            self.r_l = self._value(self.brain, self.last_fwd)
            self.r_r = self._value(self.brain_r, self.last_fwd_r)
            self.last_readout = float(self.r_l + self.r_r)

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
        self.brain_r.reset_plastic()
        self.flash = None
        self.flash_r = None
        self.r_l = 0.0
        self.r_r = 0.0
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
        extra["kc_mbon_w_r"] = np.asarray(self.brain_r.kc_mbon_w, dtype=np.float32)
        extra["kc_fam_r"] = np.asarray(self.brain_r.kc_fam, dtype=np.float32)
        self.brain.save_mb(path, self.seed, self.dan, extra=extra)

    def _take_right(self, path: Path) -> None:
        """Right-hand weights. An older single-MB file is copied into both halves."""
        with open_npz(path) as z:
            if "kc_mbon_w_r" in z.files:
                w = np.asarray(z["kc_mbon_w_r"], dtype=np.float32)
                if w.shape == self.brain_r.kc_mbon_w.shape:
                    self.brain_r.kc_mbon_w = w
                else:
                    self.brain_r.kc_mbon_w = self.brain.kc_mbon_w.copy()
                fam = np.asarray(z["kc_fam_r"], dtype=np.float32) if "kc_fam_r" in z.files else None
                if fam is not None and fam.shape == self.brain_r.kc_fam.shape:
                    self.brain_r.kc_fam = fam
                else:
                    self.brain_r.kc_fam = self.brain.kc_fam.copy()
            else:
                self.brain_r.kc_mbon_w = self.brain.kc_mbon_w.copy()
                self.brain_r.kc_fam = self.brain.kc_fam.copy()

    def load(self, path: Path) -> None:
        seed, dan = self.brain.load_mb(path)
        self.seed = seed
        self.proj = RawProjector(self.brain.n_pn, seed=seed + 17)
        if dan in ("teacher", "familiarity"):
            self.dan = dan
        self._take_right(path)
        self.last_drift = self.brain.weight_drift() + self.brain_r.weight_drift()
        self.flash = None
        self.flash_r = None
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
