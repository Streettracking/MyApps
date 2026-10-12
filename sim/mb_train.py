"""Mushroom-body recognition training: two visual roads into KC, then MBON.

Medulla features drive γ-d and lobula features drive α/β-p. KC→MBON weights
of γ-d change when a DAN fires, and they also ease back toward the birth
weight on later frames. α/β-p does not learn. Default DAN is the operator:
T injects appetitive PAM, X injects aversive PPL1. Familiarity is the other
DAN: repetition depresses KC→novelty-MBON, still only from γ-d. Monitor
labels never enter either rule. There is no wire from a motor command into
KC. Reinforcement is whatever stimulus is attached; this trainer does not
name the source. The stored readouts are approach minus avoidance. The turn
is the left readout minus the right one.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np

from .hemifield import DEFAULT_OVERLAP, clamp_overlap, hemifield
from .kc_pathways import VisualPathway, lobula_channels, medulla_channels
from .raw_sense import drop_lidar
from .learn_flash import LearnFlash, MbLayout, build_layout, record_teacher_step
from .mb_confidence import Confidence, ConfidenceCalibrator, TrainProgress
from .mb_runtime import MBForward, MushroomBodyRuntime
from .npz_compat import open_npz
from .raw_sense import RawProjector, probe_features
from .stimuli import StimulusSet

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


def _pn_trace(n_pn: int, feat: np.ndarray, yolo) -> np.ndarray:
    """The visual channels that reached this eye, packed into the PN vector.

    This is not a random map. An eye with no medulla and no lobula stays at zero.
    """
    packed = np.concatenate((medulla_channels(feat), lobula_channels(feat, yolo)))
    pn = np.zeros(int(n_pn), dtype=np.float32)
    n = min(int(n_pn), int(packed.shape[0]))
    if n:
        pn[:n] = np.clip(packed[:n], 0.0, 1.0)
    return pn


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
    def __init__(
        self,
        npz: Path,
        seed: int = 1,
        eta: float = 0.2,
        dan: str = "teacher",
        overlap: float = DEFAULT_OVERLAP,
        camera_only: bool = True,
    ):
        if dan not in ("teacher", "familiarity"):
            raise ValueError(dan)
        self.dan = dan
        self.seed = int(seed)
        self._birth_dan = self.dan
        self._birth_seed = self.seed
        self.load_note = ""
        self.brain = MushroomBodyRuntime(npz, eta=eta, seed=seed)
        # Same connectome, own KC→MBON weights. MB_L is ``brain``, MB_R is ``brain_r``.
        self.brain_r = MushroomBodyRuntime(npz, eta=eta, seed=seed)
        self.brain.vision = VisualPathway(self.brain, "left")
        self.brain_r.vision = VisualPathway(self.brain_r, "right")
        self.brain.gamma_d_kc = self.brain.vision.gamma_mask
        self.brain_r.gamma_d_kc = self.brain_r.vision.gamma_mask
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
        self.overlap = clamp_overlap(overlap)
        # Set before probe(): the canonical views must use the same input as training.
        self.camera_only = bool(camera_only)
        self.probe_init = self.probe()
        self.n_pam = 0
        self.n_ppl1 = 0
        self.n_novelty = 0
        self.teacher_pam_l = 0
        self.teacher_pam_r = 0
        self.teacher_ppl1_l = 0
        self.teacher_ppl1_r = 0
        self.reinforce = ""
        # Empty until the recognition app plugs a source in. Saving weights
        # does not write this list.
        self.stimuli = StimulusSet()
        self.cal = ConfidenceCalibrator()
        self.cal_l = ConfidenceCalibrator()
        self.cal_r = ConfidenceCalibrator()
        self.progress = TrainProgress()
        self.conf = Confidence()
        self.conf_l = Confidence()
        self.conf_r = Confidence()
        self.lidar_refresh = 0.0
        self.saved_lidar_refresh: float | None = None
        self.layout: MbLayout = build_layout(self.brain)
        self.flash: LearnFlash | None = None
        self.flash_r: LearnFlash | None = None

    def _drive(
        self,
        brain: MushroomBodyRuntime,
        feat: np.ndarray,
        yolo: np.ndarray | None = None,
    ) -> tuple[MBForward | None, float]:
        """One hemisphere. An empty half is a zero readout and does not run KC.

        The KC vector comes from the two visual pathways, not from the random
        PN map. A motor command is not an input.
        """
        if brain.vision is None:
            return None, 0.0
        kc = brain.vision.kenyon(feat, yolo)
        if float(kc.max()) <= 0.0:
            return None, 0.0
        mbon, scores = brain.scores_of_kc(kc)
        fwd = MBForward(
            pn=_pn_trace(brain.n_pn, feat, yolo),
            kc=kc,
            mbon=mbon,
            action="Freeze",
            action_scores=scores,
        )
        return fwd, self._value(brain, fwd)

    def _value(self, brain: MushroomBodyRuntime, fwd: MBForward) -> float:
        if self.dan == "familiarity":
            return -brain.novelty_drive(fwd.mbon)
        return brain.appetitive_drive(fwd.action_scores)

    def _half(self, feat: np.ndarray, side: str) -> np.ndarray:
        return hemifield(feat, side, self.overlap)

    def _sense(self, feat: np.ndarray) -> np.ndarray:
        """Input to both hemispheres. Lidar slots are cleared when camera-only."""
        raw = np.asarray(feat, dtype=np.float32).ravel()
        if not self.camera_only:
            return raw
        return drop_lidar(raw)

    def forward(self, feat: np.ndarray, yolo_l=None, yolo_r=None) -> MBForward:
        """Both halves. The stored readout is their sum, which is what the joint «УЗНАЮ» uses.

        ``yolo_l`` and ``yolo_r`` are optional conf/width/height/center vectors.
        They reach α/β-p only. Leaving them out leaves those four channels at zero.
        """
        feat = self._sense(feat)
        fwd_l, self.r_l = self._drive(self.brain, self._half(feat, "L"), yolo_l)
        fwd_r, self.r_r = self._drive(self.brain_r, self._half(feat, "R"), yolo_r)
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
        feat = self._sense(feat)
        _fwd_l, left = self._drive(self.brain, self._half(feat, "L"), None)
        _fwd_r, right = self._drive(self.brain_r, self._half(feat, "R"), None)
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
                self.reinforce = "operator"
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

    def settle(self, dopamine_l: bool = False, dopamine_r: bool = False) -> None:
        """One frame of homeostasis. Learning off leaves the weights alone.

        Depressed synapses move back toward their birth weight. A Kenyon cell
        that fired with no dopamine on that side gains a little. ``r_l`` and
        ``r_r`` stay the approach-minus-avoidance drives ``forward`` stored.
        """
        if not self.learn:
            return
        if self.last_fwd is not None:
            self.brain.recover(self.last_fwd.kc, bool(dopamine_l))
        if self.last_fwd_r is not None:
            self.brain_r.recover(self.last_fwd_r.kc, bool(dopamine_r))

    def reinforce_stimuli(
        self,
        now: float,
        skip_l: str | None = None,
        skip_r: str | None = None,
        learning: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """Apply the attached stimuli. Their names are not read here.

        ``skip_*`` is a kind already delivered to that eye on this frame.
        ``learning`` defaults to this trainer's flag. A source can be live
        while learning is off; then the returned kinds stay empty.
        """
        if learning is None:
            learning = self.learn
        kind_l, kind_r = self.stimuli.reinforcement(float(now), bool(learning), skip_l, skip_r)
        if (kind_l or kind_r) and self.learn:
            self.teach_sides(kind_l, kind_r, float(now))
        return kind_l, kind_r

    def teach_sides(self, kind_l: str | None, kind_r: str | None, t: float) -> bool:
        """DAN on one hemisphere or each. Operator ``teach`` still pulses both.

        The counters below are not written into ``mb_train_state.npz``.
        """
        if not self.learn or self.dan != "teacher":
            return False
        fired = False
        if kind_l in ("pam", "ppl1") and self.last_fwd is not None:
            self.flash = record_teacher_step(self.brain, self.last_fwd, kind_l, t)
            self._count_teacher("L", kind_l, t)
            fired = True
        if kind_r in ("pam", "ppl1") and self.last_fwd_r is not None:
            self.flash_r = record_teacher_step(self.brain_r, self.last_fwd_r, kind_r, t)
            self._count_teacher("R", kind_r, t)
            fired = True
        if not fired:
            return False
        self.reinforce = "teacher"
        self.last_drift = self.brain.weight_drift() + self.brain_r.weight_drift()
        if self.last_fwd is not None and self.last_fwd_r is not None:
            self.r_l = self._value(self.brain, self.last_fwd)
            self.r_r = self._value(self.brain_r, self.last_fwd_r)
            self.last_readout = float(self.r_l + self.r_r)
        return True

    def _count_teacher(self, side: str, kind: str, t: float) -> None:
        if kind == "pam":
            self.n_pam += 1
            self.dan_events.append((t, "PAM"))
            if side == "L":
                self.teacher_pam_l += 1
            else:
                self.teacher_pam_r += 1
        else:
            self.n_ppl1 += 1
            self.dan_events.append((t, "PPL1"))
            if side == "L":
                self.teacher_ppl1_l += 1
            else:
                self.teacher_ppl1_r += 1

    def observe(self, feat: np.ndarray, readout: float) -> Confidence:
        """Joint confidence, plus one calibrator on each hemisphere. No labels.

        The per-eye windows are not written into ``mb_train_state.npz``. An old
        file still loads, and a new file has the same arrays as before.
        """
        raw = self._sense(feat)
        energy = float(np.mean(np.abs(raw)))
        self.conf = self.cal.update(readout, energy)
        left = self._half(raw, "L")
        right = self._half(raw, "R")
        self.conf_l = self.cal_l.update(self.r_l, float(np.mean(np.abs(left))))
        self.conf_r = self.cal_r.update(self.r_r, float(np.mean(np.abs(right))))
        return self.conf

    def eye_recognized(self) -> tuple[bool, bool]:
        """Latched per-eye words. Each calibrator has its own 0.80 / 0.65 hysteresis."""
        left = bool(self.conf_l.ready and self.conf_l.recognized)
        right = bool(self.conf_r.ready and self.conf_r.recognized)
        return left, right

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
        self.teacher_pam_l = self.teacher_pam_r = 0
        self.teacher_ppl1_l = self.teacher_ppl1_r = 0
        self.reinforce = ""
        self.cal.reset()
        self.cal_l.reset()
        self.cal_r.reset()
        self.progress.reset()
        self.conf = Confidence()
        self.conf_l = Confidence()
        self.conf_r = Confidence()
        self.probe_init = self.probe()

    def save(self, path: Path) -> None:
        extra = self.progress.export_arrays(self.n_pam, self.n_ppl1, self.n_novelty)
        extra.update(self.cal.export_arrays())
        extra["lidar_refresh"] = np.float32(self.lidar_refresh)
        extra["kc_mbon_w_r"] = np.asarray(self.brain_r.kc_mbon_w, dtype=np.float32)
        extra["kc_fam_r"] = np.asarray(self.brain_r.kc_fam, dtype=np.float32)
        self.brain.save_mb(path, self.seed, self.dan, extra=extra)

    def _take_right(self, path: Path) -> None:
        """Right-hand weights. An older single-MB file is copied into both halves.

        A right-hand vector whose shape does not match is a different format.
        The caller then keeps the birth weights and leaves the file on disk.
        """
        with open_npz(path) as z:
            if "kc_mbon_w_r" in z.files:
                w = np.asarray(z["kc_mbon_w_r"], dtype=np.float32)
                if w.shape != self.brain_r.kc_mbon_w.shape:
                    raise ValueError(
                        "форма правого KC→MBON %s не совпала с мозгом %s"
                        % (w.shape, self.brain_r.kc_mbon_w.shape)
                    )
                self.brain_r.kc_mbon_w = w
                self.brain_r._bound_weights()
                fam = np.asarray(z["kc_fam_r"], dtype=np.float32) if "kc_fam_r" in z.files else None
                if fam is not None and fam.shape == self.brain_r.kc_fam.shape:
                    self.brain_r.kc_fam = fam
                else:
                    self.brain_r.kc_fam = self.brain.kc_fam.copy()
            else:
                self.brain_r.kc_mbon_w = self.brain.kc_mbon_w.copy()
                self.brain_r.kc_fam = self.brain.kc_fam.copy()

    def _abandon_load(self) -> None:
        self.seed = self._birth_seed
        self.dan = self._birth_dan
        self.proj = RawProjector(self.brain.n_pn, seed=self.seed + 17)
        self.reset()

    def load(self, path: Path) -> bool:
        """Load KC→MBON weights. False keeps the birth weights and the file.

        A matching file still loads, including an older one that has no
        right-hand vector. A shape that does not match, or a file that
        cannot be read, starts clean and sets ``load_note``.
        """
        self.load_note = ""
        try:
            self._load_file(path)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            self._abandon_load()
            self.load_note = "файл весов не загружен (%s), старт с чистых весов, файл не удалён" % exc
            return False
        return True

    def _load_file(self, path: Path) -> None:
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
        self.teacher_pam_l = self.teacher_pam_r = 0
        self.teacher_ppl1_l = self.teacher_ppl1_r = 0
        self.reinforce = ""
        self.probe_init = self.probe()
