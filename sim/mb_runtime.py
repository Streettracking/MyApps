"""Load FlyWire MB subgraph and run rate-based forward + DAN-masked plasticity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mb_constants import IDLE_GROWTH, RECOVERY_RATE, W_CEIL_FRAC, W_FLOOR_FRAC, kc_winner_count
from .npz_compat import as_plain, open_npz, scalar_str

ACTIONS = ("Approach_A", "Avoid_A", "Approach_B", "Avoid_B", "Explore", "Freeze")


def apl_sparsify(drive: np.ndarray) -> np.ndarray:
    """Global inhibition (APL) as k-winner-take-all.

    A silent frame stays silent. Otherwise the strongest Kenyon cells, about
    7% and always inside 5–10%, keep their rate and the strongest is scaled to 1.
    """
    n = int(drive.shape[0])
    out = np.zeros(n, dtype=np.float32)
    if n == 0 or float(drive.max()) <= 0.0:
        return out
    if n <= 10:
        np.maximum(drive, 0.0, out=out)
        out /= np.float32(out.max())
        return out
    k = kc_winner_count(n)
    pick = np.argpartition(drive, n - k)[n - k :]
    keep = pick[drive[pick] > 0.0]
    if keep.size == 0:
        return out
    out[keep] = drive[keep]
    out /= np.float32(out.max())
    return out


@dataclass
class MBForward:
    pn: np.ndarray
    kc: np.ndarray
    mbon: np.ndarray
    action: str
    action_scores: np.ndarray


class MushroomBodyRuntime:
    def __init__(self, npz_path: str | Path, eta: float = 0.01, seed: int = 0):
        z = open_npz(npz_path)
        self.root_ids = z["root_ids"].astype(np.int64)
        self.roles = z["roles"]
        self.cell_types = z["cell_types"].astype(str)
        if "sides" in z.files:
            self.sides = z["sides"].astype(str)
        else:
            self.sides = np.array(["right"] * len(self.root_ids))
        self.pn_idx = z["pn_idx"].astype(np.int32)
        self.kc_idx = z["kc_idx"].astype(np.int32)
        self.mbon_idx = z["mbon_idx"].astype(np.int32)
        self.n = len(self.root_ids)
        self.n_pn = len(self.pn_idx)
        self.n_kc = len(self.kc_idx)
        self.n_mbon = len(self.mbon_idx)

        # Remap global neuron indices -> role-local indices for compact mats
        self._g2pn = {int(g): i for i, g in enumerate(self.pn_idx)}
        self._g2kc = {int(g): i for i, g in enumerate(self.kc_idx)}
        self._g2mbon = {int(g): i for i, g in enumerate(self.mbon_idx)}

        self.pn_kc_pre, self.pn_kc_post, self.pn_kc_w = self._to_local(
            z["pn_kc_pre"], z["pn_kc_post"], z["pn_kc_w"], self._g2pn, self._g2kc
        )
        self.kc_mbon_pre, self.kc_mbon_post, self.kc_mbon_w0 = self._to_local(
            z["kc_mbon_pre"], z["kc_mbon_post"], z["kc_mbon_w"], self._g2kc, self._g2mbon
        )
        self.kc_mbon_w = self.kc_mbon_w0.astype(np.float32).copy()
        self.w_floor = (self.kc_mbon_w0 * np.float32(W_FLOOR_FRAC)).astype(np.float32)
        self.w_ceil = (self.kc_mbon_w0 * np.float32(W_CEIL_FRAC)).astype(np.float32)
        self.mask_av = z["kc_mbon_mask_aversive"].astype(bool)
        self.mask_ap = z["kc_mbon_mask_appetitive"].astype(bool)
        if len(self.mask_av) != len(self.kc_mbon_w):
            # safety: if masks missing/mismatched, allow all
            self.mask_av = np.ones(len(self.kc_mbon_w), dtype=bool)
            self.mask_ap = np.ones(len(self.kc_mbon_w), dtype=bool)

        self.eta = float(eta)
        self.rng = np.random.default_rng(seed)

        # Fixed sparse PN patterns for cues (individuum sensory encoder)
        self.patterns = {
            "A": self._make_pattern(0.08, seed + 1),
            "B": self._make_pattern(0.08, seed + 2),
            "peer": self._make_pattern(0.05, seed + 3),
            "peer_at_B": self._make_pattern(0.05, seed + 4),
        }

        # Map each MBON to an action bucket by type hash / known valence stubs
        self.mbon_action = self._assign_mbon_actions()
        self.last_forward: MBForward | None = None
        self.last_cues: dict[str, float] = {}
        self.last_raw: np.ndarray | None = None
        # Per-KC occupancy for familiarity depression. Updated only by that rule.
        self.kc_fam = np.zeros(self.n_kc, dtype=np.float32)
        self.n_dan_app = int(len(z["dan_appetitive_idx"])) if "dan_appetitive_idx" in z.files else 0
        self.n_dan_av = int(len(z["dan_aversive_idx"])) if "dan_aversive_idx" in z.files else 0
        if self.mask_av.shape == self.kc_mbon_w.shape:
            self.novelty_posts = np.unique(self.kc_mbon_post[self.mask_av]).astype(np.int32)
        else:
            self.novelty_posts = np.arange(self.n_mbon, dtype=np.int32)
        z.close()

    @staticmethod
    def _to_local(pre, post, w, map_pre, map_post):
        pre = pre.astype(np.int32)
        post = post.astype(np.int32)
        w = w.astype(np.float32)
        keep = []
        lp, lq, lw = [], [], []
        for i in range(len(pre)):
            a = map_pre.get(int(pre[i]))
            b = map_post.get(int(post[i]))
            if a is None or b is None:
                continue
            lp.append(a)
            lq.append(b)
            lw.append(w[i])
            keep.append(i)
        return (
            np.asarray(lp, dtype=np.int32),
            np.asarray(lq, dtype=np.int32),
            np.asarray(lw, dtype=np.float32),
        )

    def _make_pattern(self, frac: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        p = np.zeros(self.n_pn, dtype=np.float32)
        k = max(1, int(self.n_pn * frac))
        idx = rng.choice(self.n_pn, size=k, replace=False)
        p[idx] = 1.0
        return p

    def _assign_mbon_actions(self) -> np.ndarray:
        """Assign each MBON unit evenly across ACTIONS (stable by type name)."""
        out = np.zeros(self.n_mbon, dtype=np.int32)
        for i, g in enumerate(self.mbon_idx):
            t = str(self.cell_types[int(g)])
            # deterministic round-robin-ish spread so no single action dominates at t0
            out[i] = int(sum(map(ord, t)) + i) % len(ACTIONS)
        return out

    def encode_cues(self, cues: dict[str, float]) -> np.ndarray:
        u = np.zeros(self.n_pn, dtype=np.float32)
        for name, amp in cues.items():
            if amp <= 0:
                continue
            pat = self.patterns.get(name)
            if pat is None:
                continue
            u += float(amp) * pat
        u = np.clip(u, 0.0, 1.0)
        return u

    def readout(
        self, cues: dict[str, float], raw_pn: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deterministic PN → KC → MBON scores. Does not consume the policy RNG."""
        pn = self.encode_cues(cues)
        if raw_pn is not None:
            pn = np.clip(pn + raw_pn, 0.0, 1.0)
        kc = np.zeros(self.n_kc, dtype=np.float32)
        if len(self.pn_kc_w):
            np.add.at(kc, self.pn_kc_post, self.pn_kc_w * pn[self.pn_kc_pre])
        kc = apl_sparsify(kc)

        mbon, scores = self.scores_of_kc(kc)
        scores[ACTIONS.index("Explore")] += 0.15 * (float(scores.max()) + 1.0)
        if float(scores.sum()) < 1e-6:
            scores[ACTIONS.index("Explore")] = 1.0
        return pn, kc, mbon, scores

    def scores_of_kc(self, kc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """MBON rates and action buckets for an existing KC vector. Does not learn."""
        mbon = np.zeros(self.n_mbon, dtype=np.float32)
        if len(self.kc_mbon_w):
            np.add.at(mbon, self.kc_mbon_post, self.kc_mbon_w * kc[self.kc_mbon_pre])
        scores = np.zeros(len(ACTIONS), dtype=np.float32)
        for i, a_i in enumerate(self.mbon_action):
            scores[int(a_i)] += mbon[i]
        return mbon, scores

    def forward(self, cues: dict[str, float], raw_pn: np.ndarray | None = None) -> MBForward:
        pn, kc, mbon, scores = self.readout(cues, raw_pn)
        logits = scores - scores.max()
        probs = np.exp(logits / 0.5)
        probs /= probs.sum()
        action = ACTIONS[int(self.rng.choice(len(ACTIONS), p=probs))]
        return MBForward(pn=pn, kc=kc, mbon=mbon, action=action, action_scores=scores)

    def plasticity(self, last: MBForward, r: float) -> None:
        """Dopamine depresses KC→MBON synapses of KCs that were active.

        On the visual pathway that is γ-d only. α/β-p still drives the
        current MBON sum, and its synapses stay at the birth weight.
        The change stays inside that dopamine neuron's compartment. PAM
        (r>0, appetitive mask) reaches avoidance MBONs, so a reward weakens
        avoidance and the paired pattern approaches. PPL1 (r<0, aversive
        mask) reaches approach MBONs, so punishment weakens approach.
        The opposite synapse is not potentiated. Weights stay inside the
        birth-weight floor and ceiling, so a synapse does not sit at zero.
        """
        if abs(r) < 1e-9 or last.kc.sum() <= 0 or len(self.kc_mbon_w) == 0:
            return
        kc = last.kc
        pre_act = kc[self.kc_mbon_pre]
        active = self._learned_pre(pre_act > 0)
        if not np.any(active):
            return

        approach_ids = {ACTIONS.index("Approach_A"), ACTIONS.index("Approach_B")}
        avoid_ids = {ACTIONS.index("Avoid_A"), ACTIONS.index("Avoid_B")}
        post_action = self.mbon_action[self.kc_mbon_post]

        if r < 0:
            mask = self.mask_av & active
            dec = mask & np.isin(post_action, list(approach_ids))
            self.kc_mbon_w[dec] -= self.eta * abs(r) * pre_act[dec]
        else:
            mask = self.mask_ap & active
            dec = mask & np.isin(post_action, list(avoid_ids))
            self.kc_mbon_w[dec] -= self.eta * r * pre_act[dec]

        self._bound_weights()

    def appetitive_drive(self, scores: np.ndarray) -> float:
        """Approach MBON sum minus avoid MBON sum. PAM raises this by weakening avoidance."""
        ap = ACTIONS.index("Approach_A")
        av = ACTIONS.index("Avoid_A")
        bp = ACTIONS.index("Approach_B")
        bv = ACTIONS.index("Avoid_B")
        return float(scores[ap] + scores[bp] - scores[av] - scores[bv])

    def novelty_drive(self, mbon: np.ndarray) -> float:
        """Mean activity of MBONs that receive aversive-masked KC synapses."""
        if len(self.novelty_posts) == 0:
            return 0.0
        return float(mbon[self.novelty_posts].mean())

    def plasticity_familiarity(self, kc: np.ndarray) -> float:
        """Depress KC→novelty-MBON synapses for KCs that have fired before.

        The first glimpse does not depress: the occupancy trace is read first
        and updated after. Returns novelty in [0, 1] (1 = these KCs are new).
        No class label and no zone reward enter this rule.
        """
        active = kc > 0
        if not np.any(active) or len(self.kc_mbon_w) == 0:
            return 0.0
        fam = self.kc_fam[active]
        novelty = float(np.clip(1.0 - float(fam.mean()), 0.0, 1.0))
        pre = kc[self.kc_mbon_pre]
        post_ok = np.isin(self.kc_mbon_post, self.novelty_posts)
        learned = self._learned_pre(pre > 0)
        mask = self.mask_av & post_ok & learned & (self.kc_fam[self.kc_mbon_pre] > 0.05)
        if np.any(mask):
            self.kc_mbon_w[mask] -= self.eta * self.kc_fam[self.kc_mbon_pre][mask] * pre[mask]
            self._bound_weights()
        self.kc_fam = 0.985 * self.kc_fam + 0.015 * active.astype(np.float32)
        return novelty

    def _learned_pre(self, active: np.ndarray) -> np.ndarray:
        """Visual value is stored on γ-d. α/β-p keeps the weight it was born with.

        A runtime with no pathway mask still learns on every active Kenyon cell.
        """
        gamma = getattr(self, "gamma_d_kc", None)
        if gamma is None:
            return active
        return active & gamma[self.kc_mbon_pre]

    def _bound_weights(self) -> None:
        if len(self.kc_mbon_w) == 0:
            return
        np.clip(self.kc_mbon_w, self.w_floor, self.w_ceil, out=self.kc_mbon_w)

    def recover(self, kc: np.ndarray, dopamine: bool) -> None:
        """Return depressed KC→MBON weights toward the weight they were born with.

        A Kenyon cell that fired on this frame, with no dopamine, also gains a
        little. The floor and the ceiling still hold.
        """
        if len(self.kc_mbon_w) == 0:
            return
        gap = self.kc_mbon_w0 - self.kc_mbon_w
        rising = gap > 0.0
        if np.any(rising):
            self.kc_mbon_w[rising] += np.float32(RECOVERY_RATE) * gap[rising]
        if not dopamine and kc is not None and float(kc.max()) > 0.0:
            pre = kc[self.kc_mbon_pre]
            active = self._learned_pre(pre > 0.0)
            if np.any(active):
                self.kc_mbon_w[active] += np.float32(IDLE_GROWTH) * pre[active]
        self._bound_weights()

    def reset_plastic(self) -> None:
        self.kc_mbon_w = self.kc_mbon_w0.astype(np.float32).copy()
        self.kc_fam[:] = 0.0

    def save_mb(self, path: str | Path, proj_seed: int, dan: str, extra: dict | None = None) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            kc_mbon_w=np.asarray(self.kc_mbon_w, dtype=np.float32),
            kc_fam=np.asarray(self.kc_fam, dtype=np.float32),
            proj_seed=np.int32(proj_seed),
            dan=np.asarray(str(dan), dtype="<U32"),
        )
        if extra:
            for key, value in extra.items():
                payload[key] = as_plain(value)
        # Atomic: write a sibling temp file, then os.replace. A crash or a
        # restart mid-save leaves the previous file intact. np.savez appended
        # ".npz" to a bare name, so the final name keeps that rule.
        import os

        final = dest if dest.name.endswith(".npz") else dest.with_name(dest.name + ".npz")
        tmp = final.with_name(final.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, final)

    def load_mb(self, path: str | Path) -> tuple[int, str]:
        z = open_npz(path)
        w = np.asarray(z["kc_mbon_w"], dtype=np.float32)
        if w.shape != self.kc_mbon_w.shape:
            raise ValueError(
                "форма KC→MBON %s не совпала с мозгом %s" % (w.shape, self.kc_mbon_w.shape)
            )
        self.kc_mbon_w = w
        self._bound_weights()
        fam = np.asarray(z["kc_fam"], dtype=np.float32) if "kc_fam" in z.files else None
        if fam is not None and fam.shape == self.kc_fam.shape:
            self.kc_fam = fam
        seed = int(z["proj_seed"]) if "proj_seed" in z.files else 0
        dan = scalar_str(z["dan"]) if "dan" in z.files else "teacher"
        z.close()
        return seed, dan

    def weight_drift(self) -> float:
        return float(np.linalg.norm(self.kc_mbon_w - self.kc_mbon_w0))
