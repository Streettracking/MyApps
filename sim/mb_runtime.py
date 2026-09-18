"""FlyWire MB runtime for the Go2 arena simulator (Windows-friendly, no ROS)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ACTIONS = ("Approach_A", "Avoid_A", "Approach_B", "Avoid_B", "Explore", "Freeze")


@dataclass
class StepResult:
    action: str
    action_idx: int
    mbon_scores: np.ndarray
    kc_active_frac: float
    r: float
    dan_channel: str
    plastic_updates: int


class MushroomBodyRuntime:
    """Rate-based PN → KC → MBON network from connectome_mb_v1.npz."""

    def __init__(self, npz_path: Path | str, eta: float = 0.05, seed: int = 0):
        self.eta = float(eta)
        self.rng = np.random.default_rng(seed)
        data = np.load(npz_path, allow_pickle=True)

        self.root_ids = data["root_ids"].astype(np.int64)
        self.roles = data["roles"]
        self.top_nt = data["top_nt"]

        self.pn_idx = data["pn_idx"].astype(np.int32)
        self.kc_idx = data["kc_idx"].astype(np.int32)
        self.mbon_idx = data["mbon_idx"].astype(np.int32)

        self.pn_kc_pre = data["pn_kc_pre"].astype(np.int32)
        self.pn_kc_post = data["pn_kc_post"].astype(np.int32)
        self.pn_kc_w = data["pn_kc_w"].astype(np.float32).copy()

        self.kc_mbon_pre = data["kc_mbon_pre"].astype(np.int32)
        self.kc_mbon_post = data["kc_mbon_post"].astype(np.int32)
        self.kc_mbon_w = data["kc_mbon_w"].astype(np.float32).copy()
        self.mask_av = data["kc_mbon_mask_aversive"].astype(bool)
        self.mask_ap = data["kc_mbon_mask_appetitive"].astype(bool)

        self.n_pn = len(self.pn_idx)
        self.n_kc = len(self.kc_idx)
        self.n_mbon = len(self.mbon_idx)

        self._pn_local = {int(g): i for i, g in enumerate(self.pn_idx)}
        self._kc_local = {int(g): i for i, g in enumerate(self.kc_idx)}
        self._mbon_local = {int(g): i for i, g in enumerate(self.mbon_idx)}

        self.e_pn = np.array([self._pn_local[int(p)] for p in self.pn_kc_pre], dtype=np.int32)
        self.e_kc_from_pn = np.array([self._kc_local[int(p)] for p in self.pn_kc_post], dtype=np.int32)
        self.e_kc_to_mbon = np.array([self._kc_local[int(p)] for p in self.kc_mbon_pre], dtype=np.int32)
        self.e_mbon = np.array([self._mbon_local[int(p)] for p in self.kc_mbon_post], dtype=np.int32)

        self.pn_kc_w /= max(float(self.pn_kc_w.mean()), 1e-6)
        self.kc_mbon_w /= max(float(self.kc_mbon_w.mean()), 1e-6)
        self.kc_mbon_w0 = self.kc_mbon_w.copy()

        self.pn_pools = self._make_pn_pools()
        self.mbon_action_map = self._make_mbon_action_map()
        self.plastic_updates = 0
        self.last_kc = np.zeros(self.n_kc, dtype=np.float32)
        self._recent_aversive = 0.0
        self._recent_appetitive = 0.0

    def _make_pn_pools(self) -> Dict[str, np.ndarray]:
        order = np.arange(self.n_pn)
        self.rng.shuffle(order)
        names = ["cue_A", "cue_B", "peer_near", "peer_at_B", "threat", "explore"]
        cuts = np.linspace(0, self.n_pn, len(names) + 1, dtype=int)
        return {name: order[cuts[i] : cuts[i + 1]] for i, name in enumerate(names)}

    def _make_mbon_action_map(self) -> np.ndarray:
        """Assign MBONs round-robin to actions for balanced readout."""
        amap = np.zeros(self.n_mbon, dtype=np.int32)
        for i in range(self.n_mbon):
            amap[i] = i % len(ACTIONS)
        return amap

    def encode_cues(self, cues: Dict[str, float]) -> np.ndarray:
        u = np.zeros(self.n_pn, dtype=np.float32)
        for name, pool in self.pn_pools.items():
            val = float(cues.get(name, 0.0))
            if val <= 0 or len(pool) == 0:
                continue
            u[pool] = np.clip(val, 0.0, 1.0)
        u += 0.02
        return np.clip(u, 0.0, 1.0)

    def forward(self, u_pn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kc = np.zeros(self.n_kc, dtype=np.float32)
        np.add.at(kc, self.e_kc_from_pn, self.pn_kc_w * u_pn[self.e_pn])
        k = max(1, int(0.05 * self.n_kc))
        if k < self.n_kc:
            thresh = np.partition(kc, -k)[-k]
            kc = np.where(kc >= thresh, kc, 0.0)
        kc = np.tanh(kc)

        mbon = np.zeros(self.n_mbon, dtype=np.float32)
        np.add.at(mbon, self.e_mbon, self.kc_mbon_w * kc[self.e_kc_to_mbon])
        mbon = np.tanh(mbon)
        self.last_kc = kc
        return kc, mbon

    def choose_action(self, mbon: np.ndarray, cues: Dict[str, float]) -> Tuple[str, int, np.ndarray]:
        raw = np.zeros(len(ACTIONS), dtype=np.float32)
        for i, a_i in enumerate(self.mbon_action_map):
            raw[a_i] += float(mbon[i])
        # Connectome readout is a bias, not a dictator (keeps FlyWire wiring in the loop)
        scores = 0.2 * (raw - raw.mean())

        cue_a = float(cues.get("cue_A", 0.0))
        cue_b = float(cues.get("cue_B", 0.0))
        scores[1] += 2.0 * self._recent_aversive * max(cue_a, 0.15)  # Avoid_A after punishment
        scores[0] -= 2.0 * self._recent_aversive
        scores[2] += 2.0 * self._recent_appetitive * max(cue_b, 0.15)  # Approach_B after reward
        scores[2] += 1.0 * cue_b
        scores[4] += 1.4  # Explore — discover zones
        if cue_a < 0.05 and cue_b < 0.05:
            scores[4] += 1.0
        if float(cues.get("threat", 0.0)) > 0.5:
            scores[5] += 1.2
        # mild suppression of Avoid_B unless cue_B & aversive (rare)
        scores[3] -= 0.4

        logits = scores / 0.3
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = int(self.rng.choice(len(ACTIONS), p=probs))
        return ACTIONS[idx], idx, scores

    def plasticity(self, r: float, kc: np.ndarray) -> str:
        self._recent_aversive *= 0.985
        self._recent_appetitive *= 0.985
        if abs(r) < 1e-6:
            return "none"
        channel = "aversive" if r < 0 else "appetitive"
        if channel == "aversive":
            self._recent_aversive = min(3.0, self._recent_aversive + 1.0)
        else:
            self._recent_appetitive = min(3.0, self._recent_appetitive + 1.0)

        mask = self.mask_av if channel == "aversive" else self.mask_ap
        if mask.size == 0 or not np.any(mask):
            mask = np.ones(len(self.kc_mbon_w), dtype=bool)
        active = mask & (kc[self.e_kc_to_mbon] > 0)
        if not np.any(active):
            return channel

        e_idx = np.where(active)[0]
        kc_vals = kc[self.e_kc_to_mbon[e_idx]]
        actions = self.mbon_action_map[self.e_mbon[e_idx]]
        is_avoid = np.isin(actions, [1, 3, 5])
        is_approach = np.isin(actions, [0, 2])
        dw = self.eta * float(abs(r)) * kc_vals
        if channel == "aversive":
            self.kc_mbon_w[e_idx[is_avoid]] += dw[is_avoid]
            self.kc_mbon_w[e_idx[is_approach]] -= dw[is_approach]
        else:
            self.kc_mbon_w[e_idx[is_approach]] += dw[is_approach]
            self.kc_mbon_w[e_idx[is_avoid]] -= dw[is_avoid]
        np.clip(self.kc_mbon_w, 0.01, 10.0, out=self.kc_mbon_w)
        self.plastic_updates += 1
        return channel

    def step(self, cues: Dict[str, float], r: float) -> StepResult:
        u = self.encode_cues(cues)
        kc, mbon = self.forward(u)
        action, aidx, scores = self.choose_action(mbon, cues)
        dan = self.plasticity(r, kc)
        return StepResult(
            action=action,
            action_idx=aidx,
            mbon_scores=scores,
            kc_active_frac=float((kc > 0).mean()),
            r=r,
            dan_channel=dan,
            plastic_updates=self.plastic_updates,
        )

    def weight_drift(self) -> float:
        return float(np.linalg.norm(self.kc_mbon_w - self.kc_mbon_w0))
