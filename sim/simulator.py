"""Multi-agent episode loop: individuum MB + external conditions only."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mb_runtime import ACTIONS, MushroomBodyRuntime


def _learned_shift(init: dict, final: dict) -> dict:
    """How much the dog-driven score moved, minus the same for a distractor."""

    def drive(pack: dict, name: str) -> np.ndarray:
        return np.asarray(pack[name], dtype=float) - np.asarray(pack["empty"], dtype=float)

    learned_dog = drive(final, "dog") - drive(init, "dog")
    learned_dist = drive(final, "distractor") - drive(init, "distractor")
    diff = learned_dog - learned_dist
    return {
        "learned_dog_l2": float(np.linalg.norm(learned_dog)),
        "learned_dist_l2": float(np.linalg.norm(learned_dist)),
        "diff_l2": float(np.linalg.norm(diff)),
        "diff_by_action": {ACTIONS[i]: float(diff[i]) for i in range(len(ACTIONS))},
    }
from .raw_sense import RawProjector, probe_features, render_view
from .world import AgentState, ArenaConfig, ArenaWorld, Zone, default_agents, default_zones


@dataclass
class SimConfig:
    npz_path: Path
    duration_s: float = 120.0
    n_agents: int = 3
    sense_conspecifics: bool = True
    move_zones: bool = False
    eta: float = 0.05
    seed: int = 42
    explore_eps: float = 0.25  # random Explore overrides early on
    percept: str = "fixed"  # fixed labeled peer cues | raw camera+lidar
    log_dir: Path | None = None


class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        arena_cfg = ArenaConfig(move_zones=cfg.move_zones)
        self.world = ArenaWorld(arena_cfg, default_zones(), default_agents(cfg.n_agents, cfg.sense_conspecifics))
        if cfg.percept not in ("fixed", "raw"):
            raise ValueError(f"unknown percept {cfg.percept}")
        if cfg.percept == "raw":
            self.world.spawn_distractors(cfg.seed)
        self.brains: dict[str, MushroomBodyRuntime] = {}
        self.projectors: dict[str, RawProjector] = {}
        for i, ag in enumerate(self.world.agents):
            brain = MushroomBodyRuntime(cfg.npz_path, eta=cfg.eta, seed=cfg.seed + i + 1)
            self.brains[ag.agent_id] = brain
            if cfg.percept == "raw":
                # Per individuum: its own random glomerulus map, not a shared label.
                self.projectors[ag.agent_id] = RawProjector(brain.n_pn, seed=cfg.seed + 100 + i)
        self.history: list[dict] = []
        self.step_i = 0
        self._ctx = {
            ag.agent_id: {"dog_only": 0, "dist_only": 0, "both": 0, "neither": 0,
                          "dog_only_approach": 0, "dist_only_approach": 0,
                          "r_and_dog": 0, "r_and_dist": 0, "r_steps": 0}
            for ag in self.world.agents
        }
        self._probe_init = self.probe_readouts()

    def step(self) -> dict:
        # 1) external conditions (moving projection etc.)
        self.world.step_environment()

        # 2) each individuum: sense locally -> MB -> action -> move -> local r/plasticity
        snap = {"t": self.world.t, "agents": {}}
        for ag in self.world.agents:
            brain = self.brains[ag.agent_id]
            cues = self.world.cues_for(ag)
            raw_feat = None
            raw_pn = None
            if self.cfg.percept == "raw":
                # Labeled peer channels are not used. Blind peers omit other dogs
                # from the raw view; distractors stay so the control still sees motion.
                cues = {"A": cues["A"], "B": cues["B"], "peer": 0.0, "peer_at_B": 0.0}
                raw_feat, hit = render_view(ag, self.world, include_agents=ag.sense_conspecifics)
                raw_pn = self.projectors[ag.agent_id].project(raw_feat)
                brain.last_raw = raw_feat
            else:
                hit = None
                brain.last_raw = None
            fwd = brain.forward(cues, raw_pn)
            brain.last_forward = fwd
            brain.last_cues = cues
            ag.action = fwd.action
            # decaying random explore so agents sample zones before policy hardens
            eps = self.cfg.explore_eps * max(0.0, 1.0 - self.world.t / max(self.cfg.duration_s, 1.0))
            if self.rng.random() < eps:
                ag.action = "Explore"
            self.world.apply_action(ag, ag.action)
            r = self.world.reward_at(ag.x, ag.y)
            ag.r = r
            if self.world.zones["A"].contains(ag.x, ag.y):
                ag.time_in_A += self.world.cfg.dt
            if self.world.zones["B"].contains(ag.x, ag.y):
                ag.time_in_B += self.world.cfg.dt
            if abs(r) > 0:
                brain.plasticity(fwd, r)
                ag.plastic_updates += 1
            self._tally_context(ag, cues, hit, r, fwd.action)
            snap["agents"][ag.agent_id] = {
                "x": ag.x,
                "y": ag.y,
                "action": ag.action,
                "r": ag.r,
                "cues": cues,
                "pi": (ag.time_in_B - ag.time_in_A) / (ag.time_in_A + ag.time_in_B + 1e-6),
                "w_drift": brain.weight_drift(),
            }
        self.history.append(snap)
        self.step_i += 1
        return snap

    def _tally_context(self, ag, cues, hit, r: float, action: str) -> None:
        box = self._ctx[ag.agent_id]
        if self.cfg.percept == "raw" and hit is not None:
            dog, dist = hit.dog, hit.distractor
        else:
            dog = ag.sense_conspecifics and cues.get("peer", 0.0) > 0.2
            dist = False
        if abs(r) > 0:
            box["r_steps"] += 1
            if dog:
                box["r_and_dog"] += 1
            if dist:
                box["r_and_dist"] += 1
            return
        if dog and not dist:
            key = "dog_only"
        elif dist and not dog:
            key = "dist_only"
        elif dog and dist:
            key = "both"
        else:
            key = "neither"
        box[key] += 1
        if action.startswith("Approach") and key in ("dog_only", "dist_only"):
            box[key + "_approach"] += 1

    def probe_readouts(self) -> dict:
        """Action scores for canonical dog vs distractor features, zone odor off.

        Fixed-detector brains get the labeled peer cue instead of raw pixels.
        Raw brains get the same unlabeled probe vectors through their own projector.
        """
        out = {}
        for ag in self.world.agents:
            brain = self.brains[ag.agent_id]
            if self.cfg.percept == "raw":
                proj = self.projectors[ag.agent_id]
                packs = {
                    name: proj.project(probe_features(name))
                    for name in ("empty", "dog", "distractor")
                }
                cues = {"A": 0.0, "B": 0.0}
                scored = {
                    name: brain.readout(cues, pn)[3].astype(float).tolist()
                    for name, pn in packs.items()
                }
            else:
                scored = {}
                for name, cues in (
                    ("empty", {"A": 0.0, "B": 0.0}),
                    ("dog", {"A": 0.0, "B": 0.0, "peer": 1.0}),
                    ("distractor", {"A": 0.0, "B": 0.0}),
                ):
                    scored[name] = brain.readout(cues)[3].astype(float).tolist()
            out[ag.agent_id] = scored
        return out

    def run(self, steps: int | None = None, on_step=None) -> dict:
        if steps is None:
            steps = int(self.cfg.duration_s / self.world.cfg.dt)
        for _ in range(steps):
            snap = self.step()
            if on_step is not None:
                on_step(self, snap)
        return self.summary()

    def summary(self) -> dict:
        agents = {}
        for ag in self.world.agents:
            brain = self.brains[ag.agent_id]
            agents[ag.agent_id] = {
                "time_in_A": ag.time_in_A,
                "time_in_B": ag.time_in_B,
                "PI": (ag.time_in_B - ag.time_in_A) / (ag.time_in_A + ag.time_in_B + 1e-6),
                "plastic_updates": ag.plastic_updates,
                "weight_drift": brain.weight_drift(),
                "sense_conspecifics": ag.sense_conspecifics,
                "context": self._ctx[ag.agent_id],
                "probe_init": self._probe_init[ag.agent_id],
                "probe_final": None,
            }
        probe_final = self.probe_readouts()
        shifts = []
        for ag in self.world.agents:
            agents[ag.agent_id]["probe_final"] = probe_final[ag.agent_id]
            shifts.append(_learned_shift(self._probe_init[ag.agent_id], probe_final[ag.agent_id]))
            agents[ag.agent_id]["learned_shift"] = shifts[-1]
        out = {
            "duration_s": self.world.t,
            "move_zones": self.cfg.move_zones,
            "sense_conspecifics": self.cfg.sense_conspecifics,
            "percept": self.cfg.percept,
            "n_agents": self.cfg.n_agents,
            "agents": agents,
            "mean_PI": float(np.mean([a["PI"] for a in agents.values()])),
            "mean_shift_l2_dog_minus_dist": float(np.mean([s["diff_l2"] for s in shifts])),
        }
        if self.cfg.log_dir:
            self._write_logs(out)
        return out

    def _write_logs(self, summary: dict) -> None:
        d = Path(self.cfg.log_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (d / "telemetry.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "agent_id", "x", "y", "action", "r", "pi", "w_drift"])
            for snap in self.history:
                for aid, a in snap["agents"].items():
                    w.writerow(
                        [
                            f"{snap['t']:.2f}",
                            aid,
                            f"{a['x']:.4f}",
                            f"{a['y']:.4f}",
                            a["action"],
                            a["r"],
                            f"{a['pi']:.4f}",
                            f"{a['w_drift']:.4f}",
                        ]
                    )
