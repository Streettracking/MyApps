"""Multi-agent episode loop: individuum MB + external conditions only."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mb_runtime import MushroomBodyRuntime
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
    log_dir: Path | None = None


class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        arena_cfg = ArenaConfig(move_zones=cfg.move_zones)
        self.world = ArenaWorld(arena_cfg, default_zones(), default_agents(cfg.n_agents, cfg.sense_conspecifics))
        self.brains: dict[str, MushroomBodyRuntime] = {}
        for i, ag in enumerate(self.world.agents):
            self.brains[ag.agent_id] = MushroomBodyRuntime(
                cfg.npz_path, eta=cfg.eta, seed=cfg.seed + i + 1
            )
        self.history: list[dict] = []
        self.step_i = 0

    def step(self) -> dict:
        # 1) external conditions (moving projection etc.)
        self.world.step_environment()

        # 2) each individuum: sense locally -> MB -> action -> move -> local r/plasticity
        snap = {"t": self.world.t, "agents": {}}
        for ag in self.world.agents:
            brain = self.brains[ag.agent_id]
            cues = self.world.cues_for(ag)
            fwd = brain.forward(cues)
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
            }
        out = {
            "duration_s": self.world.t,
            "move_zones": self.cfg.move_zones,
            "sense_conspecifics": self.cfg.sense_conspecifics,
            "n_agents": self.cfg.n_agents,
            "agents": agents,
            "mean_PI": float(np.mean([a["PI"] for a in agents.values()])),
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
