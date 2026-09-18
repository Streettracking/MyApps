"""Headless + pygame orchestration for multi-agent FlyWire MB arena sim."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from sim.mb_runtime import MushroomBodyRuntime
from sim.world import World


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "artifacts" / "connectome_mb_v1.npz"


class Simulator:
    def __init__(
        self,
        n_agents: int = 3,
        seed: int = 0,
        blind_peers: bool = False,
        move_zones: bool = False,
        npz_path: Path = DEFAULT_NPZ,
        eta: float = 0.02,
    ):
        self.world = World.default(n_agents=n_agents, seed=seed, blind_peers=blind_peers)
        self.world.cfg.move_zones = move_zones
        self.brains: Dict[str, MushroomBodyRuntime] = {}
        for i, ag in enumerate(self.world.agents):
            self.brains[ag.agent_id] = MushroomBodyRuntime(npz_path, eta=eta, seed=seed + 17 * i)
        self.history: List[dict] = []

    def step(self) -> None:
        self.world.maybe_move_zones()
        # Perceive + decide independently (no inter-agent learning bus)
        decisions = {}
        for ag in self.world.agents:
            r = self.world.local_reward(ag)
            cues = self.world.perceive(ag)
            brain = self.brains[ag.agent_id]
            result = brain.step(cues, r)
            ag.action = result.action
            ag.r = r
            ag.plastic_updates = result.plastic_updates
            ag.weight_drift = brain.weight_drift()
            decisions[ag.agent_id] = result
            self.world.tick_timers(ag)
        for ag in self.world.agents:
            self.world.apply_action(ag, ag.action)
        self.world.time_s += self.world.cfg.dt
        row = {
            "t": round(self.world.time_s, 3),
            "agents": [
                {
                    "id": ag.agent_id,
                    "x": round(ag.x, 3),
                    "y": round(ag.y, 3),
                    "action": ag.action,
                    "r": ag.r,
                    "tA": round(ag.t_in_a, 2),
                    "tB": round(ag.t_in_b, 2),
                    "drift": round(ag.weight_drift, 4),
                }
                for ag in self.world.agents
            ],
            "zones": {
                z.name: {
                    "centroid": [round(z.centroid()[0], 3), round(z.centroid()[1], 3)],
                    "reward": z.reward,
                }
                for z in self.world.zones
            },
        }
        self.history.append(row)

    def metrics(self) -> dict:
        out = {}
        for ag in self.world.agents:
            denom = ag.t_in_a + ag.t_in_b + 1e-6
            pi = (ag.t_in_b - ag.t_in_a) / denom
            out[ag.agent_id] = {
                "PI": round(pi, 4),
                "t_in_A": round(ag.t_in_a, 2),
                "t_in_B": round(ag.t_in_b, 2),
                "weight_drift": round(ag.weight_drift, 4),
                "plastic_updates": ag.plastic_updates,
                "sense_conspecifics": ag.sense_conspecifics,
            }
        return out

    def run_headless(self, seconds: float, log_csv: Path | None = None) -> dict:
        steps = int(seconds / self.world.cfg.dt)
        for _ in range(steps):
            self.step()
        metrics = self.metrics()
        if log_csv:
            log_csv.parent.mkdir(parents=True, exist_ok=True)
            with log_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t", "agent", "x", "y", "action", "r", "tA", "tB", "drift"])
                for row in self.history:
                    for ag in row["agents"]:
                        w.writerow(
                            [
                                row["t"],
                                ag["id"],
                                ag["x"],
                                ag["y"],
                                ag["action"],
                                ag["r"],
                                ag["tA"],
                                ag["tB"],
                                ag["drift"],
                            ]
                        )
        return metrics


def run_pygame(sim: Simulator, seconds: float | None = None) -> dict:
    import pygame

    pygame.init()
    scale = 120  # px per meter
    w_px = int(sim.world.cfg.width * scale)
    h_px = int(sim.world.cfg.height * scale)
    screen = pygame.display.set_mode((w_px + 280, h_px))
    pygame.display.set_caption("FlyWire MB × Go2 Arena Simulator")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()
    running = True
    t0 = time.time()
    paused = False

    def to_px(x, y):
        return int(x * scale), int(h_px - y * scale)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_m:
                    sim.world.cfg.move_zones = not sim.world.cfg.move_zones
                elif event.key == pygame.K_p:
                    for z in sim.world.zones:
                        z.projected = not z.projected

        if not paused:
            sim.step()

        screen.fill((24, 26, 32))
        # floor
        pygame.draw.rect(screen, (36, 40, 48), pygame.Rect(0, 0, w_px, h_px))
        # zones
        for z in sim.world.zones:
            pts = [to_px(x, y) for x, y in z.polygon]
            color = (180, 60, 60) if z.name == "A" else (60, 160, 90)
            if z.projected:
                pygame.draw.polygon(screen, color, pts)
                pygame.draw.polygon(screen, (255, 255, 255), pts, 2)
            else:
                pygame.draw.polygon(screen, (70, 70, 80), pts, 2)
            cx, cy = to_px(*z.centroid())
            label = font.render(f"{z.name} r={z.reward:+.0f}", True, (240, 240, 240))
            screen.blit(label, (cx - 30, cy - 8))

        # agents
        for ag in sim.world.agents:
            px, py = to_px(ag.x, ag.y)
            pygame.draw.circle(screen, ag.color, (px, py), 12)
            hx = px + int(18 * np.cos(ag.yaw))
            hy = py - int(18 * np.sin(ag.yaw))
            pygame.draw.line(screen, (255, 255, 255), (px, py), (hx, hy), 2)
            screen.blit(font.render(ag.agent_id, True, (220, 220, 220)), (px + 14, py - 18))
            screen.blit(font.render(ag.action, True, (200, 200, 120)), (px + 14, py - 2))

        # sidebar
        panel_x = w_px + 10
        screen.blit(font.render("FlyWire MB arena", True, (255, 255, 255)), (panel_x, 10))
        screen.blit(font.render(f"t={sim.world.time_s:.1f}s", True, (200, 200, 200)), (panel_x, 32))
        screen.blit(
            font.render(
                f"zones={'MOVE' if sim.world.cfg.move_zones else 'STATIC'}  "
                f"proj={'ON' if sim.world.zones[0].projected else 'OFF'}",
                True,
                (180, 220, 180),
            ),
            (panel_x, 54),
        )
        screen.blit(font.render("SPACE pause  M move  P project  ESC quit", True, (150, 150, 160)), (panel_x, 76))
        y = 110
        for ag in sim.world.agents:
            denom = ag.t_in_a + ag.t_in_b + 1e-6
            pi = (ag.t_in_b - ag.t_in_a) / denom
            lines = [
                f"{ag.agent_id}  PI={pi:+.2f}",
                f"  A={ag.t_in_a:.1f}s B={ag.t_in_b:.1f}s",
                f"  act={ag.action} r={ag.r:+.0f}",
                f"  drift={ag.weight_drift:.3f}",
            ]
            for line in lines:
                screen.blit(font.render(line, True, ag.color), (panel_x, y))
                y += 18
            y += 8

        pygame.display.flip()
        clock.tick(int(1 / sim.world.cfg.dt))
        if seconds is not None and (time.time() - t0) >= seconds:
            running = False

    pygame.quit()
    return sim.metrics()


def main() -> None:
    p = argparse.ArgumentParser(description="FlyWire MB × Go2 arena simulator (Windows OK)")
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blind-peers", action="store_true")
    p.add_argument("--move-zones", action="store_true", help="Projected zones drift over time")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--metrics-out", type=Path, default=None)
    args = p.parse_args()

    sim = Simulator(
        n_agents=args.agents,
        seed=args.seed,
        blind_peers=args.blind_peers,
        move_zones=args.move_zones,
        npz_path=args.npz,
    )

    if args.headless:
        metrics = sim.run_headless(args.seconds, log_csv=args.log)
    else:
        metrics = run_pygame(sim, seconds=args.seconds if args.seconds > 0 else None)

    print(json.dumps(metrics, indent=2))
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
