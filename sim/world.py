"""2D arena: zones (static or moving projection), peer sensing, simple holonomic agents."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    # ray casting
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


@dataclass
class Zone:
    name: str
    reward: float
    polygon: np.ndarray  # (N,2)
    color: tuple[int, int, int] = (200, 80, 80)

    def contains(self, x: float, y: float) -> bool:
        return point_in_poly(x, y, self.polygon)

    def centroid(self) -> tuple[float, float]:
        c = self.polygon.mean(axis=0)
        return float(c[0]), float(c[1])

    def translate(self, dx: float, dy: float) -> None:
        self.polygon = self.polygon + np.array([dx, dy], dtype=np.float64)


@dataclass
class ArenaConfig:
    width: float = 6.0
    height: float = 5.0
    max_speed: float = 0.6
    peer_near_m: float = 1.2
    peer_see_m: float = 2.5
    dt: float = 0.1
    move_zones: bool = False
    zone_speed: float = 0.15  # m/s drift for projection mode


@dataclass
class AgentState:
    agent_id: str
    x: float
    y: float
    yaw: float = 0.0
    sense_conspecifics: bool = True
    color: tuple[int, int, int] = (80, 160, 255)
    action: str = "Explore"
    r: float = 0.0
    time_in_A: float = 0.0
    time_in_B: float = 0.0
    plastic_updates: int = 0
    trail: list[tuple[float, float]] = field(default_factory=list)


class ArenaWorld:
    def __init__(self, cfg: ArenaConfig, zones: list[Zone], agents: list[AgentState]):
        self.cfg = cfg
        self.zones = {z.name: z for z in zones}
        self.agents = agents
        self.t = 0.0
        self._zone_vel = {
            "A": np.array([cfg.zone_speed, 0.1 * cfg.zone_speed]),
            "B": np.array([-0.08 * cfg.zone_speed / 0.15 * 0.15, cfg.zone_speed * 0.7]),
        }

    def reward_at(self, x: float, y: float) -> float:
        r = 0.0
        for z in self.zones.values():
            if z.contains(x, y):
                r += z.reward
        return float(np.clip(r, -1.0, 1.0))

    def cues_for(self, ag: AgentState) -> dict[str, float]:
        cues = {"A": 0.0, "B": 0.0, "peer": 0.0, "peer_at_B": 0.0}
        # "Projection" visibility: stronger near zone centroid / inside
        for name in ("A", "B"):
            z = self.zones[name]
            cx, cy = z.centroid()
            d = float(np.hypot(ag.x - cx, ag.y - cy))
            if z.contains(ag.x, ag.y):
                cues[name] = 1.0
            else:
                cues[name] = float(np.clip(1.0 - d / 2.0, 0.0, 1.0) * 0.6)

        if ag.sense_conspecifics:
            for other in self.agents:
                if other.agent_id == ag.agent_id:
                    continue
                d = float(np.hypot(ag.x - other.x, ag.y - other.y))
                if d <= self.cfg.peer_see_m:
                    cues["peer"] = max(cues["peer"], float(np.clip(1.0 - d / self.cfg.peer_see_m, 0, 1)))
                if d <= self.cfg.peer_see_m and self.zones["B"].contains(other.x, other.y):
                    cues["peer_at_B"] = max(cues["peer_at_B"], 0.8)
        return cues

    def apply_action(self, ag: AgentState, action: str) -> None:
        cfg = self.cfg
        vx = vy = 0.0
        ax, ay = self.zones["A"].centroid()
        bx, by = self.zones["B"].centroid()

        def toward(tx, ty, speed):
            dx, dy = tx - ag.x, ty - ag.y
            n = np.hypot(dx, dy) + 1e-9
            return speed * dx / n, speed * dy / n

        def away(tx, ty, speed):
            dx, dy = ag.x - tx, ag.y - ty
            n = np.hypot(dx, dy) + 1e-9
            return speed * dx / n, speed * dy / n

        spd = cfg.max_speed
        if action == "Approach_A":
            vx, vy = toward(ax, ay, spd)
        elif action == "Avoid_A":
            vx, vy = away(ax, ay, spd)
        elif action == "Approach_B":
            vx, vy = toward(bx, by, spd)
        elif action == "Avoid_B":
            vx, vy = away(bx, by, spd)
        elif action == "Freeze":
            vx = vy = 0.0
        else:  # Explore: persist yaw with noise
            ag.yaw += float(np.random.uniform(-0.4, 0.4))
            vx = spd * 0.7 * np.cos(ag.yaw)
            vy = spd * 0.7 * np.sin(ag.yaw)

        # soft peer repulsion (body, not social learning)
        for other in self.agents:
            if other.agent_id == ag.agent_id:
                continue
            dx, dy = ag.x - other.x, ag.y - other.y
            d = float(np.hypot(dx, dy) + 1e-9)
            if d < 0.8:
                vx += 0.5 * dx / d
                vy += 0.5 * dy / d

        # integrate
        ag.x = float(np.clip(ag.x + vx * cfg.dt, 0.15, cfg.width - 0.15))
        ag.y = float(np.clip(ag.y + vy * cfg.dt, 0.15, cfg.height - 0.15))
        ag.trail.append((ag.x, ag.y))
        if len(ag.trail) > 400:
            ag.trail = ag.trail[-400:]

    def step_environment(self) -> None:
        """External conditions only: optional moving projected zones."""
        self.t += self.cfg.dt
        if not self.cfg.move_zones:
            return
        for name, vel in self._zone_vel.items():
            z = self.zones[name]
            z.translate(float(vel[0] * self.cfg.dt), float(vel[1] * self.cfg.dt))
            # bounce inside arena with margin
            xs = z.polygon[:, 0]
            ys = z.polygon[:, 1]
            if xs.min() < 0.2 or xs.max() > self.cfg.width - 0.2:
                vel[0] *= -1
                z.translate(float(vel[0] * self.cfg.dt), 0.0)
            if ys.min() < 0.2 or ys.max() > self.cfg.height - 0.2:
                vel[1] *= -1
                z.translate(0.0, float(vel[1] * self.cfg.dt))


def default_zones() -> list[Zone]:
    a = np.array([[1.0, 1.0], [2.2, 1.0], [2.2, 2.2], [1.0, 2.2]], dtype=np.float64)
    b = np.array([[4.0, 3.0], [5.2, 3.0], [5.2, 4.2], [4.0, 4.2]], dtype=np.float64)
    return [
        Zone("A", -1.0, a, color=(220, 70, 70)),
        Zone("B", 1.0, b, color=(70, 200, 110)),
    ]


def default_agents(n: int = 3, sense: bool = True) -> list[AgentState]:
    # Start across the room so agents encounter both projected zones
    starts = [(1.5, 1.6), (3.0, 2.5), (4.5, 3.5), (2.5, 4.0), (5.0, 1.5)]
    colors = [(80, 160, 255), (255, 180, 60), (200, 120, 255), (100, 220, 220), (240, 140, 140)]
    agents = []
    for i in range(n):
        x, y = starts[i % len(starts)]
        agents.append(
            AgentState(
                agent_id=f"go2_{i+1}",
                x=x,
                y=y,
                yaw=float(i),
                sense_conspecifics=sense,
                color=colors[i % len(colors)],
            )
        )
    return agents
