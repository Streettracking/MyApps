"""2D arena world: zones (static/moving/projected), agents, local reward."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def point_in_poly(x: float, y: float, poly: Sequence[Point]) -> bool:
    # Ray casting
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def poly_centroid(poly: Sequence[Point]) -> Point:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def translate_poly(poly: Sequence[Point], dx: float, dy: float) -> List[Point]:
    return [(x + dx, y + dy) for x, y in poly]


@dataclass
class Zone:
    name: str
    reward: float
    polygon: List[Point]
    projected: bool = True  # visual cue available when projected/marked

    def contains(self, x: float, y: float) -> bool:
        return point_in_poly(x, y, self.polygon)

    def centroid(self) -> Point:
        return poly_centroid(self.polygon)


@dataclass
class AgentState:
    agent_id: str
    x: float
    y: float
    yaw: float
    sense_conspecifics: bool = True
    speed: float = 0.55
    color: Tuple[int, int, int] = (80, 180, 255)
    action: str = "Explore"
    r: float = 0.0
    t_in_a: float = 0.0
    t_in_b: float = 0.0
    plastic_updates: int = 0
    weight_drift: float = 0.0
    freeze_timer: float = 0.0


@dataclass
class WorldConfig:
    width: float = 6.0
    height: float = 5.0
    dt: float = 0.1
    peer_near_m: float = 1.2
    vision_range_m: float = 2.5
    vision_fov_deg: float = 120.0
    move_zones: bool = False
    zone_speed: float = 0.15  # m/s drift for projected zones


@dataclass
class World:
    cfg: WorldConfig
    zones: List[Zone]
    agents: List[AgentState]
    time_s: float = 0.0
    _zone_vel: Dict[str, Point] = field(default_factory=dict)

    @staticmethod
    def default(n_agents: int = 3, seed: int = 0, blind_peers: bool = False) -> "World":
        rng = np.random.default_rng(seed)
        zones = [
            Zone("A", -1.0, [(1.0, 1.0), (2.2, 1.0), (2.2, 2.2), (1.0, 2.2)], True),
            Zone("B", 1.0, [(4.0, 3.0), (5.2, 3.0), (5.2, 4.2), (4.0, 4.2)], True),
        ]
        colors = [(80, 180, 255), (255, 160, 80), (160, 255, 120), (220, 120, 220)]
        agents = []
        starts = [(0.8, 1.5), (0.8, 2.5), (0.8, 3.5), (1.5, 0.8)]
        for i in range(n_agents):
            x, y = starts[i % len(starts)]
            agents.append(
                AgentState(
                    agent_id=f"go2_{i+1}",
                    x=x,
                    y=y,
                    yaw=float(rng.uniform(-0.3, 0.3)),
                    sense_conspecifics=not blind_peers,
                    color=colors[i % len(colors)],
                )
            )
        w = World(WorldConfig(), zones, agents)
        for z in zones:
            w._zone_vel[z.name] = (0.12 if z.name == "A" else -0.10, 0.08 if z.name == "A" else 0.11)
        return w

    def local_reward(self, agent: AgentState) -> float:
        r = 0.0
        for z in self.zones:
            if z.contains(agent.x, agent.y):
                r += z.reward
        return float(np.clip(r, -1.0, 1.0))

    def perceive(self, agent: AgentState) -> Dict[str, float]:
        cues = {
            "cue_A": 0.0,
            "cue_B": 0.0,
            "peer_near": 0.0,
            "peer_at_B": 0.0,
            "threat": 0.0,
            "explore": 0.15,
        }
        # Projected/visible zone cues: distance-based "seeing" the projection
        for z in self.zones:
            if not z.projected:
                continue
            cx, cy = z.centroid()
            d = float(np.hypot(cx - agent.x, cy - agent.y))
            # Strong only nearby / inside — far-field washout prevented learning to reach B
            if z.contains(agent.x, agent.y):
                strength = 1.0
            elif d < 1.8:
                strength = float(np.clip(1.0 - d / 1.8, 0.0, 1.0))
            else:
                strength = 0.0
            if z.name == "A":
                cues["cue_A"] = max(cues["cue_A"], strength)
            elif z.name == "B":
                cues["cue_B"] = max(cues["cue_B"], strength)

        if agent.sense_conspecifics:
            for other in self.agents:
                if other.agent_id == agent.agent_id:
                    continue
                dx, dy = other.x - agent.x, other.y - agent.y
                dist = float(np.hypot(dx, dy))
                if dist > self.cfg.vision_range_m or dist < 1e-6:
                    continue
                # FOV check
                bearing = float(np.arctan2(dy, dx))
                ang = (bearing - agent.yaw + np.pi) % (2 * np.pi) - np.pi
                if abs(ang) > np.deg2rad(self.cfg.vision_fov_deg) * 0.5:
                    continue
                if dist < self.cfg.peer_near_m:
                    cues["peer_near"] = max(cues["peer_near"], 1.0 - dist / self.cfg.peer_near_m)
                # peer standing in B?
                for z in self.zones:
                    if z.name == "B" and z.contains(other.x, other.y):
                        cues["peer_at_B"] = max(cues["peer_at_B"], 0.8)
                if dist < 0.55:
                    cues["threat"] = 1.0
        return cues

    def apply_action(self, agent: AgentState, action: str) -> None:
        dt = self.cfg.dt
        if agent.freeze_timer > 0:
            agent.freeze_timer -= dt
            return
        if action == "Freeze":
            agent.freeze_timer = 1.0
            return

        target: Optional[Point] = None
        if action == "Approach_A":
            target = next(z.centroid() for z in self.zones if z.name == "A")
        elif action == "Approach_B":
            target = next(z.centroid() for z in self.zones if z.name == "B")
        elif action == "Avoid_A":
            ax, ay = next(z.centroid() for z in self.zones if z.name == "A")
            # flee then bias toward arena center / B side so agents don't pin on walls
            fx, fy = agent.x - (ax - agent.x), agent.y - (ay - agent.y)
            target = (0.7 * fx + 0.3 * (self.cfg.width * 0.75), 0.7 * fy + 0.3 * (self.cfg.height * 0.5))
        elif action == "Avoid_B":
            bx, by = next(z.centroid() for z in self.zones if z.name == "B")
            fx, fy = agent.x - (bx - agent.x), agent.y - (by - agent.y)
            target = (0.7 * fx + 0.3 * (self.cfg.width * 0.25), 0.7 * fy + 0.3 * (self.cfg.height * 0.5))
        elif action == "Explore":
            # random waypoint in free space (biased slightly toward B half after learning happens via Approach_B)
            tx = float(np.random.uniform(0.4, self.cfg.width - 0.4))
            ty = float(np.random.uniform(0.4, self.cfg.height - 0.4))
            target = (tx, ty)

        if target is None:
            return
        tx, ty = target
        desired = float(np.arctan2(ty - agent.y, tx - agent.x))
        # simple heading control
        err = (desired - agent.yaw + np.pi) % (2 * np.pi) - np.pi
        agent.yaw += float(np.clip(err, -2.5 * dt, 2.5 * dt))
        step = agent.speed * dt
        # soft collision with peers
        nx = agent.x + np.cos(agent.yaw) * step
        ny = agent.y + np.sin(agent.yaw) * step
        for other in self.agents:
            if other.agent_id == agent.agent_id:
                continue
            if np.hypot(nx - other.x, ny - other.y) < 0.7:
                # sidestep
                nx = agent.x + np.cos(agent.yaw + 1.2) * step
                ny = agent.y + np.sin(agent.yaw + 1.2) * step
                break
        agent.x = float(np.clip(nx, 0.15, self.cfg.width - 0.15))
        agent.y = float(np.clip(ny, 0.15, self.cfg.height - 0.15))

    def maybe_move_zones(self) -> None:
        if not self.cfg.move_zones:
            return
        dt = self.cfg.dt
        for z in self.zones:
            vx, vy = self._zone_vel.get(z.name, (0.0, 0.0))
            cand = translate_poly(z.polygon, vx * dt, vy * dt)
            xs = [p[0] for p in cand]
            ys = [p[1] for p in cand]
            if min(xs) < 0.2 or max(xs) > self.cfg.width - 0.2:
                self._zone_vel[z.name] = (-vx, vy)
                cand = translate_poly(z.polygon, -vx * dt, vy * dt)
            if min(ys) < 0.2 or max(ys) > self.cfg.height - 0.2:
                self._zone_vel[z.name] = (self._zone_vel[z.name][0], -vy)
                cand = translate_poly(z.polygon, 0.0, -vy * dt)
            z.polygon = cand

    def tick_timers(self, agent: AgentState) -> None:
        dt = self.cfg.dt
        for z in self.zones:
            if z.contains(agent.x, agent.y):
                if z.name == "A":
                    agent.t_in_a += dt
                elif z.name == "B":
                    agent.t_in_b += dt
