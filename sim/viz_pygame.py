"""Pygame visualization for the arena simulator (Windows / macOS / Linux)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .simulator import SimConfig, Simulator


def run_pygame(
    npz: Path,
    duration_s: float = 180.0,
    n_agents: int = 3,
    move_zones: bool = False,
    sense: bool = True,
    scale: int = 120,
) -> dict:
    import pygame

    # Avoid absurd step counts when seconds=0 (unlimited until Esc)
    effective = duration_s if duration_s < 1e7 else 1e9
    cfg = SimConfig(
        npz_path=npz,
        duration_s=min(effective, 1e6) if effective < 1e7 else 3600.0,
        n_agents=n_agents,
        sense_conspecifics=sense,
        move_zones=move_zones,
        log_dir=None,
    )
    # For unlimited GUI, still step indefinitely until quit
    unlimited = duration_s >= 1e7 or duration_s <= 0
    sim = Simulator(cfg)
    w_m, h_m = sim.world.cfg.width, sim.world.cfg.height
    screen_w, screen_h = int(w_m * scale), int(h_m * scale)

    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h + 72))
    pygame.display.set_caption("FlyWire MB × Go2 arena sim")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    def to_px(x, y):
        return int(x * scale), int(screen_h - y * scale)

    running = True
    paused = False
    steps_total = 10**12 if unlimited else int(duration_s / sim.world.cfg.dt)
    step_i = 0

    while running and step_i < steps_total:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_z:
                    sim.world.cfg.move_zones = not sim.world.cfg.move_zones
                    sim.cfg.move_zones = sim.world.cfg.move_zones

        if not paused:
            sim.step()
            step_i += 1

        screen.fill((24, 26, 32))
        # floor
        pygame.draw.rect(screen, (36, 40, 48), pygame.Rect(0, 0, screen_w, screen_h))

        # projected zones
        for z in sim.world.zones.values():
            pts = [to_px(float(x), float(y)) for x, y in z.polygon]
            surf = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            col = (*z.color, 110)
            pygame.draw.polygon(surf, col, pts)
            screen.blit(surf, (0, 0))
            pygame.draw.polygon(screen, z.color, pts, 2)
            cx, cy = z.centroid()
            label = font.render(f"{z.name} r={z.reward:+.0f}", True, (240, 240, 240))
            screen.blit(label, to_px(cx - 0.3, cy))

        # agents
        for ag in sim.world.agents:
            pts = [to_px(x, y) for x, y in ag.trail[::2]]
            if len(pts) >= 2:
                pygame.draw.lines(screen, ag.color, False, pts, 1)
            px, py = to_px(ag.x, ag.y)
            pygame.draw.circle(screen, ag.color, (px, py), 10)
            tag = font.render(ag.action.replace("_", ""), True, (230, 230, 230))
            screen.blit(tag, (px + 12, py - 8))

        # HUD
        pygame.draw.rect(screen, (18, 18, 22), pygame.Rect(0, screen_h, screen_w, 72))
        mean_pi = float(
            np.mean(
                [
                    (a.time_in_B - a.time_in_A) / (a.time_in_A + a.time_in_B + 1e-6)
                    for a in sim.world.agents
                ]
            )
        )
        hud = (
            f"t={sim.world.t:6.1f}s  mean_PI={mean_pi:+.3f}  "
            f"zones={'MOVE' if sim.world.cfg.move_zones else 'STATIC'}  "
            f"peers={'ON' if sense else 'BLIND'}  "
            f"[Space] pause  [Z] toggle move zones  [Esc] quit"
        )
        screen.blit(font.render(hud, True, (220, 220, 220)), (12, screen_h + 12))
        y = screen_h + 36
        for ag in sim.world.agents:
            pi = (ag.time_in_B - ag.time_in_A) / (ag.time_in_A + ag.time_in_B + 1e-6)
            line = f"{ag.agent_id}: PI={pi:+.2f} A={ag.time_in_A:.0f}s B={ag.time_in_B:.0f}s r={ag.r:+.0f} {ag.action}"
            screen.blit(font.render(line, True, ag.color), (12, y))
            y += 14
            if y > screen_h + 60:
                break

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return sim.summary()
