"""Pygame visualization for the arena simulator (Windows / macOS / Linux)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mb_runtime import ACTIONS
from .raw_sense import N_AZ, OFF_BODY, OFF_LIDAR
from .simulator import SimConfig, Simulator

PANEL_W = 460
HUD_H = 96

ACTION_COLORS = {
    "Approach_A": (214, 92, 92),
    "Avoid_A": (232, 140, 72),
    "Approach_B": (72, 186, 112),
    "Avoid_B": (78, 156, 214),
    "Explore": (214, 186, 78),
    "Freeze": (168, 168, 186),
}


def bin_activity(values: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Mean-pool a population into a cols×rows heat map (KC downsample)."""
    bins = cols * rows
    acc = np.zeros(bins, dtype=np.float32)
    n = int(len(values))
    if n == 0 or bins == 0:
        return acc.reshape(rows, cols)
    idx = (np.arange(n) * bins) // n
    np.add.at(acc, idx, values.astype(np.float32, copy=False))
    counts = np.bincount(idx, minlength=bins).astype(np.float32)
    acc /= np.maximum(counts, 1.0)
    return acc.reshape(rows, cols)


def _heat(value: float, vmax: float) -> tuple[int, int, int]:
    t = 0.0 if vmax <= 1e-8 else float(np.clip(value / vmax, 0.0, 1.0))
    # dark slate → amber foci
    return (
        int(18 + 230 * t),
        int(22 + 150 * t),
        int(36 + 40 * (1.0 - t)),
    )


def _mix(base: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(18 + (c - 18) * t) for c in base)  # type: ignore[return-value]


def draw_mb_panel(screen, rect, brain, agent, font, font_sm) -> None:
    """Schematic PN / KC / MBON layers driven by the latest MBForward."""
    import pygame

    x, y, w, h = rect
    pygame.draw.rect(screen, (16, 18, 24), pygame.Rect(x, y, w, h))
    pygame.draw.line(screen, (48, 52, 64), (x, y), (x, y + h), 2)

    fwd = brain.last_forward
    title = font.render(f"MB  {agent.agent_id}", True, (236, 236, 240))
    screen.blit(title, (x + 14, y + 10))

    if agent.r < 0:
        dan_label, dan_color = "DAN aversive  PPL1", (214, 78, 78)
    elif agent.r > 0:
        dan_label, dan_color = "DAN appetitive  PAM", (72, 186, 112)
    else:
        dan_label, dan_color = "DAN silent", (110, 114, 128)
    pygame.draw.rect(screen, dan_color, pygame.Rect(x + 14, y + 36, 14, 14))
    screen.blit(font_sm.render(f"{dan_label}   r={agent.r:+.0f}", True, dan_color), (x + 36, y + 36))

    if fwd is None:
        screen.blit(font.render("no forward yet", True, (160, 160, 170)), (x + 14, y + 70))
        return

    pn_on = int(np.count_nonzero(fwd.pn))
    kc_on = int(np.count_nonzero(fwd.kc))
    stats = font_sm.render(
        f"PN {pn_on}/{len(fwd.pn)}   KC {kc_on}/{len(fwd.kc)}   MBON max {float(fwd.mbon.max()):.1f}",
        True,
        (180, 184, 196),
    )
    screen.blit(stats, (x + 14, y + 56))

    cursor_y = y + 78
    raw = getattr(brain, "last_raw", None)
    if raw is not None:
        cursor_y = _draw_raw_strip(screen, font_sm, raw, x + 14, cursor_y, w - 28)
        cursor_y += 6
    like = getattr(brain, "last_likeness", None)
    if like is not None:
        cursor_y = _draw_recognition(
            screen, font_sm, float(like), getattr(brain, "last_recog", None), x + 14, cursor_y, w - 28
        )
        cursor_y += 6
    tight = like is not None
    kc_rows = 5 if tight else (8 if raw is not None else 12)
    cursor_y = _draw_heat_block(
        screen, font_sm, fwd.pn, x + 14, cursor_y, w - 28, cols=40, rows=3, label="PN"
    )
    cursor_y += 6 if tight else 8
    cursor_y = _draw_heat_block(
        screen, font_sm, fwd.kc, x + 14, cursor_y, w - 28, cols=32, rows=kc_rows, label="KC"
    )
    cursor_y += 6 if tight else 8
    cursor_y = _draw_mbon_block(
        screen, font_sm, brain, fwd, x + 14, cursor_y, w - 28, max_cell=12 if tight else 16
    )
    cursor_y += 8 if tight else 10
    _draw_scores(screen, font_sm, fwd, agent.action, x + 14, cursor_y, w - 28, step=14 if tight else 16)


def _draw_raw_strip(screen, font, raw, x, y, width) -> int:
    """Egocentric camera row and lidar bins. Not a class label."""
    import pygame

    screen.blit(font.render("raw camera + lidar", True, (200, 204, 214)), (x, y))
    y += 16
    gap = 4
    cell = max(10, (width - gap * (N_AZ - 1)) // N_AZ)
    for i in range(N_AZ):
        r = int(np.clip(raw[OFF_BODY + i * 3 + 0], 0, 1) * 255)
        g = int(np.clip(raw[OFF_BODY + i * 3 + 1], 0, 1) * 255)
        b = int(np.clip(raw[OFF_BODY + i * 3 + 2], 0, 1) * 255)
        if r + g + b < 8:
            color = (28, 32, 40)
        else:
            color = (r, g, b)
        pygame.draw.rect(screen, color, pygame.Rect(x + i * (cell + gap), y, cell, 16))
        h = int(np.clip(raw[OFF_LIDAR + i], 0, 1) * 18)
        pygame.draw.rect(screen, (70, 150, 170), pygame.Rect(x + i * (cell + gap), y + 20, cell, h))
    return y + 42


def _draw_recognition(screen, font, likeness: float, proto, x, y, width) -> int:
    """Learned conspecific-likeness, drawn before PN. Not a zone reward."""
    import pygame

    screen.blit(
        font.render(f"recognition   likeness {likeness:.2f}", True, (176, 214, 186)),
        (x, y),
    )
    y += 16
    pygame.draw.rect(screen, (32, 34, 42), pygame.Rect(x, y, width, 12))
    pygame.draw.rect(
        screen,
        (88, 196, 140),
        pygame.Rect(x, y, max(1, int(width * float(np.clip(likeness, 0.0, 1.0)))), 12),
    )
    y += 16
    if proto is not None and len(proto):
        vmax = float(np.max(np.abs(proto)))
        if vmax < 1e-6:
            vmax = 1.0
        gap = 4
        cell = max(10, (width - gap * (len(proto) - 1)) // len(proto))
        base = y + 14
        for i, value in enumerate(proto):
            h = int(14 * max(float(value) / vmax, 0.0))
            pygame.draw.rect(
                screen,
                (70, 150, 120),
                pygame.Rect(x + i * (cell + gap), base - h, cell, max(h, 1)),
            )
        y = base + 4
    return y


def _draw_heat_block(screen, font, values, x, y, width, cols, rows, label) -> int:
    import pygame

    heat = bin_activity(values, cols, rows)
    vmax = float(heat.max()) if heat.size else 0.0
    screen.blit(font.render(f"{label}  peak {vmax:.2f}", True, (200, 204, 214)), (x, y))
    y += 18
    gap = 1
    cell_w = max(2, (width - gap * (cols - 1)) // cols)
    cell_h = max(3, min(14, cell_w))
    for r in range(rows):
        for c in range(cols):
            val = float(heat[r, c])
            color = _heat(val, vmax)
            rx = x + c * (cell_w + gap)
            ry = y + r * (cell_h + gap)
            if vmax > 0 and val >= 0.55 * vmax:
                pygame.draw.rect(screen, (255, 220, 140), pygame.Rect(rx - 1, ry - 1, cell_w + 2, cell_h + 2))
            pygame.draw.rect(screen, color, pygame.Rect(rx, ry, cell_w, cell_h))
    return y + rows * (cell_h + gap)


def _draw_mbon_block(screen, font, brain, fwd, x, y, width, max_cell: int = 16) -> int:
    import pygame

    screen.blit(font.render("MBON", True, (200, 204, 214)), (x, y))
    y += 18
    cols = 16
    n = len(fwd.mbon)
    rows = int(np.ceil(n / cols)) if n else 1
    gap = 2
    cell = max(8, min(max_cell, (width - gap * (cols - 1)) // cols))
    vmax = float(fwd.mbon.max()) if n else 0.0
    for i in range(n):
        r, c = divmod(i, cols)
        t = 0.0 if vmax <= 1e-8 else float(fwd.mbon[i] / vmax)
        base = ACTION_COLORS[ACTIONS[int(brain.mbon_action[i])]]
        color = _mix(base, 0.25 + 0.75 * t)
        rx = x + c * (cell + gap)
        ry = y + r * (cell + gap)
        if t >= 0.65:
            pygame.draw.rect(screen, (255, 236, 180), pygame.Rect(rx - 1, ry - 1, cell + 2, cell + 2))
        pygame.draw.rect(screen, color, pygame.Rect(rx, ry, cell, cell))
    return y + rows * (cell + gap)


def _draw_scores(screen, font, fwd, executed: str, x, y, width, step: int = 16) -> None:
    import pygame

    screen.blit(
        font.render(f"action scores   motor {executed}", True, (200, 204, 214)),
        (x, y),
    )
    y += 18
    scores = fwd.action_scores
    vmax = float(max(float(scores.max()), 1e-6))
    bar_w = width - 108
    for i, name in enumerate(ACTIONS):
        t = float(scores[i]) / vmax
        pygame.draw.rect(screen, (32, 34, 42), pygame.Rect(x + 100, y, bar_w, 12))
        pygame.draw.rect(
            screen,
            ACTION_COLORS[name],
            pygame.Rect(x + 100, y, max(1, int(bar_w * t)), 12),
        )
        mark = ">" if name == fwd.action else " "
        screen.blit(font.render(f"{mark}{name}", True, (210, 210, 216)), (x, y - 1))
        y += step


def run_pygame(
    npz: Path,
    duration_s: float = 180.0,
    n_agents: int = 3,
    move_zones: bool = False,
    sense: bool = True,
    scale: int = 120,
    percept: str = "fixed",
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
        percept=percept,
        log_dir=None,
    )
    # For unlimited GUI, still step indefinitely until quit
    unlimited = duration_s >= 1e7 or duration_s <= 0
    sim = Simulator(cfg)
    w_m, h_m = sim.world.cfg.width, sim.world.cfg.height
    arena_w, arena_h = int(w_m * scale), int(h_m * scale)
    screen_w = arena_w + PANEL_W

    pygame.init()
    screen = pygame.display.set_mode((screen_w, arena_h + HUD_H))
    pygame.display.set_caption("FlyWire MB × Go2 arena sim")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    font_sm = pygame.font.SysFont("consolas", 13)

    def to_px(x, y):
        return int(x * scale), int(arena_h - y * scale)

    running = True
    paused = False
    focus = 0
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
                elif pygame.K_1 <= event.key <= pygame.K_3:
                    idx = event.key - pygame.K_1
                    if idx < len(sim.world.agents):
                        focus = idx

        if not paused:
            sim.step()
            step_i += 1

        focus = min(focus, len(sim.world.agents) - 1)
        focused = sim.world.agents[focus]

        screen.fill((24, 26, 32))
        pygame.draw.rect(screen, (36, 40, 48), pygame.Rect(0, 0, arena_w, arena_h))

        for z in sim.world.zones.values():
            pts = [to_px(float(px), float(py)) for px, py in z.polygon]
            surf = pygame.Surface((arena_w, arena_h), pygame.SRCALPHA)
            col = (*z.color, 110)
            pygame.draw.polygon(surf, col, pts)
            screen.blit(surf, (0, 0))
            pygame.draw.polygon(screen, z.color, pts, 2)
            cx, cy = z.centroid()
            label = font.render(f"{z.name} r={z.reward:+.0f}", True, (240, 240, 240))
            screen.blit(label, to_px(cx - 0.3, cy))

        for obj in sim.world.distractors:
            ox, oy = to_px(obj.x, obj.y)
            still = abs(obj.vx) + abs(obj.vy) < 1e-6
            box = pygame.Rect(ox - 5, oy - 5, 10, 10)
            if still:
                pygame.draw.rect(screen, (150, 150, 146), box, 2)
            else:
                pygame.draw.rect(screen, (150, 100, 48), box)

        for ag in sim.world.agents:
            pts = [to_px(px, py) for px, py in ag.trail[::2]]
            if len(pts) >= 2:
                pygame.draw.lines(screen, ag.color, False, pts, 1)
            px, py = to_px(ag.x, ag.y)
            if ag.agent_id == focused.agent_id:
                pygame.draw.circle(screen, (255, 255, 255), (px, py), 14, 2)
            pygame.draw.circle(screen, ag.color, (px, py), 10)
            tag = font.render(ag.action.replace("_", ""), True, (230, 230, 230))
            screen.blit(tag, (px + 12, py - 8))

        draw_mb_panel(
            screen,
            (arena_w, 0, PANEL_W, arena_h),
            sim.brains[focused.agent_id],
            focused,
            font,
            font_sm,
        )

        pygame.draw.rect(screen, (18, 18, 22), pygame.Rect(0, arena_h, screen_w, HUD_H))
        mean_pi = float(
            np.mean(
                [
                    (a.time_in_B - a.time_in_A) / (a.time_in_A + a.time_in_B + 1e-6)
                    for a in sim.world.agents
                ]
            )
        )
        hud = (
            f"t={sim.world.t:5.1f}s  PI={mean_pi:+.3f}  "
            f"zones={'MOVE' if sim.world.cfg.move_zones else 'STATIC'}  "
            f"peers={'ON' if sense else 'BLIND'}  percept={sim.cfg.percept}  brain={focused.agent_id}"
        )
        screen.blit(font.render(hud, True, (220, 220, 220)), (12, arena_h + 6))
        screen.blit(
            font.render("[1][2][3] brain    [Space] pause    [Z] zones    [Esc] quit", True, (170, 174, 184)),
            (12, arena_h + 24),
        )
        y = arena_h + 46
        for i, ag in enumerate(sim.world.agents):
            pi = (ag.time_in_B - ag.time_in_A) / (ag.time_in_A + ag.time_in_B + 1e-6)
            mark = ">" if i == focus else " "
            line = (
                f"{mark}[{i+1}] {ag.agent_id}: PI={pi:+.2f} "
                f"A={ag.time_in_A:.0f}s B={ag.time_in_B:.0f}s r={ag.r:+.0f} {ag.action}"
            )
            screen.blit(font.render(line, True, ag.color), (12, y))
            y += 16

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return sim.summary()
