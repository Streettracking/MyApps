"""One teaching step, drawn as a fading KC→MBON flash.

The picture uses the same plasticity as ``MushroomBodyRuntime.plasticity``.
It does not teach, and it does not add a detector. ``connectome_mb_v1.npz``
has FlyWire root ids and cell types, not xyz, so KC sit on the lobe of their
type and MBON keep their real names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mb_runtime import ACTIONS, MBForward, MushroomBodyRuntime

FADE_S = 1.6
_REST_ALPHA = 0.20

_LOBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("γ", ("KCg-d", "KCg-s1", "KCg-s2", "KCg-s3", "KCg-m")),
    ("αβ", ("KCab-p", "KCab")),
    ("α′β′", ("KCapbp-ap1", "KCapbp-ap2", "KCapbp-m", "KCa'b'-ap1")),
)


@dataclass
class MbLayout:
    """Stable 2D places for every KC and MBON. Built once per brain."""

    kc_x: np.ndarray
    kc_y: np.ndarray
    mbon_x: np.ndarray
    mbon_y: np.ndarray
    mbon_label: list[str]
    mbon_approach: np.ndarray
    kc_root: np.ndarray
    lobe_name: tuple[str, ...]
    lobe_x: tuple[float, ...]
    n_pam: int
    n_ppl1: int


@dataclass
class LearnFlash:
    """Synapses that actually moved on one PAM or PPL1 event."""

    t: float
    kind: str
    kc_on: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    dw: np.ndarray
    toward: np.ndarray
    n_syn: int
    sum_abs: float
    before: float
    after: float
    mbon_before: np.ndarray
    mbon_after: np.ndarray
    strong_pre: int
    strong_post: int


def _approach_avoid() -> tuple[set[int], set[int]]:
    approach = {ACTIONS.index("Approach_A"), ACTIONS.index("Approach_B")}
    avoid = {ACTIONS.index("Avoid_A"), ACTIONS.index("Avoid_B")}
    return approach, avoid


def build_layout(brain: MushroomBodyRuntime) -> MbLayout:
    type_lobe: dict[str, int] = {}
    for lobe_i, (_name, names) in enumerate(_LOBES):
        for name in names:
            type_lobe[name] = lobe_i
    n_lobes = len(_LOBES) + 1
    buckets: list[list[tuple[int, str, int, str]]] = [[] for _ in range(n_lobes)]
    for i, g in enumerate(brain.kc_idx):
        g = int(g)
        ctype = str(brain.cell_types[g])
        side = str(brain.sides[g])
        root = int(brain.root_ids[g])
        buckets[type_lobe.get(ctype, len(_LOBES))].append((i, side, root, ctype))

    kc_x = np.zeros(brain.n_kc, dtype=np.float32)
    kc_y = np.zeros(brain.n_kc, dtype=np.float32)
    kc_root = np.zeros(brain.n_kc, dtype=np.int64)
    present = [i for i, bucket in enumerate(buckets) if bucket]
    # KC occupy the left of the synapse graph. MBON sit on the right edge.
    span = 0.70
    for slot, lobe_i in enumerate(present):
        center = 0.08 + span * (slot + 0.5) / max(len(present), 1)
        for side, sign in (("left", -1.0), ("right", 1.0)):
            rows = [row for row in buckets[lobe_i] if row[1] == side]
            leftover = [row for row in buckets[lobe_i] if row[1] not in ("left", "right")]
            if side == "right":
                rows.extend(leftover)
            rows.sort(key=lambda row: (row[3], row[2]))
            n = len(rows)
            for rank, (i, _side, root, _ctype) in enumerate(rows):
                kc_y[i] = 0.03 + 0.94 * ((rank + 0.5) / max(n, 1))
                kc_x[i] = center + sign * 0.055
                kc_root[i] = root
    lobe_names = []
    lobe_x = []
    for slot, lobe_i in enumerate(present):
        center = 0.08 + span * (slot + 0.5) / max(len(present), 1)
        lobe_names.append(_LOBES[lobe_i][0] if lobe_i < len(_LOBES) else "др.")
        lobe_x.append(center)

    approach, avoid = _approach_avoid()
    groups = {"ap": [], "av": [], "other": []}
    for i, g in enumerate(brain.mbon_idx):
        g = int(g)
        action = int(brain.mbon_action[i])
        if action in approach:
            groups["ap"].append(i)
        elif action in avoid:
            groups["av"].append(i)
        else:
            groups["other"].append(i)
    mbon_x = np.full(brain.n_mbon, 0.94, dtype=np.float32)
    mbon_y = np.full(brain.n_mbon, 0.5, dtype=np.float32)
    bands = (("ap", 0.04, 0.46), ("av", 0.54, 0.96), ("other", 0.47, 0.53))
    for key, y0, y1 in bands:
        rows = groups[key]
        rows.sort(key=lambda i: (str(brain.cell_types[int(brain.mbon_idx[i])]), int(brain.root_ids[int(brain.mbon_idx[i])])))
        n = len(rows)
        for rank, i in enumerate(rows):
            mbon_y[i] = y0 + (y1 - y0) * ((rank + 0.5) / max(n, 1))

    seen: set[str] = set()
    labels: list[str] = []
    approach_flag = np.zeros(brain.n_mbon, dtype=bool)
    for i, g in enumerate(brain.mbon_idx):
        g = int(g)
        side = "Л" if str(brain.sides[g]) == "left" else "П"
        base = f"{brain.cell_types[g]} {side}"
        label = base if base not in seen else f"{base} #{str(int(brain.root_ids[g]))[-4:]}"
        seen.add(label)
        labels.append(label)
        approach_flag[i] = int(brain.mbon_action[i]) in approach

    return MbLayout(
        kc_x=kc_x,
        kc_y=kc_y,
        mbon_x=mbon_x,
        mbon_y=mbon_y,
        mbon_label=labels,
        mbon_approach=approach_flag,
        kc_root=kc_root,
        lobe_name=tuple(lobe_names),
        lobe_x=tuple(lobe_x),
        n_pam=int(brain.n_dan_app),
        n_ppl1=int(brain.n_dan_av),
    )


def record_teacher_step(brain: MushroomBodyRuntime, fwd: MBForward, kind: str, t: float) -> LearnFlash:
    """Apply one teacher DAN and keep the synapses whose weight actually moved."""
    r = 1.0 if kind == "pam" else -1.0
    before = brain.appetitive_drive(fwd.action_scores)
    mbon_before = np.array(fwd.mbon, dtype=np.float32, copy=True)
    w0 = np.array(brain.kc_mbon_w, dtype=np.float32, copy=True)
    brain.plasticity(fwd, r)
    dw_all = brain.kc_mbon_w - w0
    changed = np.flatnonzero(np.abs(dw_all) > 1e-5)
    mbon_after, scores_after = brain.scores_of_kc(fwd.kc)
    after = brain.appetitive_drive(scores_after)
    kc_on = np.flatnonzero(fwd.kc > 0).astype(np.int32, copy=False)
    if changed.size:
        order = np.argsort(np.abs(dw_all[changed]))
        changed = changed[order]
        pre = brain.kc_mbon_pre[changed]
        post = brain.kc_mbon_post[changed]
        dw = dw_all[changed].astype(np.float32, copy=False)
        post_action = brain.mbon_action[post]
        approach, avoid = _approach_avoid()
        is_ap = np.isin(post_action, list(approach))
        is_av = np.isin(post_action, list(avoid))
        toward = (is_ap & (dw > 0)) | (is_av & (dw < 0))
        strong = int(np.argmax(np.abs(dw)))
        strong_pre = int(pre[strong])
        strong_post = int(post[strong])
        sum_abs = float(np.abs(dw).sum())
    else:
        pre = np.zeros(0, dtype=np.int32)
        post = np.zeros(0, dtype=np.int32)
        dw = np.zeros(0, dtype=np.float32)
        toward = np.zeros(0, dtype=bool)
        strong_pre = -1
        strong_post = -1
        sum_abs = 0.0
    return LearnFlash(
        t=float(t),
        kind="PAM" if kind == "pam" else "PPL1",
        kc_on=kc_on,
        pre=pre.astype(np.int32, copy=False),
        post=post.astype(np.int32, copy=False),
        dw=dw,
        toward=toward,
        n_syn=int(changed.size),
        sum_abs=sum_abs,
        before=float(before),
        after=float(after),
        mbon_before=mbon_before,
        mbon_after=np.array(mbon_after, dtype=np.float32, copy=True),
        strong_pre=strong_pre,
        strong_post=strong_post,
    )


def _footer_pair(screen, font, rect, left: LearnFlash | None, right: LearnFlash | None) -> None:
    y = rect.bottom - 56

    def bits(name: str, flash: LearnFlash | None) -> str:
        if flash is None:
            return "%s —" % name
        return "%s  синапсов %s  Σ|Δw| %.2f  %+.0f→%+.0f" % (
            name,
            flash.n_syn,
            flash.sum_abs,
            flash.before,
            flash.after,
        )

    screen.blit(font.render(bits("MB_L", left), True, (220, 226, 216)), (rect.x + 10, y))
    screen.blit(font.render(bits("MB_R", right), True, (220, 226, 216)), (rect.x + 10, y + 16))
    screen.blit(
        font.render("один DAN на обе половины. Зелёный — к подходу, красный — к избеганию.", True, (150, 160, 156)),
        (rect.x + 10, y + 34),
    )


def flash_alpha(age: float) -> float:
    """Bright for about 1.5 s, then a quiet residue until the next event."""
    if age < 0:
        age = 0.0
    if age >= FADE_S:
        return _REST_ALPHA
    return 1.0 - (1.0 - _REST_ALPHA) * (age / FADE_S)


def _scratch_surface(w: int, h: int):
    import pygame

    surf = getattr(_scratch_surface, "surf", None)
    if surf is None or surf.get_size() != (w, h):
        surf = pygame.Surface((max(w, 1), max(h, 1)), pygame.SRCALPHA)
        _scratch_surface.surf = surf
    else:
        surf.fill((0, 0, 0, 0))
    return surf


def draw_learn_panel(
    screen,
    font,
    font_sm,
    rect,
    layout: MbLayout | None,
    flash: LearnFlash | None,
    now: float,
    flash_r: LearnFlash | None = None,
) -> None:
    """Draw the flash. Called only while the panel is open.

    ``flash`` is MB_L and ``flash_r`` is MB_R. Both get the same PAM/PPL1 pulse.
    """
    import pygame

    pygame.draw.rect(screen, (12, 16, 20), rect)
    pygame.draw.rect(screen, (70, 110, 90), rect, 1)
    title = "вспышка обучения    две половины    G скрыть" if flash_r is not None else "вспышка обучения    G скрыть"
    screen.blit(font_sm.render(title, True, (210, 230, 214)), (rect.x + 10, rect.y + 6))
    if layout is None:
        screen.blit(font_sm.render("схема появится вместе с мозгом", True, (180, 186, 198)), (rect.x + 10, rect.y + 36))
        return

    age = None if flash is None else float(now) - float(flash.t)
    alpha = 0.0 if age is None else flash_alpha(age)
    _lamps(screen, font_sm, rect, layout, flash, alpha)
    if flash_r is None:
        graph = pygame.Rect(rect.x + 8, rect.y + 58, rect.w - 210, rect.h - 146)
        legend = pygame.Rect(rect.right - 196, rect.y + 58, 184, graph.h)
        _graph(screen, font_sm, graph, layout, flash, alpha)
        _legend(screen, font_sm, legend, layout, flash)
        _footer(screen, font_sm, rect, layout, flash)
        return
    span = rect.h - 58 - 74
    half_h = max(40, (span - 8) // 2)
    top = pygame.Rect(rect.x + 8, rect.y + 58, rect.w - 16, half_h)
    bot = pygame.Rect(rect.x + 8, top.bottom + 6, rect.w - 16, half_h)
    _graph(screen, font_sm, top, layout, flash, alpha)
    age_r = float(now) - float(flash_r.t)
    alpha_r = flash_alpha(age_r)
    _graph(screen, font_sm, bot, layout, flash_r, alpha_r)
    screen.blit(font_sm.render("MB_L", True, (186, 214, 196)), (top.x + 6, top.bottom - 16))
    screen.blit(font_sm.render("MB_R", True, (186, 214, 196)), (bot.x + 6, bot.bottom - 16))
    _footer_pair(screen, font_sm, rect, flash, flash_r)


def _lamps(screen, font, rect, layout: MbLayout, flash: LearnFlash | None, alpha: float) -> None:
    import pygame

    kind = flash.kind if flash is not None else ""
    pam_on = kind == "PAM"
    ppl_on = kind == "PPL1"
    y = rect.y + 30
    _lamp(screen, font, rect.x + 18, y, "PAM", f"{layout.n_pam} клеток", (70, 210, 120), pam_on, alpha if pam_on else 0.0)
    _lamp(screen, font, rect.x + 210, y, "PPL1", f"{layout.n_ppl1} клеток", (220, 84, 72), ppl_on, alpha if ppl_on else 0.0)
    hint = "учитель ещё не нажимал T или X" if flash is None else f"учитель включил {kind}"
    screen.blit(font.render(hint, True, (168, 186, 176)), (rect.x + 400, rect.y + 32))


def _lamp(screen, font, x, y, name, count, color, on: bool, alpha: float) -> None:
    import pygame

    radius = 11 if on and alpha > 0.55 else 8
    glow = tuple(int(c * (0.35 + 0.65 * alpha)) for c in color) if on else (48, 54, 60)
    pygame.draw.circle(screen, glow, (x + 8, y + 8), radius)
    if on:
        pygame.draw.circle(screen, (240, 250, 240), (x + 8, y + 8), radius, 1)
    screen.blit(font.render(name, True, color if on else (140, 146, 154)), (x + 24, y - 2))
    screen.blit(font.render(count, True, (130, 140, 136)), (x + 24, y + 14))


def _graph(screen, font, graph, layout: MbLayout, flash: LearnFlash | None, alpha: float) -> None:
    import pygame

    pygame.draw.rect(screen, (16, 20, 26), graph)
    for name, cx in zip(layout.lobe_name, layout.lobe_x):
        px = graph.x + int(cx * graph.w)
        screen.blit(font.render(name, True, (150, 164, 176)), (px - 10, graph.y + 2))
    screen.blit(font.render("MBON", True, (150, 164, 176)), (graph.right - 52, graph.y + 2))
    if flash is None or flash.n_syn == 0 and len(flash.kc_on) == 0:
        return

    def pt(nx: float, ny: float) -> tuple[int, int]:
        return (
            graph.x + 6 + int(np.clip(nx, 0.0, 1.0) * (graph.w - 16)),
            graph.y + 20 + int(np.clip(ny, 0.0, 1.0) * (graph.h - 28)),
        )

    overlay = _scratch_surface(graph.w, graph.h)
    ink = int(np.clip(alpha, 0.0, 1.0) * 168)
    if ink < 8 or flash.n_syn == 0:
        changed_kc: set[int] = set()
    else:
        mag = np.abs(flash.dw)
        hi = float(mag.max()) if mag.size else 1.0
        if hi < 1e-8:
            hi = 1.0
        changed_kc = set(int(i) for i in flash.pre)
        for pre, post, dw, toward in zip(flash.pre, flash.post, flash.dw, flash.toward):
            color = (64, 196, 112, ink) if bool(toward) else (210, 72, 64, ink)
            width = 1 + int(2.0 * (abs(float(dw)) / hi))
            a = pt(float(layout.kc_x[int(pre)]), float(layout.kc_y[int(pre)]))
            b = pt(float(layout.mbon_x[int(post)]), float(layout.mbon_y[int(post)]))
            pygame.draw.line(overlay, color, (a[0] - graph.x, a[1] - graph.y), (b[0] - graph.x, b[1] - graph.y), width)
    dot_a = max(ink, 70)
    for i in flash.kc_on:
        i = int(i)
        p = pt(float(layout.kc_x[i]), float(layout.kc_y[i]))
        if i in changed_kc:
            pygame.draw.circle(overlay, (236, 244, 220, dot_a), (p[0] - graph.x, p[1] - graph.y), 3)
        else:
            pygame.draw.circle(overlay, (120, 140, 150, max(40, dot_a // 2)), (p[0] - graph.x, p[1] - graph.y), 2)
    if flash.n_syn:
        delta = flash.mbon_after - flash.mbon_before
        for post in np.unique(flash.post):
            post = int(post)
            p = pt(float(layout.mbon_x[post]), float(layout.mbon_y[post]))
            d = float(delta[post])
            toward = (bool(layout.mbon_approach[post]) and d >= 0) or (not bool(layout.mbon_approach[post]) and d < 0)
            color = (90, 210, 120, 230) if toward else (220, 90, 80, 230)
            pygame.draw.circle(overlay, color, (p[0] - graph.x, p[1] - graph.y), 5)
    screen.blit(overlay, graph.topleft)


def _legend(screen, font, legend, layout: MbLayout, flash: LearnFlash | None) -> None:
    import pygame

    screen.blit(font.render("MBON Δ   верх подход", True, (186, 196, 188)), (legend.x, legend.y))
    screen.blit(font.render("низ — избегание", True, (160, 150, 146)), (legend.x, legend.y + 16))
    if flash is None or flash.n_syn == 0:
        screen.blit(font.render("нет изменённых", True, (140, 148, 156)), (legend.x, legend.y + 36))
        screen.blit(font.render("синапсов", True, (140, 148, 156)), (legend.x, legend.y + 52))
        return
    delta = flash.mbon_after - flash.mbon_before
    posts = np.unique(flash.post)
    order = [int(i) for i in posts[np.argsort(-np.abs(delta[posts]))]]
    notable = [i for i in order if abs(float(delta[i])) >= 1.0]
    rows = notable if len(notable) >= 4 else order[:6]
    y = legend.y + 36
    shown = 0
    for post in rows:
        if y > legend.bottom - 16:
            break
        d = float(delta[post])
        if abs(d) < 0.05:
            continue
        toward = (bool(layout.mbon_approach[post]) and d > 0) or (not bool(layout.mbon_approach[post]) and d < 0)
        color = (120, 210, 140) if toward else (220, 110, 100)
        num = f"{d:+.0f}" if abs(d) >= 10 else f"{d:+.1f}"
        text = f"{num}  {layout.mbon_label[post]}"
        screen.blit(font.render(text[:24], True, color), (legend.x, y))
        y += 16
        shown += 1
    extra = len(rows) - shown
    if extra > 0 and y <= legend.bottom - 16:
        screen.blit(font.render(f"ещё {extra}", True, (140, 148, 156)), (legend.x, y))


def _footer(screen, font, rect, layout: MbLayout, flash: LearnFlash | None) -> None:
    y = rect.bottom - 78
    if flash is None:
        screen.blit(font.render("T — PAM,  X — PPL1. Линии только у синапсов, чей вес изменился.", True, (176, 186, 180)), (rect.x + 10, y))
        screen.blit(font.render("зелёный — к подходу, красный — к избеганию, толщина = |Δw|", True, (150, 160, 156)), (rect.x + 10, y + 18))
        screen.blit(font.render("id и типы FlyWire. KC по долям γ, αβ, α′β′: xyz в файле нет.", True, (130, 142, 138)), (rect.x + 10, y + 36))
        return
    screen.blit(
        font.render(
            f"синапсов {flash.n_syn}     Σ|Δw| {flash.sum_abs:.2f}     KC активны {len(flash.kc_on)}",
            True,
            (220, 226, 216),
        ),
        (rect.x + 10, y),
    )
    screen.blit(
        font.render(
            f"выход MBON  {flash.before:+.0f} → {flash.after:+.0f}     сдвиг {flash.after - flash.before:+.0f}",
            True,
            (220, 226, 216),
        ),
        (rect.x + 10, y + 18),
    )
    if flash.strong_pre >= 0:
        root = int(layout.kc_root[flash.strong_pre])
        mbon = layout.mbon_label[flash.strong_post]
        strong = f"сильнейший KC {root} → {mbon}"
    else:
        strong = "синапс не сдвинулся: KC молчали или вес упёрся в ноль"
    screen.blit(font.render(strong, True, (176, 196, 186)), (rect.x + 10, y + 36))
    screen.blit(
        font.render("зелёный — к подходу, красный — к избеганию. Яркая ~1.5 с.", True, (140, 156, 148)),
        (rect.x + 10, y + 54),
    )


def flash_payload(flash: LearnFlash | None, limit: int = 400) -> dict | None:
    """Compact teaching step for the laptop. Strongest synapses only."""
    if flash is None:
        return None
    n = int(len(flash.dw))
    if n > limit:
        idx = np.argsort(np.abs(flash.dw))[-limit:]
    else:
        idx = np.arange(n)
    return {
        "t": float(flash.t),
        "kind": str(flash.kind),
        "n_syn": int(flash.n_syn),
        "sum_abs": float(flash.sum_abs),
        "before": float(flash.before),
        "after": float(flash.after),
        "kc_on": [int(i) for i in np.asarray(flash.kc_on).ravel()[:400]],
        "pre": [int(i) for i in np.asarray(flash.pre).ravel()[idx]],
        "post": [int(i) for i in np.asarray(flash.post).ravel()[idx]],
        "dw": [float(i) for i in np.asarray(flash.dw).ravel()[idx]],
        "toward": [bool(i) for i in np.asarray(flash.toward).ravel()[idx]],
        "mbon_before": [float(i) for i in np.asarray(flash.mbon_before).ravel()],
        "mbon_after": [float(i) for i in np.asarray(flash.mbon_after).ravel()],
    }


def flash_from_payload(data: dict | None) -> LearnFlash | None:
    if not data:
        return None
    return LearnFlash(
        t=float(data.get("t", 0.0)),
        kind=str(data.get("kind", "")),
        kc_on=np.asarray(data.get("kc_on", []), dtype=np.int32),
        pre=np.asarray(data.get("pre", []), dtype=np.int32),
        post=np.asarray(data.get("post", []), dtype=np.int32),
        dw=np.asarray(data.get("dw", []), dtype=np.float32),
        toward=np.asarray(data.get("toward", []), dtype=bool),
        n_syn=int(data.get("n_syn", 0)),
        sum_abs=float(data.get("sum_abs", 0.0)),
        before=float(data.get("before", 0.0)),
        after=float(data.get("after", 0.0)),
        mbon_before=np.asarray(data.get("mbon_before", []), dtype=np.float32),
        mbon_after=np.asarray(data.get("mbon_after", []), dtype=np.float32),
        strong_pre=-1,
        strong_post=-1,
    )
