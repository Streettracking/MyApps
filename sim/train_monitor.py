"""Live monitor for conspecific-recognition training.

Operator labels and ground truth drawn here never go back into the learner.
The window also owns the drive keys. They are reported to the caller; this
module does not send UDP itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sdl_thread import require_main_thread
from .ui_guard import (
    apply_cursor,
    borrowed_window,
    flatten_theme_colors,
    force_setmode,
    install_cursors,
    no_caption,
    no_card_cache,
    no_present,
    no_sdl_window,
    no_snap,
    sdl_window_on_resize,
    throwaway_window,
    tuple_colors,
)
from .dock_layout import (
    BLOCK_TITLES,
    BLOCKS,
    CONTROLS_BUTTON_H,
    HEADER as DOCK_HEADER,
    LIDAR_MIN_H,
    SPLIT as DOCK_SPLIT,
    content_rect as dock_content,
    default_tree,
    find_leaf,
    header_controls,
    hit_header,
    layout_dock,
    swap_blocks,
    tree_from_dict,
)
from .hemifield import DEFAULT_OVERLAP
from .ui_settings import default_settings, load_settings, persist_enabled, save_settings
from .ui_theme import (
    AMBER_SOFT,
    BG_BOT,
    BG_TOP,
    CARD,
    ESTOP_EDGE,
    ESTOP_LABEL,
    INK_DIM,
    INSET,
    KEY_BADGE,
    KEY_BADGE_INK,
    LABEL,
    LAMP_ESTOP_HI,
    LAMP_GREEN_HI,
    LAMP_RED_HI,
    LAMP_WASH,
    LINE,
    MENU,
    PEACH_SOFT,
    RING,
    SCROLL_THUMB,
    SCROLL_TRACK,
    SHADOW,
    TEACHER_SOFT,
    VALUE,
    apply_theme,
    theme_name,
)
from .tabnum import (
    EYES_STATUS,
    MODE_W,
    RAW_STATUS,
    SKIP_STATUS,
    SOUND_STATUS,
    format_box_tag,
    format_command,
    format_drift_span,
    format_drive_line,
    format_familiar,
    format_kc_scales,
    format_lamps,
    format_pref_line,
    format_stim_line,
    format_half,
    format_hebb_proto,
    format_kc,
    format_lifetime,
    format_metrics_line,
    format_overlap,
    format_pilot_mode,
    format_pilot_rest,
    format_plaque,
    format_purity,
    format_raw,
    format_record,
    format_session_counts,
    format_span,
    paint_mixed,
    paint_wrapped,
    skips_are_quiet,
)

WIN_W = 1920
LIDAR_LEARN_NOTE = "в обучении не участвует"
WIN_H = 1080
MIN_SCALE = 0.75
MIN_CONTENT_W = int(round(WIN_W * MIN_SCALE))  # 1440
MIN_FONT_PX = 13
SCROLL_BAR = 8
MENU_H = 32
FOOTER_H = 56

# Light warm chrome is the default. Status colours stay semantic in both themes.
# Peach is on. Amber is a warning. Red is an error, E-STOP, or PPL1.
# Lamp rims, the card shadow, and the key plaque live in ui_theme.
CARD_ALPHA = 217  # ~0.85
INK = VALUE
PEACH = (201, 120, 91)  # #C9785B actively on
GREEN = (141, 181, 150)  # #8DB596 узнаю / PAM
RED = (217, 136, 128)  # #D98880 PPL1 and errors
TEACHER = (46, 154, 206)  # #2E9ACE teacher watch, same in both themes
ESTOP = (184, 40, 36)
AMBER = (214, 154, 52)  # #D69A34 warning
LAMP_OFF = ("off",)
PANEL = CARD
BTN = CARD
BTN_EDGE = LINE


@dataclass
class MonitorInput:
    quit: bool = False
    steer_x: float = 0.0
    steer_z: float = 0.0
    stop: bool = False
    stand_up: bool = False
    stand_down: bool = False
    recovery: bool = False
    estop: bool = False
    pause_learn: bool = False
    reset: bool = False
    save: bool = False
    load: bool = False
    screenshot: bool = False
    label: str | None = None
    treat: bool = False
    punish: bool = False
    beep_toggle: bool = False
    lidar_reset: bool = False
    lidar_toggle: bool = False
    flash_toggle: bool = False
    autonomy_toggle: bool = False
    steer_toggle: bool = False
    teacher_toggle: bool = False
    boxes_toggle: bool = False
    brain_toggle: bool = False
    go2_control: bool = False
    fullscreen_toggle: bool = False
    record_toggle: bool = False
    doggibot: bool = False
    takeover: bool = False
    treat_down: bool = False
    punish_down: bool = False
    focused: bool = True
    teacher_weight: float | None = None
    approach_reset: bool = False
    run_stage: str | None = None
    run_side: str | None = None


@dataclass
class MonitorView:
    title: str = "recognition training"
    camera: np.ndarray | None = None
    lidar: np.ndarray | None = None
    sensor_error: str = ""
    likeness: float = 0.0
    learned_match: float = 0.0
    peer_curve: list[tuple[float, float]] = field(default_factory=list)
    other_curve: list[tuple[float, float]] = field(default_factory=list)
    metric_curve: list[dict] = field(default_factory=list)
    proto_self: np.ndarray | None = None
    proto_other: np.ndarray | None = None
    log_lines: list[str] = field(default_factory=list)
    paused: bool = False
    n_self: int = 0
    n_other: int = 0
    udp_status: str = ""
    last_command: str = ""
    mode_label: str = "sim"
    operator_label: str = "sim ground truth"
    t: float = 0.0
    focused: bool = True
    keys_hint: str = ""
    learner: str = "hebb"
    dan_mode: str = ""
    kc_on: int = 0
    kc_n: int = 0
    kc_bins: np.ndarray | None = None
    drift: float = 0.0
    session_drift: float | None = None
    search_lamp: bool = False
    drift_curve: list[tuple[float, float]] = field(default_factory=list)
    dan_events: list[tuple[float, str]] = field(default_factory=list)
    treat_flash: bool = False
    readout_caption: str = ""
    n_pam: int = 0
    n_ppl1: int = 0
    recognized: bool = False
    confidence: float = 0.0
    confidence_ready: bool = False
    session_time: float = 0.0
    total_time: float = 0.0
    total_pam: int = 0
    total_ppl1: int = 0
    session_sep: float | None = None
    total_sep: float | None = None
    session_acc: float | None = None
    total_acc: float | None = None
    session_labeled: int = 0
    total_labeled: int = 0
    session_novelty: int = 0
    total_novelty: int = 0
    marks: list = field(default_factory=list)
    lidar_mode: str = ""
    lidar_hold: str = ""
    lidar_warning: str = ""
    lidar_fresh_on: bool = True
    learn_flash: object | None = None
    mb_layout: object | None = None
    pilot_mode: str = "РУЧНОЕ"
    pilot_phase: str = "stop"
    pilot_who: str = "оператор"
    pilot_hint: str = ""
    range_line: str = ""
    fly_line: str = ""
    steer: str = "bilateral"
    learn_flash_r: object | None = None
    learning_on: bool = True
    autonomy_on: bool = False
    onboard: bool = False
    record_on: bool = False
    record_saved: int = 0
    record_bytes: int = 0
    record_idle: bool = False
    hemi_l: float | None = None
    hemi_r: float | None = None
    hemi_z: float | None = None
    lifetime_known: bool = True
    overlap: float = DEFAULT_OVERLAP
    last_seen_side: str = ""
    yolo_state: str = ""
    teacher_counts: str = ""
    teacher_skips: str = ""
    teacher_flash_l: str = ""
    teacher_flash_r: str = ""
    teacher_boxes: list = field(default_factory=list)
    eye_l_recognized: bool = False
    eye_r_recognized: bool = False
    eye_l_confidence: float = 0.0
    eye_r_confidence: float = 0.0
    eye_l_ready: bool = False
    eye_r_ready: bool = False
    phase_ru: str = ""
    eyes_line: str = ""
    recog_line: str = ""
    teacher_weight: float = 0.0
    lead_plaque: str = ""
    brain_offline: bool = False
    drive_l: float | None = None
    drive_r: float | None = None
    turn: float | None = None
    stim_dog: float | None = None
    stim_cube: float | None = None
    pref_index: float | None = None
    pref_seconds: float = 0.0
    kc_gd_frac: float | None = None
    kc_abp_frac: float | None = None
    pam_l: int = 0
    pam_r: int = 0
    ppl1_l: int = 0
    ppl1_r: int = 0
    run_stage: str = "до"
    run_side: str = "слева"


def fit_window(ww: int, wh: int) -> tuple:
    """Width sets a uniform scale. Returns ``(scale, logical_w, logical_h, content_w, content_h)``.

    Scale never drops below 0.75, so the canvas stays at least 1440 window
    pixels wide. Extra window height grows the logical canvas from the top;
    a shorter window keeps the 1080-tall layout and scrolls. Content is
    top-left, never letterboxed.
    """
    ww = max(1, int(ww))
    wh = max(1, int(wh))
    if ww >= MIN_CONTENT_W:
        scale = ww / float(WIN_W)
        content_w = ww
    else:
        scale = MIN_SCALE
        content_w = MIN_CONTENT_W
    min_content_h = max(1, int(round(WIN_H * scale)))
    if wh >= min_content_h:
        content_h = wh
        logical_h = max(WIN_H, int(round(wh / scale)))
        while int(round(logical_h * scale)) < wh:
            logical_h += 1
    else:
        content_h = min_content_h
        logical_h = WIN_H
    return scale, WIN_W, logical_h, content_w, content_h


def font_px(requested: int, scale: float) -> int:
    """Logical font size whose on-screen size is at least ``MIN_FONT_PX``."""
    import math

    need = int(math.ceil(float(MIN_FONT_PX) / max(float(scale), 1e-6)))
    return max(int(requested), need)


def scroll_limits(ww: int, wh: int) -> tuple:
    _scale, _lw, _lh, content_w, content_h = fit_window(ww, wh)
    return max(0, content_w - int(ww)), max(0, content_h - int(wh))


def scrollbar_geom(ww: int, wh: int, scroll_x: int = 0, scroll_y: int = 0, content_h: int | None = None) -> dict:
    """Tracks and thumbs in window pixels. Missing bars are ``None``."""
    import pygame

    ww = int(ww)
    wh = int(wh)
    max_x, max_y = scroll_limits(ww, wh)
    _scale, _lw, _lh, content_w, fitted_h = fit_window(ww, wh)
    if content_h is None:
        content_h = fitted_h
    else:
        content_h = max(1, int(content_h))
        max_y = max(0, content_h - int(wh))
    sx = max(0, min(int(scroll_x), max_x))
    sy = max(0, min(int(scroll_y), max_y))
    show_x = max_x > 0
    show_y = max_y > 0
    track_x = track_y = thumb_x = thumb_y = None
    if show_y:
        track_h = wh - (SCROLL_BAR if show_x else 0)
        track_y = pygame.Rect(ww - SCROLL_BAR, 0, SCROLL_BAR, max(1, track_h))
        thumb_h = max(28, int(track_y.h * min(1.0, wh / float(max(content_h, 1)))))
        thumb_h = min(thumb_h, track_y.h)
        travel = max(1, track_y.h - thumb_h)
        thumb_y = pygame.Rect(track_y.x, int(round(travel * sy / float(max_y))) if max_y else 0, SCROLL_BAR, thumb_h)
    if show_x:
        track_w = ww - (SCROLL_BAR if show_y else 0)
        track_x = pygame.Rect(0, wh - SCROLL_BAR, max(1, track_w), SCROLL_BAR)
        thumb_w = max(28, int(track_x.w * min(1.0, ww / float(max(content_w, 1)))))
        thumb_w = min(thumb_w, track_x.w)
        travel = max(1, track_x.w - thumb_w)
        thumb_x = pygame.Rect(int(round(travel * sx / float(max_x))) if max_x else 0, track_x.y, thumb_w, SCROLL_BAR)
    return {
        "max_x": max_x,
        "max_y": max_y,
        "scroll": (sx, sy),
        "track_x": track_x,
        "track_y": track_y,
        "thumb_x": thumb_x,
        "thumb_y": thumb_y,
        "content": (content_w, content_h),
    }


def logical_from_window(pos, ww: int, wh: int, scroll=(0, 0), footer_h: int | None = None, canvas_h: int | None = None) -> tuple:
    """Window pixels → canvas pixels, including scroll. A bar click is ``(-1, -1)``."""
    scale, logical_w, logical_h, content_w, content_h = fit_window(ww, wh)
    footer = FOOTER_H if footer_h is None else max(1, int(footer_h))
    if canvas_h is not None:
        logical_h = max(1, int(canvas_h))
        content_h = max(1, int(round(logical_h * scale)))
    geom = scrollbar_geom(ww, wh, scroll[0], scroll[1], content_h=content_h if canvas_h is not None else None)
    px, py = int(pos[0]), int(pos[1])
    for bar in (geom["track_x"], geom["track_y"]):
        if bar is not None and bar.collidepoint(px, py):
            return (-1, -1)
    sx, sy = geom["scroll"]
    bands = _chrome_bands(scale, wh, content_h, footer_h=footer)
    if bands["pinned"]:
        menu_px = bands["menu_px"]
        footer_px = bands["footer_px"]
        x = px + sx
        if py < menu_px:
            y_logical = py / scale
        elif py >= wh - footer_px:
            y_logical = (logical_h - footer) + (py - (wh - footer_px)) / scale
        else:
            y_logical = MENU_H + (py - menu_px + sy) / scale
        if x < 0 or x >= content_w or py < 0 or py >= wh:
            return (-1, -1)
        lx = int(round(x / scale))
        ly = int(round(y_logical))
        return (max(0, min(logical_w - 1, lx)), max(0, min(logical_h - 1, ly)))
    x = px + sx
    y = py + sy
    if x < 0 or y < 0 or x >= content_w or y >= content_h:
        return (-1, -1)
    lx = int(round(x / scale))
    ly = int(round(y / scale))
    return (max(0, min(logical_w - 1, lx)), max(0, min(logical_h - 1, ly)))


def window_from_logical(pos, ww: int, wh: int, scroll=(0, 0), footer_h: int | None = None, canvas_h: int | None = None) -> tuple:
    """Canvas pixels → window pixels. Scroll shifts the point toward the origin.

    When the canvas is taller than the client, the menu stays at y=0 and the
    footer stays on the bottom edge. Only the blocks between them scroll.
    """
    scale, _lw, logical_h, _cw, content_h = fit_window(ww, wh)
    footer = FOOTER_H if footer_h is None else max(1, int(footer_h))
    if canvas_h is not None:
        logical_h = max(1, int(canvas_h))
        content_h = max(1, int(round(logical_h * scale)))
    geom = scrollbar_geom(ww, wh, scroll[0], scroll[1], content_h=content_h if canvas_h is not None else None)
    sx, sy = geom["scroll"]
    lx = float(pos[0])
    ly = float(pos[1])
    wx = int(round(lx * scale)) - sx
    bands = _chrome_bands(scale, wh, content_h, footer_h=footer)
    if not bands["pinned"]:
        wy = int(round(ly * scale)) - sy
        return wx, wy
    menu_px = bands["menu_px"]
    footer_px = bands["footer_px"]
    if ly < MENU_H:
        wy = int(round(ly * scale))
    elif ly >= logical_h - footer:
        wy = (wh - footer_px) + int(round((ly - (logical_h - footer)) * scale))
    else:
        wy = menu_px + int(round((ly - MENU_H) * scale)) - sy
    return wx, wy


def _chrome_bands(scale: float, wh: int, content_h: int, footer_h: int | None = None) -> dict:
    """Menu and footer heights in window pixels. ``pinned`` when the canvas is taller."""
    footer = FOOTER_H if footer_h is None else max(1, int(footer_h))
    menu_px = max(1, int(round(MENU_H * scale)))
    footer_px = max(1, int(round(footer * scale)))
    if menu_px + footer_px >= int(wh):
        room = max(2, int(wh))
        menu_px = max(1, int(round(room * MENU_H / float(MENU_H + footer))))
        footer_px = max(1, room - menu_px)
    return {
        "menu_px": menu_px,
        "footer_px": footer_px,
        "footer_h": footer,
        "pinned": int(content_h) > int(wh),
    }


def monitor_layout(w: int, h: int) -> dict:
    """Rects for one logical frame. Width stays 1920; height grows with the window.

    The camera keeps a 16:9 slot. Leftover height goes to the lidar card and
    the right-hand panel. The OS window scales that frame uniformly.
    """
    m = 16
    header = 48
    footer = 56
    controls_h = 44
    estop_h = 64
    controls_y = h - footer - estop_h - 10
    top = header + 8
    robot_h = 44
    robot_y = controls_y - 10 - robot_h
    bottom = robot_y - 10
    span = max(200, bottom - top)
    right_min = 480
    cam_w = int(w * 0.62)
    cam_w = min(cam_w, w - m - 16 - right_min)
    cam_w = max(cam_w, w // 2)
    if cam_w > w - m - 16 - 280:
        cam_w = w - m - 16 - 280
    gap = 10
    lid_h_min = 140
    cam_h = int(cam_w * 9 / 16)
    if cam_h + gap + lid_h_min > span:
        cam_h = max(120, span - gap - lid_h_min)
    cam = (m, top, cam_w, cam_h)
    lid_top = top + cam_h + gap
    lid_w = min(440, max(200, cam_w // 2))
    lid_h = max(100, bottom - lid_top)
    lid = (m, lid_top, lid_w, lid_h)
    rx = m + cam_w + 16
    panel = (rx, top, w - rx - m, max(80, bottom - top))

    def place(items, x, y, height, limit):
        gap_b = 8
        total = sum(ww for _name, ww in items) + gap_b * (len(items) - 1)
        avail = max(1, limit - x)
        scale = min(1.0, avail / float(total))
        out = {}
        cursor = x
        for name, ww in items:
            rw = max(64, int(ww * scale))
            if cursor + rw > limit:
                rw = max(48, limit - cursor)
            out[name] = (cursor, y, rw, height)
            cursor += rw + gap_b
        return out

    header_items = [("brain", 168), ("teacher", 280), ("record", 250)]
    header_total = sum(ww for _name, ww in header_items) + 8 * (len(header_items) - 1)
    header_x = w - 16 - header_total
    buttons = place(header_items, header_x, 8, 32, w - 16)
    estop_w = 240
    estop = (w - m - estop_w, controls_y, estop_w, estop_h)
    buttons.update(
        place(
            [
                ("learn", 220),
                ("auto", 190),
                ("take", 180),
                ("steer", 210),
                ("treat", 190),
            ],
            m,
            controls_y + (estop_h - controls_h),
            controls_h,
            estop[0] - 8,
        )
    )
    buttons["estop"] = estop
    buttons.update(
        {
            "cam": cam,
            "lid": lid,
            "panel": panel,
            "controls_y": controls_y,
            "footer_y": h - footer,
            "lidar_fresh": (lid[0] + lid[2] + 12, lid[1], 240, 32),
            "robot": (m, robot_y, w - 2 * m, robot_h),
        }
    )
    buttons.update(
        place(
            [
                ("stand_up", 140),
                ("stand_down", 120),
                ("recovery", 160),
                ("lidar_reset", 200),
            ],
            m + 8,
            robot_y + 4,
            max(24, robot_h - 8),
            w - m - 8,
        )
    )
    return buttons


def eye_tone(ready: bool, recognized: bool, percent: float) -> str:
    """Grey is quiet, yellow is climbing, green is the latched «УЗНАЮ»."""
    if recognized:
        return "green"
    if ready and float(percent) > 0.0:
        return "yellow"
    return "grey"


def _ui_font(size: int):
    """Light grotesque: Segoe UI Light on Windows, otherwise Inter or DejaVu Sans."""
    import os

    import pygame

    require_main_thread()

    windir = os.environ.get("WINDIR", r"C:\Windows")
    for path in (
        os.path.join(windir, "Fonts", "segoeuil.ttf"),
        os.path.join(windir, "Fonts", "segoeuisl.ttf"),
        "/usr/share/fonts/truetype/macos/Inter-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(path):
            return pygame.font.Font(path, int(size))
    return pygame.font.SysFont("dejavusans", int(size))


def _mix(a, b, t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _surf_from_rgb(rgb: np.ndarray):
    import pygame

    require_main_thread()
    arr = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
    _surf_from_rgb.hold = arr
    return pygame.surfarray.make_surface(arr)


_surf_from_rgb.hold = None


def _text_line_h(font, sample: str = "учить р") -> int:
    """Full line box: font metrics and the rendered ink, including descent."""
    ink = 0
    for text in ("учить р", sample):
        if text:
            ink = max(ink, int(font.render(text, True, LABEL).get_height()))
    metrics = int(font.get_ascent()) + abs(int(font.get_descent()))
    return max(int(font.get_height()), int(font.get_linesize()), metrics, ink, 1)


def _plot(screen, rect, series, y0, y1, color, font, label):
    """Caption above the series. Lines, the frame, and the scale stay under it."""
    import pygame

    require_main_thread()
    label_rect = None
    graph = pygame.Rect(rect)
    if label:
        label_h = _text_line_h(font, label)
        gap = 4
        label_rect = pygame.Rect(rect.x, rect.y, rect.w, min(label_h, rect.h))
        below = rect.h - label_rect.h - gap
        if below >= 8:
            graph = pygame.Rect(rect.x, label_rect.bottom + gap, rect.w, below)
        else:
            graph = pygame.Rect(rect.x, label_rect.bottom, rect.w, max(0, rect.bottom - label_rect.bottom))
        screen.blit(font.render(label, True, PEACH), (label_rect.x + 10, label_rect.y))
    if graph.w < 8 or graph.h < 8:
        return label_rect, graph
    prev = screen.get_clip()
    clip = graph if prev is None else graph.clip(prev)
    if clip.w <= 0 or clip.h <= 0:
        return label_rect, graph
    screen.set_clip(clip)
    try:
        radius = min(12, graph.w // 2, graph.h // 2)
        pygame.draw.rect(screen, INSET, graph, border_radius=radius)
        pygame.draw.rect(screen, LINE, graph, 1, border_radius=radius)
        if y1 <= y0:
            y1 = y0 + 1.0
        span = max(1, graph.h - 16)
        pts = []
        for i, (_x, y) in enumerate(series):
            if len(series) == 1:
                u = 0.5
            else:
                u = i / (len(series) - 1)
            px = graph.x + 8 + int(u * max(0, graph.w - 16))
            v = (float(y) - y0) / (y1 - y0)
            py = graph.bottom - 8 - int(np.clip(v, 0.0, 1.0) * span)
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(screen, color, False, pts, 2)
        elif pts:
            pygame.draw.circle(screen, color, pts[0], 3)
    finally:
        screen.set_clip(prev)
    return label_rect, graph


def _bars(screen, font, origin, values, width, title, color):
    import pygame

    require_main_thread()
    x, y = origin
    screen.blit(font.render(title, True, PEACH), (x, y))
    y += 18
    labels = ("body", "floor", "ratio", "mot", "near", "wide", "close")
    if values is None:
        values = np.zeros(7)
    vals = np.asarray(values, dtype=float).ravel()
    vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
    if vmax < 1e-6:
        vmax = 1.0
    gap = 6
    n = max(1, len(vals))
    cell = max(18, (width - gap * (n - 1)) // n)
    base = y + 64
    for i, value in enumerate(vals[:n]):
        h = int(58 * max(float(value) / vmax, 0.0))
        rx = x + i * (cell + gap)
        pygame.draw.rect(screen, BG_TOP, pygame.Rect(rx, y, cell, 58), border_radius=3)
        pygame.draw.rect(screen, color, pygame.Rect(rx, base - h, cell, max(h, 1)), border_radius=3)
        lab = labels[i] if i < len(labels) else str(i)
        screen.blit(font.render(lab, True, LABEL), (rx, base + 2))
    return base + 18


def _snap(screen, rect):
    """Own the pixels. A subsurface of the live canvas is not a safe cache."""
    import pygame

    require_main_thread()
    snap = pygame.Surface((int(rect.w), int(rect.h)))
    snap.blit(screen, (0, 0), area=rect)
    return snap


def _scheme_rgb(key: str) -> tuple:
    from .ui_theme import ACTIVE

    raw = ACTIVE[key]
    return (int(raw[0]), int(raw[1]), int(raw[2]))


def _paperclip_points(width: int, height: int, scale: int = 4) -> list:
    """Paperclip stroke, same path as the Go2 control icon, in scaled pixels."""
    import math

    big_w = int(width) * scale
    big_h = int(height) * scale

    def arc(cx, cy, radius, start, end, steps=14):
        pts = []
        steps = max(1, int(steps))
        for i in range(steps + 1):
            t = start + (end - start) * i / float(steps)
            pts.append((cx + radius * math.cos(t), cy + radius * math.sin(t)))
        return pts

    path = [(40.0, 34.0), (40.0, 72.0)]
    path += arc(50, 72, 10, math.pi, 0.0)
    path += [(60.0, 22.0)]
    path += arc(45, 22, 15, 0.0, -math.pi)
    path += [(30.0, 78.0)]
    path += arc(50, 78, 20, math.pi, 0.0)
    path += [(70.0, 30.0)]
    angle = math.radians(35)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    unit = (height * 0.74 * scale) / 100.0
    points = []
    for x, y in path:
        dx, dy = x - 50.0, y - 52.0
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        points.append((big_w / 2.0 + rx * unit, big_h / 2.0 + ry * unit))
    return points


def _blit_paperclip(screen, rect, fill, stroke) -> None:
    """Rounded plate and a paperclip, drawn large and smoothed down."""
    import pygame

    scale = 4
    big = pygame.Surface((max(1, rect.w) * scale, max(1, rect.h) * scale), pygame.SRCALPHA)
    pygame.draw.rect(big, fill, big.get_rect(), border_radius=5 * scale)
    pts = [(int(round(x)), int(round(y))) for x, y in _paperclip_points(rect.w, rect.h, scale)]
    if len(pts) >= 2:
        pygame.draw.lines(big, stroke, False, pts, max(1, int(1.7 * scale)))
    screen.blit(pygame.transform.smoothscale(big, (rect.w, rect.h)), rect.topleft)


def _iter_leaves(node):
    from .dock_layout import Leaf, Split

    if isinstance(node, Leaf):
        return [node]
    if isinstance(node, Split):
        return _iter_leaves(node.a) + _iter_leaves(node.b)
    return []


class TrainMonitor:
    def __init__(self, title: str, fullscreen: bool = False):
        import os

        import pygame

        from .win_dpi import clip_window_box, enable_dpi_awareness, work_area

        enable_dpi_awareness()
        pygame.init()
        from .tabnum import reset_fonts

        reset_fonts()
        if tuple_colors():
            flatten_theme_colors()
        if not no_caption():
            pygame.display.set_caption(title)
        self._persist = persist_enabled()
        self._settings = load_settings() if self._persist else default_settings()
        apply_theme(self._settings.get("theme") if self._persist else "light")
        self.pin_on_top = bool(self._settings.get("pin_on_top"))
        self._footer_h = FOOTER_H
        self._footer_lines = []
        self._show_theme_label = True
        self._header_short = False
        self._client_size = (WIN_W, WIN_H)
        self.estop_bar_rect = pygame.Rect(0, 0, 0, 0)
        self.lead_slider_rect = pygame.Rect(0, 0, 0, 0)
        self._lead_drag = False
        self._pin_surface = None
        self._pin_applied = None
        self.dock = tree_from_dict(self._settings.get("layout") if self._persist else None)
        self.menu_open = False
        self.help_id = None
        self.help_scroll = 0
        self._help_rect = None
        self._help_body = None
        self._help_lines = []
        self._menu_hits: list = []
        self._menu_panel = None
        self._menu_marks = {}
        self._menu_captions = {}
        self._menu_label_x = 0
        self._menu_rule = None
        self._menu_switch_rect = None
        self._chrome: list = []
        self._split_drag = None
        self._split_dirty = False
        self._block_drag = None
        self.brain_packet = None
        self._card_cache: dict = {}
        self._journal_cache = None
        self.run_stage_rects = []
        self.run_side_rects = []
        self._brain_panel_cache = None
        self._bg = None
        self._win_bg = None
        self._last_view = None
        self.screen = None
        self.window = None
        self._window_size = None
        self._minimized = False
        self.brain_open = False
        self.go2_open = False
        self.go2_server_open = False
        self.go2_missing = False
        self.fullscreen = bool(fullscreen) or bool(self._persist and self._settings.get("fullscreen"))
        self._maximized = bool(self._persist and self._settings.get("maximized"))
        if self._persist and not self.fullscreen:
            saved_w = self._settings.get("w")
            saved_h = self._settings.get("h")
            if saved_w and saved_h:
                saved_x, saved_y, saved_w, saved_h = clip_window_box(
                    self._settings.get("x"),
                    self._settings.get("y"),
                    saved_w,
                    saved_h,
                    work_area(),
                )
                self._window_size = (int(saved_w), int(saved_h))
                if saved_x is not None and saved_y is not None:
                    os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (int(saved_x), int(saved_y))
        self._open_display()
        self._apply_pin()
        if self._maximized and not self.fullscreen:
            self._try_maximize()
        self._font_need = None
        self._ensure_fonts(1.0)
        self._bg = None
        self._win_bg = None
        self._content = (0, 0, WIN_W, WIN_H)
        self.scroll_x = 0
        self.scroll_y = 0
        self._scroll_drag = None
        self.cam_inner = None
        self.lid_inner = None
        self.journal_rect = None
        self.log_rows = 5
        self.eye_lamps: dict = {}
        self._fixed_rows: list = []
        self._anchors: list = []
        self.clock = pygame.time.Clock()
        import time

        self._booted = time.monotonic()
        self._toast_text = ""
        self._toast_until = 0.0
        self._apply_layout()
        self.show_stand = False
        self.beep_on = False
        self.flash_open = False
        self.audio_ok: bool | None = None
        self._beep_sound = None
        self._prev_rec = False

    def _rect(self, name: str):
        import pygame

        x, y, w, h = self.layout[name]
        return pygame.Rect(int(x), int(y), int(w), int(h))

    def _zero(self):
        import pygame

        return pygame.Rect(0, 0, 0, 0)

    def _apply_layout(self, height: int = WIN_H) -> None:
        import pygame

        footer_h = int(getattr(self, "_footer_h", FOOTER_H))
        height = max(MENU_H + footer_h + 96, int(height))
        footer_y = height - footer_h
        area_h = max(96, footer_y - MENU_H)
        placed = layout_dock(self.dock, 12, MENU_H, WIN_W - 24, area_h)
        self._dock_blocks = placed["blocks"]
        self._dock_splitters = placed["splitters"]
        self._dock_leaves = {leaf.block: leaf for leaf in _iter_leaves(self.dock)}

        def block(name: str):
            rect = self._dock_blocks.get(name)
            if rect is None:
                return self._zero()
            return pygame.Rect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))

        self.cam_rect = block("camera")
        self.lid_rect = block("lidar")
        self.panel_rect = block("brain")
        self.journal_rect = block("journal")
        controls = block("controls")
        self.controls_rect = controls

        def row(items, x, y, row_h, limit):
            """Place buttons at their content width. The caption-to-key gap is not scaled."""
            gap_b = 8
            out = {}
            cursor = x
            for name, ww in items:
                rw = max(1, int(ww))
                if cursor >= limit:
                    out[name] = self._zero()
                    continue
                out[name] = pygame.Rect(cursor, y, rw, row_h)
                cursor += rw + gap_b
            return out

        leaf = self._dock_leaves.get("controls")
        inner = None
        if controls.w > 8 and leaf is not None and leaf.visible and not leaf.collapsed:
            got = dock_content((controls.x, controls.y, controls.w, controls.h), False)
            if got is not None:
                inner = pygame.Rect(*got)
        if inner is not None and inner.h >= 24:
            row_h = min(44, inner.h)
            estop_h = min(inner.h, max(row_h + 16, CONTROLS_BUTTON_H))
            items = [
                ("learn", self._keyed_button_width(("СТОП ОБУЧЕНИЯ", "СТАРТ ОБУЧЕНИЯ"), "P", True)),
                ("auto", self._keyed_button_width(("СТОП АВТО", "СТАРТ АВТО"), "A", True)),
                ("take", self._keyed_button_width(("ПЕРЕХВАТ",), "M", True)),
                ("steer", self._keyed_button_width(("БИЛАТЕРАЛЬНО", "СЕКТОРЫ"), "K", False)),
                ("treat", self._keyed_button_width(("ЛАКОМСТВО",), "T", True)),
            ]
            gap_b = 8
            total = sum(ww for _name, ww in items) + gap_b * (len(items) - 1)
            estop_floor = self._estop_min_width()
            preferred = min(260, max(estop_floor, inner.w // 4))
            spare = inner.w - gap_b - total
            if spare >= preferred:
                estop_w = preferred
            elif spare >= estop_floor:
                estop_w = spare
            else:
                estop_w = estop_floor
            estop_rect = pygame.Rect(inner.right - estop_w, inner.y, estop_w, estop_h)
            placed_row = row(items, inner.x, inner.y, row_h, estop_rect.x - gap_b)
            placed_row["estop"] = estop_rect
        else:
            placed_row = {name: self._zero() for name in ("learn", "auto", "take", "steer", "treat", "estop")}
        self.learn_rect = placed_row["learn"]
        self.auto_rect = placed_row["auto"]
        self.take_rect = placed_row["take"]
        self.steer_rect = placed_row["steer"]
        self.treat_rect = placed_row["treat"]
        self.estop_rect = placed_row["estop"]

        robot = block("robot")
        self.robot_rect = robot
        robot_leaf = self._dock_leaves.get("robot")
        robot_inner = None
        if robot.w > 8 and robot_leaf is not None and robot_leaf.visible and not robot_leaf.collapsed:
            got = dock_content((robot.x, robot.y, robot.w, robot.h), False)
            if got is not None:
                robot_inner = pygame.Rect(*got)
        if robot_inner is not None and robot_inner.w >= 40 and robot_inner.h >= 20:
            self._place_robot_row(robot_inner)
        else:
            self.stand_up_rect = self._zero()
            self.stand_down_rect = self._zero()
            self.recovery_rect = self._zero()
            self.lidar_reset_rect = self._zero()

        lid_leaf = self._dock_leaves.get("lidar")
        lid_inner = None
        if self.lid_rect.w > 8 and lid_leaf is not None and lid_leaf.visible and not lid_leaf.collapsed:
            got = dock_content((self.lid_rect.x, self.lid_rect.y, self.lid_rect.w, self.lid_rect.h), False)
            if got is not None:
                lid_inner = pygame.Rect(*got)
        if lid_inner is not None:
            self._place_lidar_column(lid_inner)
        else:
            self.lidar_plot_rect = self._zero()
            self.lidar_column_rect = self._zero()
            self.lidar_fresh_rect = self._zero()
        self._place_header()
        self.layout = {
            "footer_y": footer_y,
            "controls_y": controls.y,
            "cam": (self.cam_rect.x, self.cam_rect.y, self.cam_rect.w, self.cam_rect.h),
            "lid": (self.lid_rect.x, self.lid_rect.y, self.lid_rect.w, self.lid_rect.h),
            "panel": (self.panel_rect.x, self.panel_rect.y, self.panel_rect.w, self.panel_rect.h),
        }

    def _try_maximize(self) -> None:
        win = borrowed_window()
        if win is None:
            return
        try:
            win.maximize()
        except Exception:
            return

    def _apply_window_box(self, data: dict, borrowed: bool) -> bool:
        """Read x, y, size from one kept Window, or from a throwaway when bisecting."""
        try:
            win = borrowed_window() if borrowed else throwaway_window()
            if win is None:
                return False
            pos = win.position
            size = win.size
            data["x"], data["y"] = int(pos[0]), int(pos[1])
            data["w"], data["h"] = int(size[0]), int(size[1])
            data["maximized"] = bool(getattr(win, "maximized", False))
            return True
        except Exception:
            return False

    def _save_ui(self, geometry_only: bool = False) -> None:
        if not self._persist:
            return
        import time

        data = default_settings()
        data.update(self._settings)
        data["theme"] = theme_name()
        data["pin_on_top"] = bool(getattr(self, "pin_on_top", False))
        data["fullscreen"] = bool(self.fullscreen)
        data["maximized"] = bool(getattr(self, "_maximized", False))
        data["layout"] = self.dock.to_dict()
        # Resize must not build a Window. Borrowing the SDL window stores
        # that Python object on the window and does not keep it alive.
        used_window = False
        if not no_sdl_window() and (sdl_window_on_resize() or not geometry_only):
            used_window = self._apply_window_box(data, borrowed=not sdl_window_on_resize())
        if not used_window and self._window_size:
            data["w"], data["h"] = int(self._window_size[0]), int(self._window_size[1])
        self._settings = data
        if geometry_only:
            now = time.monotonic()
            if now - float(getattr(self, "_save_at", -10.0)) < 0.5:
                self._save_pending = data
                return
        self._save_at = time.monotonic()
        self._save_pending = None
        save_settings(data)

    def _flush_save(self) -> None:
        if getattr(self, "_save_pending", None) is None:
            return
        import time

        if time.monotonic() - float(getattr(self, "_save_at", -10.0)) < 0.5:
            return
        self.flush_settings()

    def flush_settings(self) -> None:
        """Write a resize that was held back so a MoveWindow storm stays on the event pump."""
        pending = getattr(self, "_save_pending", None)
        if pending is None or not self._persist:
            return
        import time

        self._save_pending = None
        self._save_at = time.monotonic()
        self._settings = pending
        save_settings(pending)

    def _drop_surface_caches(self) -> None:
        """Forget pictures taken from a canvas or window that is about to go away."""
        self._bg = None
        self._win_bg = None
        self._card_cache = {}
        self._journal_cache = None
        self._brain_panel_cache = None

    def _set_theme(self, name: str) -> None:
        apply_theme(name)
        self._drop_surface_caches()
        self._save_ui()

    def _reset_layout(self) -> None:
        self.dock = default_tree()
        height = self.screen.get_height() if self.screen is not None else WIN_H
        self._apply_layout(height)
        self._save_ui()

    def _leaf(self, name: str):
        return self._dock_leaves.get(name) or find_leaf(self.dock, name)

    def _content_of(self, name: str):
        import pygame

        leaf = self._leaf(name)
        rect = self._dock_blocks.get(name) if hasattr(self, "_dock_blocks") else None
        if leaf is None or rect is None or not leaf.visible or leaf.collapsed:
            return None
        got = dock_content(rect, False)
        if got is None:
            return None
        box = pygame.Rect(*got)
        if box.w < 8 or box.h < 8:
            return None
        return box

    def _desktop(self):
        import pygame

        try:
            sizes = pygame.display.get_desktop_sizes()
        except Exception:
            sizes = []
        if sizes:
            return int(sizes[0][0]), int(sizes[0][1])
        info = pygame.display.Info()
        w = int(getattr(info, "current_w", 0) or 0)
        h = int(getattr(info, "current_h", 0) or 0)
        if w < 320 or h < 240:
            return WIN_W, WIN_H
        return w, h

    def _initial_window_size(self):
        """About 80% of the desktop, inside the screen so the title bar fits."""
        dw, dh = self._desktop()
        ww = max(640, int(round(dw * 0.80)))
        wh = max(480, int(round(dh * 0.80)))
        ww = min(ww, max(320, dw - 16))
        wh = min(wh, max(240, dh - 48))
        return ww, wh

    def _open_display(self) -> None:
        """Bordered resizable window. Drawing uses a logical frame at least 1920×1080.

        Fullscreen is only the FULLSCREEN flag (F11 or --fullscreen). Width
        sets a uniform scale (never below 0.75). Extra height stretches the
        lidar, the brain panel, and the journal. A narrower window scrolls.
        SCALED on Windows opens a borderless frame that cannot be minimized.
        """
        import pygame

        require_main_thread()
        self._drop_surface_caches()
        if self.fullscreen:
            dw, dh = self._desktop()
            try:
                self.window = pygame.display.set_mode((dw, dh), pygame.FULLSCREEN)
            except pygame.error:
                self.fullscreen = False
        if not self.fullscreen:
            if self._window_size is None:
                self._window_size = self._initial_window_size()
            w, h = int(self._window_size[0]), int(self._window_size[1])
            self.window = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            actual = self.window.get_size()
            # The first set_mode after FULLSCREEN can keep the desktop size.
            if actual != (w, h):
                self.window = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                actual = self.window.get_size()
            if actual == (w, h):
                self._window_size = actual
            elif actual[0] >= 64 and actual[1] >= 64 and self._window_size is None:
                self._window_size = (int(actual[0]), int(actual[1]))
        if self.screen is None:
            self.screen = pygame.Surface((WIN_W, WIN_H))
        self._minimized = False
        pygame.event.clear()
        install_cursors()

    def toggle_fullscreen(self) -> bool:
        """F11. Leaving fullscreen restores the window size from before it."""
        self.fullscreen = not self.fullscreen
        self._minimized = False
        self._open_display()
        self._apply_pin()
        self._save_ui()
        return self.fullscreen

    def _logical_pos(self, pos) -> tuple:
        """Window pixels → the logical canvas. Scrollbars do not hit a button."""
        win = self.window
        if win is None:
            return (int(pos[0]), int(pos[1]))
        ww, wh = win.get_size()
        if ww < 2 or wh < 2:
            return (int(pos[0]), int(pos[1]))
        self._clamp_scroll(ww, wh)
        canvas_h = self.screen.get_height() if self.screen is not None else None
        return logical_from_window(
            pos,
            ww,
            wh,
            (self.scroll_x, self.scroll_y),
            footer_h=int(getattr(self, "_footer_h", FOOTER_H)),
            canvas_h=canvas_h,
        )

    def _mouse(self) -> tuple:
        import pygame

        return self._logical_pos(pygame.mouse.get_pos())

    def _on_resize(self, w: int, h: int) -> None:
        import pygame

        require_main_thread()
        if w < 64 or h < 64:
            self._minimized = True
            return
        self._minimized = False
        if self.fullscreen:
            return
        # Windows frees the window surface inside the resize message, and
        # pygame's event watch points the existing Surface at the new pixels.
        # get_surface() picks that object up. A second set_mode frees it
        # again while events still name it. RECOGNIZER_FORCE_SETMODE is the
        # 7518 path that called set_mode on every resize.
        self._bind_display()
        live = self.window
        already = live is not None and live.get_size() == (w, h)
        if force_setmode() or not already:
            self._drop_surface_caches()
            self.window = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            self._bind_display()
            pygame.event.clear()
        self._remember_client(w, h)

    def _remember_client(self, w: int | None = None, h: int | None = None) -> None:
        """Layout and scroll follow the live client, not a stale event size."""
        self._bind_display()
        if self.window is not None:
            got = self.window.get_size()
            if int(got[0]) >= 64 and int(got[1]) >= 64:
                w, h = int(got[0]), int(got[1])
        if w is None or h is None:
            return
        self._minimized = False
        self._window_size = (int(w), int(h))
        self._clamp_scroll(int(w), int(h))
        self._save_ui(geometry_only=True)
        self._apply_pin()

    def _adopt_client(self) -> None:
        """WINDOWMAXIMIZED / WINDOWSIZECHANGED already resized the surface."""
        if self.fullscreen:
            return
        self._remember_client()

    def _sync_maximized_from_size(self) -> None:
        """A restore that still fills the work area is still maximized."""
        from .win_dpi import work_area

        area = work_area()
        if area is None or self.window is None:
            self._maximized = False
        else:
            w, h = self.window.get_size()
            self._maximized = int(w) >= int(area[2]) - 8 and int(h) >= int(area[3]) - 8
        if self._persist:
            self._settings["maximized"] = bool(self._maximized)

    def _clamp_scroll(self, ww: int | None = None, wh: int | None = None) -> None:
        if ww is None or wh is None:
            if self.window is None:
                return
            ww, wh = self.window.get_size()
        ww, wh = int(ww), int(wh)
        max_x, _fit_y = scroll_limits(ww, wh)
        scale = fit_window(ww, wh)[0]
        canvas_h = self.screen.get_height() if self.screen is not None else fit_window(ww, wh)[2]
        max_y = max(0, int(round(canvas_h * scale)) - wh)
        self.scroll_x = max(0, min(int(self.scroll_x), max_x))
        self.scroll_y = max(0, min(int(self.scroll_y), max_y))

    def _scroll_by(self, dx: int, dy: int) -> None:
        self.scroll_x += int(dx)
        self.scroll_y += int(dy)
        self._clamp_scroll()

    def scroll_geometry(self, size=None) -> dict:
        if size is None:
            if self.window is None:
                return scrollbar_geom(WIN_W, WIN_H, 0, 0)
            size = self.window.get_size()
        ww, wh = int(size[0]), int(size[1])
        scale = fit_window(ww, wh)[0]
        canvas_h = self.screen.get_height() if self.screen is not None else None
        content_h = None if canvas_h is None else max(1, int(round(canvas_h * scale)))
        return scrollbar_geom(ww, wh, self.scroll_x, self.scroll_y, content_h=content_h)

    def _bind_display(self) -> None:
        """Point self.window at the surface SDL is using right now."""
        import pygame

        live = pygame.display.get_surface()
        if live is None:
            return
        previous = self.window
        self.window = live
        if previous is not None and previous is not live:
            self._drop_surface_caches()

    def _ensure_fonts(self, scale: float) -> None:
        """Rebuild UI fonts when the window scale changes. Screen size stays ≥ 13 px."""
        from .tabnum import set_min_px

        need = font_px(MIN_FONT_PX, scale)
        if getattr(self, "_font_need", None) == need and self.font is not None:
            return
        self._font_need = need
        set_min_px(need)
        self.font = _ui_font(font_px(15, scale))
        self.font_sm = _ui_font(font_px(14, scale))
        self.font_tiny = _ui_font(font_px(11, scale))
        self.font_big = _ui_font(font_px(26, scale))
        self.font_ind = _ui_font(font_px(18, scale))

    def _sync_canvas(self, ww: int, wh: int) -> None:
        import pygame

        scale, logical_w, fit_h, content_w, _ch = fit_window(ww, wh)
        self._ensure_fonts(scale)
        self._client_size = (int(ww), int(wh))
        wrap_w = max(80, min(WIN_W, int(round(ww / max(scale, 1e-6)))) - 32)
        footer_h, lines = self._measure_footer(wrap_w, scale, content_w > int(ww))
        self._footer_h = footer_h
        self._footer_lines = lines
        use_h = self._canvas_height(fit_h, scale, wh, footer_h)
        if self.screen is None or self.screen.get_size() != (logical_w, use_h):
            self._drop_surface_caches()
            self.screen = pygame.Surface((logical_w, use_h))
        self._apply_layout(use_h)
        self._clamp_scroll(ww, wh)
        self._place_header()

    def _present(self) -> None:
        """Scale the logical canvas from the top-left. A minimized window skips the blit."""
        import pygame

        if self._minimized or self.window is None:
            return
        ww, wh = self.window.get_size()
        if ww < 64 or wh < 64:
            self._minimized = True
            return
        frame = self.present_into((ww, wh))
        self.window.blit(frame, (0, 0))
        pygame.display.flip()

    def present_into(self, size) -> "object":
        """Composite the canvas into ``size``, top-aligned, with scrollbars when needed."""
        import pygame

        require_main_thread()
        ww, wh = int(size[0]), int(size[1])
        _scale, logical_w, logical_h, content_w, content_h = fit_window(ww, wh)
        self._content = (0, 0, content_w, content_h)
        self._clamp_scroll(ww, wh)
        source = self.screen
        if source is None:
            source = pygame.Surface((logical_w, logical_h))
        if no_present():
            out = pygame.Surface((max(1, ww), max(1, wh)))
            out.fill(BG_BOT)
            out.blit(source, (-int(self.scroll_x), -int(self.scroll_y)))
            return out
        sw, sh = source.get_size()
        target_w = content_w
        target_h = max(1, int(round(sh * (content_w / float(max(sw, 1))))))
        if source.get_size() != (target_w, target_h):
            scaled = pygame.transform.smoothscale(source, (target_w, target_h))
        else:
            scaled = source
        footer_h = int(getattr(self, "_footer_h", FOOTER_H))
        bands = _chrome_bands(content_w / float(max(logical_w, 1)), wh, target_h, footer_h=footer_h)
        if bands["pinned"]:
            out = self._pin_chrome(scaled, ww, wh, sh, bands)
        elif (target_w, target_h) == (ww, wh) and self.scroll_x == 0 and self.scroll_y == 0:
            out = scaled
        else:
            out = pygame.Surface((ww, wh))
            out.fill(BG_BOT)
            out.blit(scaled, (-self.scroll_x, -self.scroll_y))
        return self._paint_scrollbars(out, ww, wh)

    def _pin_chrome(self, scaled, ww: int, wh: int, logical_h: int, bands: dict):
        """Menu at the top of the client, keys at the bottom, blocks scroll between."""
        import pygame

        sh = scaled.get_height()
        sw = scaled.get_width()
        menu_px = int(bands["menu_px"])
        footer_px = int(bands["footer_px"])
        footer_h = int(bands.get("footer_h", FOOTER_H))
        menu_src = max(1, int(round(MENU_H * sh / float(max(logical_h, 1)))))
        footer_src = max(1, int(round(footer_h * sh / float(max(logical_h, 1)))))
        if menu_src + footer_src >= sh:
            menu_src = max(1, sh // 5)
            footer_src = max(1, sh - menu_src - 1)
        if menu_px + footer_px > wh:
            menu_px = menu_src if menu_src + footer_src <= wh else max(1, wh // 3)
            footer_px = footer_src if menu_px + footer_src <= wh else max(1, wh - menu_px)
        out = pygame.Surface((ww, wh))
        out.fill(BG_BOT)
        out.blit(scaled, (-self.scroll_x, 0), pygame.Rect(0, 0, sw, menu_src))
        out.blit(
            scaled,
            (-self.scroll_x, wh - footer_px),
            pygame.Rect(0, sh - footer_src, sw, footer_src),
        )
        mid_view = max(1, wh - menu_px - footer_px)
        mid_room = max(1, sh - menu_src - footer_src)
        src_scroll = min(max(0, int(self.scroll_y)), max(0, mid_room - mid_view))
        out.blit(
            scaled,
            (-self.scroll_x, menu_px),
            pygame.Rect(0, menu_src + src_scroll, sw, min(mid_view, mid_room - src_scroll)),
        )
        return out

    def _paint_scrollbars(self, surface, ww: int, wh: int):
        import pygame

        geom = self.scroll_geometry((ww, wh))
        for track, thumb in ((geom["track_x"], geom["thumb_x"]), (geom["track_y"], geom["thumb_y"])):
            if track is None or thumb is None:
                continue
            pygame.draw.rect(surface, SCROLL_TRACK, track)
            pygame.draw.rect(surface, SCROLL_THUMB, thumb, border_radius=4)
        return surface

    def _on_wheel(self, event) -> None:
        import pygame

        step = 48
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        dx = int(getattr(event, "x", 0) or 0)
        dy = int(getattr(event, "y", 0) or 0)
        if self._help_wheel(dy):
            return
        if shift:
            self._scroll_by(-dy * step - dx * step, 0)
        else:
            self._scroll_by(-dx * step, -dy * step)

    def _help_wheel(self, dy: int) -> bool:
        panel = getattr(self, "_help_rect", None)
        if not self.help_id or panel is None:
            return False
        if not panel.collidepoint(self._mouse()):
            return False
        self.help_scroll = max(0, int(self.help_scroll) - int(dy) * 28)
        return True

    def _toggle_help(self, name: str) -> None:
        if self.help_id == name:
            self.help_id = None
            return
        self.help_id = name
        self.help_scroll = 0

    def _chrome_at(self, pos):
        for rect, kind, arg in self._chrome:
            if rect.collidepoint(pos):
                return kind, arg
        return None, None

    def _help_hit(self, pos) -> bool:
        """Toggle or dismiss the one explanation. A click outside still falls through."""
        if getattr(self, "help_bar_rect", None) is not None and self.help_bar_rect.collidepoint(pos):
            self._toggle_help("bar")
            return True
        kind, arg = self._chrome_at(pos)
        if kind == "info":
            self._toggle_help(arg)
            return True
        panel = getattr(self, "_help_rect", None)
        if self.help_id and panel is not None and panel.collidepoint(pos):
            return True
        if self.help_id:
            self.help_id = None
        return False

    def _help_anchor(self):
        if self.help_id == "bar":
            rect = getattr(self, "help_bar_rect", None)
            if rect is not None and rect.w > 0:
                return rect
            return None
        for rect, kind, arg in self._chrome:
            if kind == "info" and arg == self.help_id:
                return rect
        return None

    def _draw_help(self, screen) -> None:
        """One explanation over the block. Long text scrolls inside the panel."""
        import pygame

        from .ui_help import layout_page, page

        if not self.help_id:
            self._help_rect = None
            self._help_body = None
            self._help_lines = []
            return
        try:
            title, _blocks = page(self.help_id)
        except KeyError:
            self.help_id = None
            self._help_rect = None
            self._help_body = None
            self._help_lines = []
            return
        anchor = self._help_anchor()
        if anchor is None:
            self.help_id = None
            self._help_rect = None
            self._help_body = None
            self._help_lines = []
            return
        sw, sh = screen.get_size()
        width = min(440, max(160, sw - 16))
        height = min(520, max(120, sh - 80))
        x = int(anchor.x)
        y = int(anchor.bottom) + 6
        if y + height > sh - 8:
            y = int(anchor.y) - 6 - height
        x = max(8, min(x, sw - 8 - width))
        y = max(8, min(y, sh - 8 - height))
        panel = pygame.Rect(x, y, width, height)
        self._help_rect = panel
        pygame.draw.rect(screen, MENU, panel)
        self._card(screen, panel, radius=10)
        pygame.draw.rect(screen, LINE, panel, 1, border_radius=10)
        title_clip = pygame.Rect(panel.x + 12, panel.y + 4, panel.w - 24, 24)
        previous = screen.get_clip()
        screen.set_clip(title_clip if previous is None else title_clip.clip(previous))
        try:
            label = self.font_sm.render(title, True, PEACH)
            screen.blit(label, (panel.x + 12, panel.y + 6))
        finally:
            screen.set_clip(previous)
        pygame.draw.line(screen, LINE, (panel.x + 10, panel.y + 30), (panel.right - 10, panel.y + 30), 1)
        gutter = 16
        body = pygame.Rect(panel.x + 12, panel.y + 36, panel.w - 24 - gutter, panel.h - 48)
        self._help_body = body
        pygame.draw.rect(screen, INSET, body)
        line_h = int(self.font_sm.get_height()) + 6
        rows, total = layout_page(self.help_id, lambda text: self.font_sm.size(text)[0], body.w - 8, line_h)
        max_scroll = max(0, int(total) - body.h)
        self.help_scroll = max(0, min(int(self.help_scroll), max_scroll))
        self._help_lines = []
        screen.set_clip(body if previous is None else body.clip(previous))
        try:
            for top, kind, line in rows:
                ink_y = body.y + int(top) - int(self.help_scroll)
                if ink_y + line_h <= body.y or ink_y >= body.bottom:
                    continue
                color = PEACH if kind == "h" else LABEL
                glyph = self.font_sm.render(line, True, color)
                screen.blit(glyph, (body.x + 4, ink_y))
                ink = pygame.Rect(body.x + 4, ink_y, glyph.get_width(), glyph.get_height()).clip(body)
                if ink.w > 0 and ink.h > 0:
                    self._help_lines.append(ink)
        finally:
            screen.set_clip(previous)
        if max_scroll <= 0:
            return
        track = pygame.Rect(panel.right - 14, body.y, 6, body.h)
        pygame.draw.rect(screen, SCROLL_TRACK, track, border_radius=3)
        thumb_h = max(18, int(body.h * body.h / float(max(total, 1))))
        thumb_h = min(thumb_h, body.h)
        travel = max(1, body.h - thumb_h)
        thumb_y = body.y + int(round(self.help_scroll / float(max_scroll) * travel))
        pygame.draw.rect(screen, SCROLL_THUMB, pygame.Rect(track.x, thumb_y, 6, thumb_h), border_radius=3)

    def _menu_at(self, pos):
        for rect, action in self._menu_hits:
            if rect.collidepoint(pos):
                return action
        return None

    def _run_menu(self, action) -> None:
        kind = action[0]
        if kind == "locked":
            return
        if kind == "toggle":
            if action[1] == "controls":
                return
            leaf = find_leaf(self.dock, action[1])
            if leaf is not None:
                leaf.visible = not leaf.visible
                self._apply_layout(self.screen.get_height() if self.screen is not None else WIN_H)
                self._save_ui()
        elif kind == "reset":
            self._reset_layout()
            self.menu_open = False
        elif kind == "doggibot":
            self.menu_open = False
        elif kind == "go2":
            self.menu_open = False

    def _drag_splitter(self, win_pos) -> None:
        if not self._split_drag:
            return
        node, axis, bounds = self._split_drag
        lx, ly = self._logical_pos(win_pos)
        if lx < 0:
            return
        x, y, w, h = bounds
        if axis == "x":
            span = max(1, w - DOCK_SPLIT)
            ratio = (lx - x) / float(span)
            node.ratio = min(0.85, max(0.15, ratio))
        else:
            span = max(1, h - DOCK_SPLIT)
            ratio = (ly - y) / float(span)
            node.ratio = min(0.9, max(0.1, ratio))
        node.locked = True
        self._split_dirty = True
        height = self.screen.get_height() if self.screen is not None else WIN_H
        self._apply_layout(height)

    def _finish_block_drag(self, win_pos) -> None:
        name = self._block_drag
        self._block_drag = None
        if not name:
            return
        lx, ly = self._logical_pos(win_pos)
        if lx < 0:
            return
        target = hit_header(self._dock_blocks, self._dock_leaves, lx, ly)
        if target and target != name and swap_blocks(self.dock, name, target):
            self._apply_layout(self.screen.get_height() if self.screen is not None else WIN_H)
            self._save_ui()

    def _on_primary_down(self, win_pos, inp) -> None:
        import pygame

        if self._begin_scroll_drag(win_pos):
            return
        pos = self._logical_pos(win_pos)
        if pos[0] < 0:
            return
        if self._help_hit(pos):
            return
        if self._lead_at(win_pos, inp, drag=False):
            self._lead_drag = True
            return
        approach = getattr(self, "approach_rect", None)
        if approach is not None and approach.w > 8 and approach.collidepoint(pos):
            inp.approach_reset = True
            return
        if self._header_chrome_hit(pos):
            return
        if self.view_rect.collidepoint(pos):
            self.menu_open = not self.menu_open
            return
        if self.doggibot_rect.collidepoint(pos):
            self.menu_open = False
            inp.doggibot = True
            return
        if self.menu_open:
            action = self._menu_at(pos)
            if action is not None:
                self._run_menu(action)
                if action[0] == "doggibot":
                    inp.doggibot = True
                elif action[0] == "go2":
                    inp.go2_control = True
                return
            panel = self._menu_panel
            if panel is not None and panel.collidepoint(pos):
                return
            self.menu_open = False
        kind, arg = self._chrome_at(pos)
        if kind == "collapse":
            if arg == "controls":
                return
            leaf = find_leaf(self.dock, arg)
            if leaf is not None:
                leaf.collapsed = not leaf.collapsed
                self._apply_layout(self.screen.get_height() if self.screen is not None else WIN_H)
                self._save_ui()
            return
        if kind == "close":
            if arg == "controls":
                return
            leaf = find_leaf(self.dock, arg)
            if leaf is not None:
                leaf.visible = False
                self._apply_layout(self.screen.get_height() if self.screen is not None else WIN_H)
                self._save_ui()
            return
        for splitter in self._dock_splitters:
            rect = pygame.Rect(*splitter["rect"])
            if rect.collidepoint(pos):
                self._split_drag = (splitter["node"], splitter["axis"], splitter["bounds"])
                return
        header = hit_header(self._dock_blocks, self._dock_leaves, pos[0], pos[1])
        if header:
            self._block_drag = header
            return
        for rect, name in getattr(self, "run_stage_rects", ()) or ():
            if rect.collidepoint(pos):
                inp.run_stage = name
                return
        for rect, name in getattr(self, "run_side_rects", ()) or ():
            if rect.collidepoint(pos):
                inp.run_side = name
                return
        bar = getattr(self, "estop_bar_rect", None)
        if bar is not None and bar.w >= 8 and bar.collidepoint(pos):
            inp.estop = True
        elif self.estop_rect.collidepoint(pos):
            inp.estop = True
        elif self.treat_rect.collidepoint(pos):
            inp.treat = True
        elif self.lidar_reset_rect.collidepoint(pos):
            inp.lidar_reset = True
        elif self.lidar_fresh_rect.collidepoint(pos):
            inp.lidar_toggle = True
        elif self.learn_rect.collidepoint(pos):
            inp.pause_learn = True
        elif self.auto_rect.collidepoint(pos):
            inp.autonomy_toggle = True
        elif self.take_rect.collidepoint(pos):
            inp.takeover = True
        elif self.steer_rect.collidepoint(pos):
            inp.steer_toggle = True
        elif self.record_rect.collidepoint(pos):
            inp.record_toggle = True
        elif self.teacher_rect.collidepoint(pos):
            inp.teacher_toggle = True
        elif self.go2_rect.collidepoint(pos):
            inp.go2_control = True
        elif self.brain_rect.collidepoint(pos):
            inp.brain_toggle = True
        elif self.show_stand and self.stand_up_rect.collidepoint(pos):
            inp.stand_up = True
        elif self.show_stand and self.stand_down_rect.collidepoint(pos):
            inp.stand_down = True
        elif self.show_stand and self.recovery_rect.collidepoint(pos):
            inp.recovery = True

    def _begin_scroll_drag(self, pos) -> bool:
        geom = self.scroll_geometry()
        px, py = int(pos[0]), int(pos[1])
        if geom["thumb_y"] is not None and geom["thumb_y"].collidepoint(px, py):
            self._scroll_drag = ("y", py - geom["thumb_y"].y)
            return True
        if geom["thumb_x"] is not None and geom["thumb_x"].collidepoint(px, py):
            self._scroll_drag = ("x", px - geom["thumb_x"].x)
            return True
        if geom["track_y"] is not None and geom["track_y"].collidepoint(px, py):
            self._page_scroll("y", py, geom)
            return True
        if geom["track_x"] is not None and geom["track_x"].collidepoint(px, py):
            self._page_scroll("x", px, geom)
            return True
        return False

    def _page_scroll(self, axis: str, coord: int, geom: dict) -> None:
        if self.window is None:
            return
        ww, wh = self.window.get_size()
        if axis == "y" and geom["thumb_y"] is not None:
            page = max(40, int(wh * 0.8))
            self._scroll_by(0, -page if coord < geom["thumb_y"].centery else page)
        elif axis == "x" and geom["thumb_x"] is not None:
            page = max(40, int(ww * 0.8))
            self._scroll_by(-page if coord < geom["thumb_x"].centerx else page, 0)

    def _drag_scroll(self, pos) -> None:
        if not self._scroll_drag or self.window is None:
            return
        axis, grab = self._scroll_drag
        ww, wh = self.window.get_size()
        geom = self.scroll_geometry((ww, wh))
        if axis == "y" and geom["track_y"] is not None and geom["thumb_y"] is not None:
            travel = max(1, geom["track_y"].h - geom["thumb_y"].h)
            thumb_top = max(0, min(int(pos[1]) - grab, travel))
            self.scroll_y = int(round(thumb_top / float(travel) * geom["max_y"])) if geom["max_y"] else 0
        elif axis == "x" and geom["track_x"] is not None and geom["thumb_x"] is not None:
            travel = max(1, geom["track_x"].w - geom["thumb_x"].w)
            thumb_left = max(0, min(int(pos[0]) - grab, travel))
            self.scroll_x = int(round(thumb_left / float(travel) * geom["max_x"])) if geom["max_x"] else 0
        self._clamp_scroll(ww, wh)

    def pump(self) -> MonitorInput:
        import pygame

        require_main_thread()
        self._bind_display()
        self._flush_save()
        inp = MonitorInput()
        pending_size = None
        pending_full = False
        trust_surface = False
        note_restore = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                inp.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.help_id:
                        self.help_id = None
                    elif self.menu_open:
                        self.menu_open = False
                    else:
                        inp.quit = True
                elif event.key == pygame.K_p:
                    inp.pause_learn = True
                elif event.key == pygame.K_r:
                    inp.reset = True
                elif event.key == pygame.K_s:
                    inp.save = True
                elif event.key == pygame.K_l:
                    inp.load = True
                elif event.key == pygame.K_F12:
                    inp.screenshot = True
                elif event.key == pygame.K_d:
                    inp.label = "dog"
                elif event.key == pygame.K_n:
                    inp.label = "none"
                elif event.key == pygame.K_SPACE:
                    inp.stop = True
                elif event.key == pygame.K_e:
                    inp.estop = True
                elif event.key == pygame.K_t:
                    inp.treat = True
                elif event.key == pygame.K_x:
                    inp.punish = True
                elif event.key == pygame.K_b:
                    self.beep_on = not self.beep_on
                    if self.beep_on:
                        self._ensure_audio()
                    inp.beep_toggle = True
                elif event.key == pygame.K_c:
                    inp.lidar_reset = True
                elif event.key == pygame.K_v:
                    inp.lidar_toggle = True
                elif event.key == pygame.K_g and not getattr(event, "repeat", False):
                    self.flash_open = not self.flash_open
                    inp.flash_toggle = True
                elif event.key == pygame.K_a and not getattr(event, "repeat", False):
                    inp.autonomy_toggle = True
                elif event.key == pygame.K_m and not getattr(event, "repeat", False):
                    inp.takeover = True
                elif event.key == pygame.K_k and not getattr(event, "repeat", False):
                    inp.steer_toggle = True
                elif event.key == pygame.K_y and not getattr(event, "repeat", False):
                    inp.teacher_toggle = True
                elif event.key == pygame.K_h and not getattr(event, "repeat", False):
                    inp.boxes_toggle = True
                elif event.key == pygame.K_j and not getattr(event, "repeat", False):
                    inp.brain_toggle = True
                elif event.key == pygame.K_F11 and not getattr(event, "repeat", False):
                    pending_full = True
                elif event.key == pygame.K_u and not getattr(event, "repeat", False):
                    inp.record_toggle = True
                elif event.key in (pygame.K_KP_PLUS,) or getattr(event, "unicode", "") == "+":
                    inp.stand_up = True
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) or getattr(event, "unicode", "") == "-":
                    inp.stand_down = True
            elif event.type == pygame.MOUSEWHEEL:
                self._on_wheel(event)
            elif event.type == pygame.MOUSEMOTION:
                if self._lead_drag:
                    buttons = getattr(event, "buttons", (1, 0, 0))
                    if buttons and buttons[0]:
                        self._lead_at(event.pos, inp, drag=True)
                    else:
                        self._lead_drag = False
                elif self._scroll_drag is not None:
                    buttons = getattr(event, "buttons", (1, 0, 0))
                    if buttons and buttons[0]:
                        self._drag_scroll(event.pos)
                elif self._split_drag is not None:
                    self._drag_splitter(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._lead_drag = False
                self._scroll_drag = None
                self._finish_block_drag(event.pos)
                self._split_drag = None
                if self._split_dirty:
                    self._split_dirty = False
                    self._save_ui()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._on_primary_down(event.pos, inp)
            elif event.type == pygame.VIDEORESIZE:
                pending_size = (int(event.w), int(event.h))
            elif event.type == getattr(pygame, "WINDOWMINIMIZED", -1):
                self._minimized = True
            elif event.type == getattr(pygame, "WINDOWMAXIMIZED", -3):
                self._minimized = False
                self._maximized = True
                if self._persist:
                    self._settings["maximized"] = True
                trust_surface = True
            elif event.type == getattr(pygame, "WINDOWSIZECHANGED", -5):
                trust_surface = True
            elif event.type == getattr(pygame, "WINDOWRESTORED", -2):
                self._minimized = False
                trust_surface = True
                note_restore = True
            elif event.type == getattr(pygame, "WINDOWSHOWN", -4):
                self._minimized = False
        if pending_full:
            self.toggle_fullscreen()
            inp.fullscreen_toggle = True
        elif trust_surface:
            # The surface already has the client size. Do not set_mode.
            self._adopt_client()
            if note_restore:
                self._sync_maximized_from_size()
        elif pending_size is not None:
            self._on_resize(pending_size[0], pending_size[1])
        inp.focused = bool(pygame.key.get_focused())
        if inp.focused:
            # Arrows do not drive. They used to move the dog: up and down
            # were forward and back, left and right were yaw, left positive.
            # Onboard sent that as a manual Move after takeover.
            keys = pygame.key.get_pressed()
            inp.treat_down = bool(keys[pygame.K_t])
            inp.punish_down = bool(keys[pygame.K_x])
        self._sync_cursor()
        return inp

    def _cursor_kind(self) -> str:
        """Which system cursor the pointer is over. A drag keeps its cursor."""
        import pygame

        if self._split_drag is not None:
            axis = self._split_drag[1]
            return "ns" if axis == "y" else "we"
        if self._scroll_drag is not None or self._block_drag:
            return "hand"
        if self.window is None:
            return "arrow"
        pos = pygame.mouse.get_pos()
        logical = self._logical_pos(pos)
        if logical[0] < 0:
            return "hand"
        for splitter in getattr(self, "_dock_splitters", []) or []:
            rect = pygame.Rect(*splitter["rect"])
            if rect.collidepoint(logical):
                return "ns" if splitter["axis"] == "y" else "we"
        blocks = getattr(self, "_dock_blocks", None) or {}
        leaves = getattr(self, "_dock_leaves", None) or {}
        if hit_header(blocks, leaves, logical[0], logical[1]):
            return "hand"
        return "arrow"

    def _sync_cursor(self) -> None:
        apply_cursor(self._cursor_kind())

    def save_screenshot(self, path: str) -> None:
        import pygame

        require_main_thread()
        pygame.image.save(self.screen, path)

    def draw(self, view: MonitorView) -> None:
        import pygame

        require_main_thread()
        self._bind_display()
        self._ensure_pin()
        self._last_view = view
        if self.window is not None and not self._minimized:
            ww, wh = self.window.get_size()
            if ww >= 64 and wh >= 64:
                self._sync_canvas(ww, wh)
        screen = self.screen
        self._fill_bg(screen)
        self.run_stage_rects = []
        self.run_side_rects = []
        self._fixed_rows = []
        self._anchors = []
        self._glyph_rects = []
        self._glyph_group = ""
        self._brain_rows = []
        self._chrome = []
        self._menu_hits = []
        self._menu_panel = None
        self._menu_marks = {}
        self._menu_captions = {}
        self._menu_label_x = 0
        self._menu_rule = None
        self._menu_switch_rect = None
        self.cam_inner = None
        self.lid_inner = None
        self.show_stand = bool(view.onboard)
        self._draw_menu_bar(screen, view)
        self._draw_blocks(screen, view)
        self._draw_splitters(screen)
        footer = int(self.layout["footer_y"])
        pygame.draw.line(screen, LINE, (24, footer), (WIN_W - 24, footer), 1)
        self._draw_footer_lines(screen, footer)
        if view.learner == "mb":
            if view.recognized and not self._prev_rec and self.beep_on:
                self._play_beep()
            self._prev_rec = view.recognized
        else:
            self._prev_rec = False
        if self.menu_open:
            self._draw_menu(screen)
        self._draw_toast(screen)
        self._draw_help(screen)
        self._draw_pin_tip(screen)
        self._present()
        self.clock.tick(20)

    def _draw_footer_lines(self, screen, footer: int) -> None:
        lines = getattr(self, "_footer_lines", None) or []
        if not lines:
            _h, lines = self._measure_footer(WIN_W - 32)
        for item in lines:
            role, font, text, y = item[:4]
            if not text:
                continue
            color = VALUE if role == "command" else LABEL
            screen.blit(font.render(text, True, color), (16, footer + y))

    def _draw_menu_bar(self, screen, view: MonitorView) -> None:
        import pygame

        pygame.draw.rect(screen, MENU, pygame.Rect(0, 0, WIN_W, MENU_H))
        pygame.draw.line(screen, LINE, (0, MENU_H - 1), (WIN_W, MENU_H - 1), 1)
        hot = self.view_rect.collidepoint(self._mouse()) or self.menu_open
        self._button(screen, self.view_rect, "View", None, hot)
        dog_hot = self.doggibot_rect.collidepoint(self._mouse())
        self._button(screen, self.doggibot_rect, "DoggiBOT", None, dog_hot)
        self._mini_glyph(screen, self.help_bar_rect, "i")
        title = self.font.render(view.title, True, LABEL)
        title_x = self.doggibot_rect.right + 12
        screen.blit(title, (title_x, (MENU_H - title.get_height()) // 2))
        self._link_lamp(screen, view, title_x + title.get_width() + 18)
        self._go2_button(screen)
        self._brain_button(screen, view)
        self._teacher_button(screen, view)
        self._record_button(screen, view)
        self._draw_estop_bar(screen)
        self._draw_header_chrome(screen)

    def _view_size(self) -> tuple:
        size = getattr(self, "_client_size", None)
        if size and int(size[0]) >= 64 and int(size[1]) >= 64:
            return int(size[0]), int(size[1])
        if self.window is not None:
            w, h = self.window.get_size()
            if int(w) >= 64 and int(h) >= 64:
                return int(w), int(h)
        return WIN_W, WIN_H

    def _visible_right(self) -> int:
        """Logical x of the last visible pixel on the menu bar."""
        ww, wh = self._view_size()
        scale, _lw, fit_h, content_w, _ch = fit_window(ww, wh)
        canvas_h = self.screen.get_height() if self.screen is not None else fit_h
        vbar = SCROLL_BAR if int(round(canvas_h * scale)) > wh else 0
        right_px = max(1, ww - vbar)
        logical = (int(getattr(self, "scroll_x", 0)) + right_px) / max(float(scale), 1e-6)
        return max(1, min(WIN_W, int(logical)))

    def _canvas_height(self, fit_h: int, scale: float, wh: int, footer_h: int) -> int:
        """Match the client when every block still has room above the footer.

        A shorter client keeps the tall canvas and scrolls, instead of
        covering a button or squeezing a card down to nothing.
        """
        if wh >= int(round(fit_h * scale)) - 1:
            return int(fit_h)
        fill_h = max(1, int(round(wh / max(float(scale), 1e-6))))
        while int(round(fill_h * scale)) < wh:
            fill_h += 1
        fill_h = min(int(fit_h), fill_h)
        if self._dock_fits(fill_h, footer_h):
            return fill_h
        return int(fit_h)

    def _dock_fits(self, height: int, footer_h: int) -> bool:
        """True when every visible card still has a readable content area."""
        area_h = int(height) - int(footer_h) - MENU_H
        if area_h < 96:
            return False
        placed = layout_dock(self.dock, 12, MENU_H, WIN_W - 24, area_h)
        blocks = placed["blocks"]
        for leaf in _iter_leaves(self.dock):
            if not leaf.visible:
                continue
            rect = blocks.get(leaf.block)
            if rect is None or rect[2] < 48:
                return False
            if leaf.collapsed:
                if rect[3] < DOCK_HEADER // 2:
                    return False
                continue
            inner = dock_content(rect, False)
            if inner is None or inner[2] < 48 or inner[3] < 40:
                return False
        cam = blocks.get("camera")
        cam_leaf = next((leaf for leaf in _iter_leaves(self.dock) if leaf.block == "camera"), None)
        if cam is not None and cam_leaf is not None and cam_leaf.visible and not cam_leaf.collapsed:
            pref = DOCK_HEADER + max(80, int(round(cam[2] * 9 / 16)))
            if cam[3] < int(round(pref * 0.80)):
                return False
        lid = blocks.get("lidar")
        lid_leaf = next((leaf for leaf in _iter_leaves(self.dock) if leaf.block == "lidar"), None)
        if lid is not None and lid_leaf is not None and lid_leaf.visible and not lid_leaf.collapsed:
            if lid[3] < LIDAR_MIN_H:
                return False
        return True

    def _wrap_words(self, font, text: str, width: int) -> list:
        words = str(text or "").split()
        if not words:
            return [""]
        lines = []
        cur = ""
        limit = max(8, int(width))
        for word in words:
            trial = word if not cur else cur + " " + word
            if font.size(trial)[0] <= limit or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines or [""]

    def _footer_sources(self) -> tuple:
        from .tabnum import phrase as tab_phrase

        view = getattr(self, "_last_view", None)
        status = ""
        command = "—"
        keys = ""
        focused = True
        if view is not None:
            status = view.udp_status or ""
            command = format_command(view.last_command or "—")
            keys = view.keys_hint or ""
            focused = bool(view.focused)
        focus = "окно в фокусе" if focused else "нажмите на окно — клавиши не читаются"
        line = tab_phrase(status, 78) + "  " + tab_phrase(focus, 42)
        return line, command, keys

    def _measure_footer(self, width: int, scale: float = 1.0, hscroll: bool = False) -> tuple:
        """Status, command, and wrapped keys. Height is reserved before the dock is placed.

        Each line keeps its full box, descent included, and a bottom margin that
        stays inside the client. A horizontal scrollbar covers the last pixels
        of the window, so that margin also clears the bar.
        """
        import math

        from .tabnum import load_font

        mono = load_font(14)
        sm = self.font_sm
        status, command, keys = self._footer_sources()
        groups = (
            (mono, self._wrap_words(mono, status, width)),
            (mono, self._wrap_words(mono, command, width)),
            (sm, self._wrap_words(sm, keys, width)),
        )
        pad = 6
        gap = 4
        screen_pad = 8 + (SCROLL_BAR if hscroll else 0)
        bottom = max(pad, int(math.ceil(screen_pad / max(float(scale), 1e-6))))
        y = pad
        placed = []
        roles = ("status", "command", "keys")
        for role, (font, lines) in zip(roles, groups):
            for line in lines:
                line_h = _text_line_h(font, line or "учить р")
                placed.append((role, font, line, y, line_h))
                y += line_h + gap
        height = max(FOOTER_H, y - gap + bottom)
        return height, placed

    def _place_header(self) -> None:
        """Pack the menu bar into the visible client. Hide the theme label, then shorten captions."""
        import pygame

        if self.font_sm is None:
            return
        limit = self._visible_right()
        margin = 12
        switch_w, switch_h = 36, 20
        pin_w, pin_h = 30, 26
        label_w = self.font_sm.size("Тёмная тема")[0]
        self.view_rect = pygame.Rect(8, 4, 52, 24)
        dog_w = self.font_sm.size("DoggiBOT")[0] + 28
        self.doggibot_rect = pygame.Rect(self.view_rect.right + 8, 4, dog_w, 24)
        title = "recognition training"
        view = getattr(self, "_last_view", None)
        if view is not None and view.title:
            title = view.title
        title_w = max(self.font.size("recognition training")[0], self.font.size(title)[0])
        lamp_w = 14 + self.font_sm.size("борт")[0]
        min_help = self.doggibot_rect.right + 12 + title_w + 18 + lamp_w + 8
        variants = ((True, False), (False, False), (False, True))
        chosen = variants[-1]
        placed = None
        for show_label, short in variants:
            items = self._header_items(short)
            total = sum(ww for _name, ww in items) + 8 * (len(items) - 1)
            switch_right = limit - margin
            switch = pygame.Rect(switch_right - switch_w, (MENU_H - switch_h) // 2, switch_w, switch_h)
            if show_label:
                label = pygame.Rect(switch.x - 8 - label_w, 0, label_w, MENU_H)
                pin = pygame.Rect(label.x - 14 - pin_w, (MENU_H - pin_h) // 2, pin_w, pin_h)
            else:
                label = pygame.Rect(switch.x, 0, 0, 0)
                pin = pygame.Rect(switch.x - 8 - pin_w, (MENU_H - pin_h) // 2, pin_w, pin_h)
            show_bar = self._estop_needs_bar()
            if show_bar:
                bar_w = self._estop_bar_width()
                bar = pygame.Rect(pin.x - 8 - bar_w, 4, bar_w, 24)
                header_limit = bar.x - 8
            else:
                bar = pygame.Rect(pin.x, 0, 0, 0)
                header_limit = pin.x - 8
            header_x = max(min_help + 36, header_limit - total)
            buttons = self._header_row(items, header_x, header_limit)
            help_rect = pygame.Rect(buttons["go2"].x - 36, 4, 28, 24)
            stop = bar.x - 4 if bar.w >= 8 else pin.x - 4
            fits = (
                switch.right <= limit - 4
                and pin.x >= help_rect.right + 4
                and buttons["record"].right <= stop
                and (bar.w < 8 or bar.x >= buttons["record"].right + 4)
                and help_rect.x >= min_help
            )
            placed = (pin, label, switch, buttons, help_rect, bar)
            chosen = (show_label, short)
            if fits:
                break
        pin, label, switch, buttons, help_rect, bar = placed
        self.pin_rect = pin
        self.theme_label_rect = label
        self.theme_switch_rect = switch
        self.go2_rect = buttons["go2"]
        self.brain_rect = buttons["brain"]
        self.teacher_rect = buttons["teacher"]
        self.record_rect = buttons["record"]
        self.help_bar_rect = help_rect
        self.estop_bar_rect = bar
        self._show_theme_label = bool(chosen[0])
        self._header_short = bool(chosen[1])

    def _estop_bar_width(self) -> int:
        return self.font_sm.size("E-STOP")[0] + 64

    def _estop_needs_bar(self) -> bool:
        """E-STOP stays on the buttons card. That card cannot be closed or folded."""
        return False

    def _draw_estop_bar(self, screen) -> None:
        rect = getattr(self, "estop_bar_rect", None)
        if rect is None or rect.w < 8:
            return
        self._button(
            screen,
            rect,
            "E-STOP",
            None,
            rect.collidepoint(self._mouse()),
            label_color=ESTOP_LABEL,
            fill=ESTOP,
            key="E",
        )

    def _header_items(self, short: bool) -> list:
        if not short:
            return [
                ("go2", self._go2_button_width(False)),
                ("brain", 168),
                ("teacher", 280),
                ("record", 250),
            ]
        return [
            ("go2", self._go2_button_width(True)),
            ("brain", self._caption_slot("МОЗГ", lamp=True, key=True)),
            ("teacher", self._caption_slot("УЧ", lamp=True, key=True)),
            ("record", self._caption_slot("ЗАП", lamp=True, key=True)),
        ]

    def _caption_slot(self, text: str, lamp: bool, key: bool) -> int:
        pad = 36 if lamp else 16
        badge = 28 if key else 8
        return max(72, self.font_sm.size(text)[0] + pad + badge)

    def _header_row(self, items, x: int, limit: int) -> dict:
        import pygame

        gap_b = 8
        total = sum(ww for _name, ww in items) + gap_b * (len(items) - 1)
        avail = max(1, limit - x)
        scale = min(1.0, avail / float(max(total, 1)))
        out = {}
        cursor = x
        for name, ww in items:
            rw = max(48, int(ww * scale))
            if cursor + rw > limit:
                rw = max(32, limit - cursor)
            out[name] = pygame.Rect(cursor, 4, max(1, rw), 24)
            cursor += rw + gap_b
        return out

    def _place_header_chrome(self):
        """Paperclip, then «Тёмная тема», then the pill, packed to the right edge."""
        self._place_header()
        return self.pin_rect, self.theme_label_rect, self.theme_switch_rect

    def _header_chrome_hit(self, pos) -> bool:
        pin = getattr(self, "pin_rect", None)
        if pin is not None and pin.collidepoint(pos):
            self._toggle_pin()
            return True
        label = getattr(self, "theme_label_rect", None)
        switch = getattr(self, "theme_switch_rect", None)
        on_label = label is not None and label.collidepoint(pos)
        on_switch = (
            switch is not None
            and switch.x <= int(pos[0]) < switch.right
            and 0 <= int(pos[1]) < MENU_H
        )
        if on_label or on_switch:
            self._toggle_theme()
            return True
        return False

    def _toggle_theme(self) -> None:
        self._set_theme("light" if theme_name() == "dark" else "dark")

    def _toggle_pin(self) -> None:
        self.pin_on_top = not bool(self.pin_on_top)
        self._apply_pin()
        self._save_ui()

    def _apply_pin(self) -> bool:
        """Native topmost on Windows. Other systems no-op. Size and position stay."""
        from .window_pin import set_always_on_top

        ok = set_always_on_top(bool(getattr(self, "pin_on_top", False)))
        self._pin_surface = id(self.window) if self.window is not None else 0
        self._pin_applied = bool(self.pin_on_top)
        return ok

    def _ensure_pin(self) -> None:
        """Re-apply after SDL replaces the window, without a set_mode of our own."""
        token = id(self.window) if self.window is not None else 0
        if token == getattr(self, "_pin_surface", None) and getattr(self, "_pin_applied", None) == bool(self.pin_on_top):
            return
        self._apply_pin()

    def _draw_header_chrome(self, screen) -> None:
        if getattr(self, "_show_theme_label", True) and self.theme_label_rect.w > 0:
            label = self.font_sm.render("Тёмная тема", True, INK_DIM)
            text_y = (MENU_H - label.get_height()) // 2
            screen.blit(label, (self.theme_label_rect.x, text_y))
        self._draw_theme_switch(screen, self.theme_switch_rect)
        hover = self.pin_rect.collidepoint(self._mouse())
        fill, stroke = self._pin_colors(hover)
        _blit_paperclip(screen, self.pin_rect, fill, stroke)

    def _pin_colors(self, hover: bool) -> tuple:
        if self.pin_on_top:
            return (PEACH_SOFT if hover else PEACH), _scheme_rgb("knob")
        plate = "pin_plate_hot" if hover else "pin_plate"
        return _scheme_rgb(plate), tuple(int(c) for c in INK_DIM)

    def _draw_theme_switch(self, screen, rect) -> None:
        """Pill track and a round knob. Knob on the right when the theme is dark."""
        import pygame

        on = theme_name() == "dark"
        fill = PEACH if on else _scheme_rgb("switch_off")
        radius = max(1, rect.h // 2)
        pygame.draw.rect(screen, fill, rect, border_radius=radius)
        knob_r = 7
        cx = rect.x + (rect.w - radius if on else radius)
        cy = rect.y + radius
        pygame.draw.circle(screen, _scheme_rgb("knob"), (int(cx), int(cy)), knob_r)

    def _draw_pin_tip(self, screen) -> None:
        import pygame

        from .window_pin import PIN_TIP

        self._pin_tip_rect = None
        rect = getattr(self, "pin_rect", None)
        if rect is None or rect.w < 8 or not rect.collidepoint(self._mouse()):
            return
        text = self.font_tiny.render(PIN_TIP, True, _scheme_rgb("pin_tip_ink"))
        pad_x, pad_y = 8, 3
        box = pygame.Rect(0, 0, text.get_width() + pad_x * 2, text.get_height() + pad_y * 2)
        box.top = rect.bottom + 4
        box.right = min(WIN_W - 4, max(rect.right, box.w + 4))
        if box.left < 4:
            box.left = 4
        pygame.draw.rect(screen, _scheme_rgb("pin_tip"), box, border_radius=4)
        screen.blit(text, (box.x + pad_x, box.y + pad_y))
        self._pin_tip_rect = box

    def _draw_blocks(self, screen, view: MonitorView) -> None:
        titles = dict(BLOCK_TITLES)
        if view.lidar_mode:
            titles["lidar"] = view.lidar_mode
        if getattr(view, "brain_offline", False):
            from .brain_panel import brain_header

            titles["brain"] = brain_header(True)
        import pygame

        for name in BLOCKS:
            rect = self._dock_blocks.get(name) if hasattr(self, "_dock_blocks") else None
            leaf = self._leaf(name)
            if rect is None or leaf is None or not leaf.visible:
                continue
            box = pygame.Rect(*rect)
            self._draw_block_chrome(screen, box, titles.get(name, name), name, leaf.collapsed)
            if leaf.collapsed:
                continue
            content = self._content_of(name)
            if content is None:
                continue
            if name == "camera":
                self._draw_clipped(screen, content, "camera", lambda: self._draw_camera_block(screen, view, content))
            elif name == "lidar":
                self._draw_clipped(screen, content, "lidar", lambda: self._draw_lidar_block(screen, view, content))
            elif name == "brain":
                self._draw_clipped(screen, content, "brain", lambda: self._draw_brain_block(screen, view, content))
            elif name == "journal":
                self._draw_clipped(screen, content, "journal", lambda: self._draw_journal_block(screen, view, content))
            elif name == "controls":
                self._draw_clipped(screen, content, "controls", lambda: self._draw_controls_block(screen, view))
            elif name == "robot":
                self._draw_clipped(screen, content, "robot", lambda: self._draw_robot_block(screen))

    def _draw_block_chrome(self, screen, rect, title: str, name: str, collapsed: bool) -> None:
        import pygame

        self._card(screen, rect, radius=12)
        # The buttons card carries E-STOP, so it has no collapse or close.
        pinned = name == "controls"
        bar = header_controls((rect.x, rect.y, rect.w, rect.h), pinned=pinned)
        pygame.draw.rect(screen, MENU, pygame.Rect(*bar["bar"]))
        pygame.draw.line(screen, LINE, (rect.x + 8, rect.y + DOCK_HEADER - 1), (rect.right - 8, rect.y + DOCK_HEADER - 1), 1)
        info = pygame.Rect(*bar["info"])
        label = self.font_sm.render(title, True, LABEL)
        title_clip = pygame.Rect(rect.x + 12, rect.y, max(1, info.x - rect.x - 16), min(DOCK_HEADER, rect.h))
        previous = screen.get_clip()
        screen.set_clip(title_clip if previous is None else title_clip.clip(previous))
        try:
            screen.blit(label, (rect.x + 12, rect.y + 4))
        finally:
            screen.set_clip(previous)
        self._chrome.append((info, "info", name))
        self._mini_glyph(screen, info, "i")
        if pinned:
            return
        collapse = pygame.Rect(*bar["collapse"])
        close = pygame.Rect(*bar["close"])
        self._chrome.append((collapse, "collapse", name))
        self._chrome.append((close, "close", name))
        self._mini_glyph(screen, collapse, "–" if not collapsed else "+")
        self._mini_glyph(screen, close, "×")

    def _mini_glyph(self, screen, rect, text: str) -> None:
        import pygame

        pygame.draw.rect(screen, CARD, rect, border_radius=4)
        pygame.draw.rect(screen, LINE, rect, 1, border_radius=4)
        label = self.font_tiny.render(text, True, LABEL)
        screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))

    def _draw_splitters(self, screen) -> None:
        import pygame

        for splitter in getattr(self, "_dock_splitters", []):
            rect = pygame.Rect(*splitter["rect"])
            pygame.draw.rect(screen, LINE, rect)
            if splitter["axis"] == "y":
                mid = rect.centery
                pygame.draw.line(screen, INK_DIM, (rect.centerx - 18, mid), (rect.centerx + 18, mid), 2)
            else:
                mid = rect.centerx
                pygame.draw.line(screen, INK_DIM, (mid, rect.centery - 18), (mid, rect.centery + 18), 2)

    def _blit_menu_check(self, screen, rect, color) -> None:
        """Smoothed check, about 2px. Drawn large, then scaled into a fixed slot."""
        import pygame

        scale = 4
        big = pygame.Surface((max(1, rect.w) * scale, max(1, rect.h) * scale), pygame.SRCALPHA)
        ink = tuple(int(c) for c in color)

        def pt(x, y):
            return (int(round(x * rect.w / 14.0 * scale)), int(round(y * rect.h / 14.0 * scale)))

        pts = [pt(2.2, 7.2), pt(5.6, 11.0), pt(12.2, 2.8)]
        width = max(1, int(round(2.0 * scale)))
        pygame.draw.lines(big, ink, False, pts, width)
        cap = max(1, width // 2)
        for point in pts:
            pygame.draw.circle(big, ink, point, cap)
        screen.blit(pygame.transform.smoothscale(big, (rect.w, rect.h)), rect.topleft)

    def _menu_check_ink(self, locked: bool) -> tuple:
        """Accent peach. The locked E-STOP row uses the muted theme gray."""
        if locked:
            return self._menu_locked_ink()
        return tuple(int(c) for c in PEACH)

    def _menu_locked_ink(self) -> tuple:
        """Theme gray, pulled toward the menu so the row reads quieter than a label."""
        ink = _scheme_rgb("ink_dim")
        plate = _scheme_rgb("menu")
        mix = 0.45
        return tuple(int(round(ink[channel] + (plate[channel] - ink[channel]) * mix)) for channel in range(3))

    def _fill_menu_row(self, screen, panel, row) -> None:
        """Pale hover across the whole row, clipped to the rounded card."""
        import pygame

        color = _scheme_rgb("pin_plate_hot")
        band = pygame.Surface((panel.w, panel.h), pygame.SRCALPHA)
        local = pygame.Rect(0, row.y - panel.y, panel.w, row.h)
        pygame.draw.rect(band, (int(color[0]), int(color[1]), int(color[2]), 255), local)
        mask = pygame.Surface((panel.w, panel.h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=10)
        band.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(band, panel.topleft)

    def _draw_menu(self, screen) -> None:
        import pygame

        titles = {
            "camera": "Камера",
            "lidar": "Лидар",
            "brain": "Панель мозга",
            "journal": "Журнал",
            "robot": "Робот",
            "controls": "Кнопки (E-STOP)",
        }
        blocks = []
        for name in ("camera", "lidar", "brain", "journal", "robot", "controls"):
            if name == "controls":
                blocks.append((("locked", name), titles[name], "locked", name))
            else:
                blocks.append((("toggle", name), titles[name], "check", name))
        actions = [
            (("reset",), "Сброс раскладки", "button", None),
        ]
        rows = blocks + actions
        row_h = 28
        rule_h = 9
        mark = 16
        gap = 8
        pad_l = 12
        pad_r = 12
        note = "всегда виден"
        note_gap = 12
        note_w = self.font_tiny.size(note)[0]
        text_w = max(self.font_sm.size(label)[0] for _action, label, _kind, _arg in rows)
        locked_w = self.font_sm.size(titles["controls"])[0]
        width = max(
            pad_l + mark + gap + text_w + pad_r,
            pad_l + mark + gap + locked_w + note_gap + note_w + pad_r,
            220,
        )
        x = self.view_rect.x
        y = self.view_rect.bottom + 4
        panel = pygame.Rect(x, y, width, 8 + row_h * len(rows) + rule_h)
        self._menu_panel = panel
        pygame.draw.rect(screen, MENU, panel, border_radius=10)
        self._card(screen, panel, radius=10)
        self._menu_hits = []
        self._menu_marks = {}
        self._menu_captions = {}
        self._menu_switch_rect = None
        self._menu_aside = None
        self._menu_aside_text = ""
        label_x = panel.x + pad_l + mark + gap
        self._menu_label_x = label_x
        pointer = self._mouse()
        cursor = panel.y + 4

        def paint(action, label, kind, arg) -> None:
            nonlocal cursor
            row = pygame.Rect(panel.x, cursor, panel.w, row_h)
            self._menu_hits.append((row, action))
            self._menu_captions[action] = label
            slot = pygame.Rect(panel.x + pad_l, row.centery - mark // 2, mark, mark)
            self._menu_marks[action] = slot
            hot = kind != "locked" and row.collidepoint(pointer)
            if hot:
                self._fill_menu_row(screen, panel, row)
            caption = LABEL
            if kind == "check":
                leaf = self._leaf(arg)
                if leaf is not None and leaf.visible:
                    self._blit_menu_check(screen, slot, self._menu_check_ink(False))
                else:
                    caption = INK_DIM
            elif kind == "locked":
                caption = self._menu_locked_ink()
                self._blit_menu_check(screen, slot, caption)
            glyph = self.font_sm.render(label, True, caption)
            screen.blit(glyph, (label_x, row.centery - glyph.get_height() // 2))
            if kind == "locked":
                badge = self.font_tiny.render(note, True, caption)
                nx = row.right - pad_r - badge.get_width()
                ny = row.centery - badge.get_height() // 2
                screen.blit(badge, (nx, ny))
                self._menu_aside = pygame.Rect(nx, ny, badge.get_width(), badge.get_height())
                self._menu_aside_text = note
            cursor += row_h

        for action, label, kind, arg in blocks:
            paint(action, label, kind, arg)
        rule_y = cursor + rule_h // 2
        self._menu_rule = pygame.Rect(panel.x + 12, rule_y, panel.w - 24, 1)
        pygame.draw.line(screen, LINE, (panel.x + 12, rule_y), (panel.right - 12, rule_y), 1)
        cursor += rule_h
        for action, label, kind, arg in actions:
            paint(action, label, kind, arg)

    def _draw_camera_block(self, screen, view: MonitorView, content) -> None:
        cam_inner = self._frame(screen, content, view.camera, "", view.sensor_error, chrome=False)
        self.cam_inner = cam_inner
        if cam_inner is None:
            return
        if view.learner == "mb":
            self._camera_overlap(screen, cam_inner, view)
            self._draw_det_boxes(screen, cam_inner, view.teacher_boxes)
            self._eye_plaques(screen, cam_inner, view)
        else:
            self._camera_halves(screen, cam_inner, view)

    def _fresh_button_width(self) -> int:
        from .tabnum import SLOT_GAP, cell_px

        return (
            32
            + self.font_sm.size("свежий лидар")[0]
            + SLOT_GAP
            + 5 * cell_px(14)
            + SLOT_GAP
            + self.font_sm.size("с")[0]
            + SLOT_GAP
            + self.font_sm.size("V")[0]
            + 14
        )

    def _lidar_table_width(self) -> int:
        from .tabnum import cell_px

        labels = ("муха L", "муха R", "Δ", "z", "вектор", "билатерально", "секторы", "дальн", "вперёд", "сектор")
        label_w = max(self.font_sm.size(name)[0] for name in labels)
        return label_w + 12 + 9 * cell_px(14) + 10 + self.font_sm.size("8/8 подход")[0]

    def _place_lidar_column(self, lid_inner) -> None:
        """Square picture on the left. Buttons and the metric table stay on the right."""
        import pygame

        fresh_w = self._fresh_button_width()
        table_w = self._lidar_table_width()
        gap = 8
        column_w = min(max(table_w, fresh_w, 160), max(0, lid_inner.w - gap))
        plot_side = min(lid_inner.h, max(0, lid_inner.w - gap - column_w))
        if plot_side < 48 and lid_inner.w > 96 and lid_inner.h >= 48:
            plot_side = min(lid_inner.h, max(48, lid_inner.w // 3))
            column_w = max(0, lid_inner.w - gap - plot_side)
        col_x = lid_inner.right - column_w
        if lid_inner.x + plot_side + gap > col_x:
            plot_side = max(0, col_x - gap - lid_inner.x)
        plot = pygame.Rect(lid_inner.x, lid_inner.y, plot_side, plot_side)
        column = pygame.Rect(col_x, lid_inner.y, max(0, lid_inner.right - col_x), lid_inner.h)
        self.lidar_plot_rect = plot
        self.lidar_column_rect = column
        button_h = 28
        self.lidar_fresh_rect = pygame.Rect(column.x, column.y, min(fresh_w, max(0, column.w)), button_h)

    def _ui_scale(self) -> float:
        size = getattr(self, "_client_size", None)
        if size and int(size[0]) >= 64 and int(size[1]) >= 64:
            return float(fit_window(int(size[0]), int(size[1]))[0])
        return 1.0

    def _badge_gap(self) -> int:
        """Logical pixels that stay at least 6px on screen between a caption and its key."""
        import math

        scale = max(self._ui_scale(), 1e-6)
        return max(6, int(math.ceil(6.0 / scale)))

    def _key_badge_width(self, key: str) -> int:
        from .tabnum import load_font

        glyph_w = load_font(13).size(str(key))[0]
        return max(16, glyph_w + 8)

    def _keyed_button_width(self, labels, key: str, lamp: bool) -> int:
        """Width that keeps the key plaque 6px clear of the longer caption."""
        if self.font_sm is None:
            return 180
        label_w = max(self.font_sm.size(text)[0] for text in labels)
        left = 32 if lamp else 12
        if not key:
            return left + label_w + 12
        return left + label_w + self._badge_gap() + self._key_badge_width(key) + 6

    def _estop_min_width(self) -> int:
        """E-STOP stays at least 160 wide, and its centered caption clears the key."""
        if self.font_sm is None:
            return 160
        label = self.font_sm.size("E-STOP")[0]
        badge = self._key_badge_width("E")
        gap = self._badge_gap()
        return max(160, label + 2 * badge + 12 + 2 * gap)

    def _place_robot_row(self, inner) -> None:
        """Stand, lie, recovery, and lidar reset share one row of this card."""
        import pygame

        specs = (
            ("stand_up", ("ВСТАТЬ",), "+", False),
            ("stand_down", ("ЛЕЧЬ",), "-", False),
            ("recovery", ("ПОДЪЁМ",), "", False),
            ("lidar_reset", ("СБРОС ЛИДАРА",), "C", False),
        )
        gap = 8
        widths = [self._keyed_button_width(labels, key, lamp) for _name, labels, key, lamp in specs]
        # The card is the E-STOP row. These buttons use that content height,
        # so the face under the row is not left empty.
        row_h = max(1, min(inner.h, CONTROLS_BUTTON_H))
        y = inner.y
        x = inner.x
        rects = {}
        for (name, _labels, _key, _lamp), width in zip(specs, widths):
            rw = max(1, int(width))
            if x >= inner.right:
                rects[name] = self._zero()
                continue
            rects[name] = pygame.Rect(x, y, rw, row_h)
            x += rw + gap
        self.stand_up_rect = rects["stand_up"]
        self.stand_down_rect = rects["stand_down"]
        self.recovery_rect = rects["recovery"]
        self.lidar_reset_rect = rects["lidar_reset"]

    def _draw_lidar_block(self, screen, view: MonitorView, content) -> None:
        import pygame

        plot = getattr(self, "lidar_plot_rect", None) or content
        lid_msg = "" if view.lidar is not None else view.lidar_hold
        inner = self._frame(screen, plot, view.lidar, "", lid_msg, chrome=False)
        self.lid_inner = inner
        if inner is not None and view.marks:
            self._draw_marks(screen, inner, view.marks)
        self._lidar_fresh_button(screen, view)
        column = getattr(self, "lidar_column_rect", None)
        if column is None or column.w < 8:
            return
        top = self.lidar_fresh_rect.bottom + 6
        top = self._lidar_learn_note(screen, column, top, content.bottom)
        if view.lidar_warning and top + 18 <= content.bottom:
            warn = self.font_sm.render(view.lidar_warning[:42], True, AMBER)
            if column.x + warn.get_width() <= content.right:
                screen.blit(warn, (column.x, top))
                top += warn.get_height() + 4
        self._lidar_table(screen, view, column, top, content.bottom)

    def _lidar_learn_note(self, screen, column, top: int, bottom: int) -> int:
        """Grey line: lidar is shown, and it does not teach. The table stays below."""
        import pygame

        glyph = self.font_tiny.render(LIDAR_LEARN_NOTE, True, INK_DIM)
        height = glyph.get_height()
        if top + height > int(bottom) or column.w < 8:
            self.lidar_learn_rect = pygame.Rect(column.x, top, 0, 0)
            return top
        screen.blit(glyph, (column.x, top))
        rect = pygame.Rect(column.x, top, min(column.w, glyph.get_width()), height)
        self.lidar_learn_rect = rect
        self._keep_glyphs([rect])
        return top + height + 4

    def _lidar_table(self, screen, view: MonitorView, column, top: int, bottom: int) -> None:
        """Label, monospace number, unit. A row that does not fit is omitted whole."""
        import pygame

        from .tabnum import cell_px, lidar_metric_rows

        rows = lidar_metric_rows(view.fly_line, view.range_line)
        if not rows or top >= bottom:
            return
        labels = [label for label, _number, _unit in rows]
        label_w = max(self.font_sm.size(label)[0] for label in labels)
        num_chars = 9
        cell = cell_px(14)
        num_w = num_chars * cell
        gap_n = 12
        gap_u = 10
        num_x = column.x + label_w + gap_n
        unit_x = num_x + num_w + gap_u
        widest = 0
        for _label, _number, unit in rows:
            unit_w = self.font_sm.size(unit)[0] if unit else 0
            widest = max(widest, (unit_x + unit_w) - column.x)
        if widest > column.w or unit_x > column.right:
            return
        from .tabnum import baseline_shift, load_font

        mono = load_font(14)
        ui_dy, mono_dy = baseline_shift(self.font_sm, mono)
        step = max(ui_dy + self.font_sm.get_height(), mono_dy + mono.get_height())
        room = int(bottom) - int(top)
        fit = room // step if step else 0
        if fit <= 0:
            return
        shown = rows[:fit]
        if len(shown) == len(rows) and room >= step * len(rows):
            step = min(30, room // len(rows))
        previous = screen.get_clip()
        clip = pygame.Rect(column.x, top, column.w, max(0, int(bottom) - int(top)))
        screen.set_clip(clip if previous is None else clip.clip(previous))
        try:
            y = int(top)
            for label, number, unit in shown:
                if y + self.font_sm.get_height() > bottom:
                    break
                self._plain_glyph(screen, label, column.x, y + ui_dy, LABEL)
                if number:
                    self._cells(screen, number.rjust(num_chars), num_x, y + mono_dy, VALUE)
                if unit:
                    self._plain_glyph(screen, unit, unit_x, y + ui_dy, LABEL)
                y += step
        finally:
            screen.set_clip(previous)

    def _plain_glyph(self, screen, text: str, x: int, y: int, color) -> None:
        import pygame

        if not text:
            return
        glyph = self.font_sm.render(text, True, color)
        screen.blit(glyph, (int(x), int(y)))
        self._keep_glyphs([pygame.Rect(int(x), int(y), glyph.get_width(), glyph.get_height())])

    def _brain_sig(self, view: MonitorView, content) -> tuple:
        return (
            int(content.w),
            int(content.h),
            theme_name(),
            round(float(getattr(view, "likeness", 0.0) or 0.0), 3),
            round(float(getattr(view, "drift", 0.0) or 0.0), 3),
            int(float(getattr(view, "t", 0.0) or 0.0)),
            bool(getattr(view, "recognized", False)),
            str(getattr(view, "teacher_flash_l", "") or ""),
            str(getattr(view, "teacher_flash_r", "") or ""),
            bool(self.flash_open),
            str(getattr(view, "learner", "") or ""),
            str(getattr(view, "mode_label", "") or ""),
            str(getattr(view, "teacher_counts", "") or ""),
            str(getattr(view, "teacher_skips", "") or ""),
            str(getattr(view, "lead_plaque", "") or ""),
            round(float(getattr(view, "teacher_weight", 0.0) or 0.0), 3),
            bool(getattr(view, "brain_offline", False)),
            None if getattr(view, "drive_l", None) is None else round(float(view.drive_l), 2),
            None if getattr(view, "drive_r", None) is None else round(float(view.drive_r), 2),
            None if getattr(view, "turn", None) is None else round(float(view.turn), 3),
            None if getattr(view, "stim_dog", None) is None else round(float(view.stim_dog), 2),
            None if getattr(view, "stim_cube", None) is None else round(float(view.stim_cube), 2),
            None if getattr(view, "pref_index", None) is None else round(float(view.pref_index), 3),
            int(getattr(view, "pref_seconds", 0) or 0),
            None if getattr(view, "kc_gd_frac", None) is None else round(float(view.kc_gd_frac), 3),
            None if getattr(view, "kc_abp_frac", None) is None else round(float(view.kc_abp_frac), 3),
            int(getattr(view, "pam_l", 0) or 0),
            int(getattr(view, "pam_r", 0) or 0),
            int(getattr(view, "ppl1_l", 0) or 0),
            int(getattr(view, "ppl1_r", 0) or 0),
            int(getattr(view, "n_pam", 0) or 0),
            int(getattr(view, "total_pam", 0) or 0),
            int(getattr(view, "n_ppl1", 0) or 0),
            int(getattr(view, "kc_on", 0) or 0),
            int(getattr(view, "kc_n", 0) or 0),
            _series_mark(getattr(view, "peer_curve", ())),
            _series_mark(getattr(view, "other_curve", ())),
            _series_mark(getattr(view, "drift_curve", ())),
            _bins_mark(getattr(view, "kc_bins", None)),
            _events_mark(getattr(view, "dan_events", ())),
        )

    def _brain_panel_fresh(self, view: MonitorView, content) -> bool:
        cached = self._brain_panel_cache
        if not cached:
            return False
        size, sig, snap = cached
        if snap is None or size != (int(content.w), int(content.h)):
            return False
        return sig == self._brain_sig(view, content)

    def _draw_brain_block(self, screen, view: MonitorView, content) -> None:
        import pygame

        body = pygame.Rect(content.x, content.y, content.w, content.h)
        if getattr(view, "brain_offline", False):
            from .brain_panel import BRAIN_OFFLINE_TITLE, offline_plate

            plate = offline_plate(_scheme_rgb("card"), _scheme_rgb("ink_dim"))
            pygame.draw.rect(screen, plate, body)
            caption = self.font_sm.render(BRAIN_OFFLINE_TITLE, True, INK_DIM)
            screen.blit(
                caption,
                (body.centerx - caption.get_width() // 2, body.centery - caption.get_height() // 2),
            )
            self._brain_panel_cache = None
            return
        log_source = self.journal_rect
        self.log_rows = max(5, (log_source.h - 34) // 16) if log_source is not None else 5
        if not no_snap() and self._brain_panel_fresh(view, content):
            screen.blit(self._brain_panel_cache[2], content.topleft)
            return
        previous = screen.get_clip()
        clip = body if previous is None else body.clip(previous)
        screen.set_clip(clip)
        try:
            self._draw_brain_body(screen, view, body)
        finally:
            screen.set_clip(previous)
        if not no_snap() and screen.get_rect().contains(content):
            self._brain_panel_cache = (
                (int(content.w), int(content.h)),
                self._brain_sig(view, content),
                _snap(screen, content),
            )

    def _draw_brain_body(self, screen, view: MonitorView, body) -> None:
        import pygame

        if view.learner == "mb" and self.flash_open:
            self._draw_mb_head(screen, view, body.x + 8, body.y)
            from .learn_flash import draw_learn_panel

            draw_learn_panel(
                screen,
                self.font,
                self.font_sm,
                pygame.Rect(body.x + 8, body.y + 78, body.w - 16, max(80, body.h - 96)),
                view.mb_layout,
                view.learn_flash,
                view.t,
                view.learn_flash_r,
            )
            if self.journal_rect is not None:
                self.log_rows = max(3, (self.journal_rect.h - 34) // 16)
        elif view.learner == "mb":
            self._draw_mb(screen, view, pygame.Rect(body.x + 4, body.y + 4, body.w - 8, body.h - 8))
        else:
            self._draw_hebb(screen, view, pygame.Rect(body.x + 4, body.y + 4, body.w - 8, body.h - 8))

    def _run_marker_layout(self, content, view: MonitorView):
        """Two chip rows and a drive sample, or nothing when the journal is short."""
        import pygame

        from .run_marker import SIDES, STAGES

        if view.learner != "mb" or content.w < 160:
            return [], [], None, 0
        chip_h = max(18, int(self.font_sm.get_height()) + 4)
        sample_h = int(self.font_sm.get_height())
        gap = 3
        block = chip_h + gap + chip_h + gap + sample_h + 6
        if content.h < block + 16:
            return [], [], None, 0
        x = content.x + 4
        width = content.w - 8
        y = content.y + 2
        stage_rects = self._chip_row(x, y, width, chip_h, STAGES)
        side_rects = self._chip_row(x, y + chip_h + gap, width, chip_h, SIDES)
        sample_y = y + chip_h + gap + chip_h + gap
        return stage_rects, side_rects, sample_y, block

    def _chip_row(self, x: int, y: int, width: int, height: int, names) -> list:
        import pygame

        gap = 4
        count = len(names)
        chip_w = max(1, (int(width) - gap * (count - 1)) // count)
        rects = []
        cursor = int(x)
        for name in names:
            rects.append((pygame.Rect(cursor, int(y), chip_w, int(height)), name))
            cursor += chip_w + gap
        return rects

    def _paint_run_chips(self, screen, rects, selected: str) -> None:
        import pygame

        for rect, name in rects:
            on = name == selected
            pygame.draw.rect(screen, PEACH if on else INSET, rect, border_radius=4)
            pygame.draw.rect(screen, LINE, rect, 1, border_radius=4)
            glyph = self.font_sm.render(name, True, VALUE if on else LABEL)
            screen.blit(
                glyph,
                (rect.centerx - glyph.get_width() // 2, rect.centery - glyph.get_height() // 2),
            )

    def _paint_run_sample(self, screen, view: MonitorView, x: int, y: int) -> None:
        from .run_marker import run_label
        from .tabnum import signed

        tag = self.font_sm.render(run_label(view.run_stage, view.run_side), True, LABEL)
        screen.blit(tag, (x, y))
        cursor = x + tag.get_width() + 8
        left = self.font_sm.render("Л", True, LABEL)
        screen.blit(left, (cursor, y))
        cursor += left.get_width() + 4
        cursor = self._cells(screen, signed(view.drive_l, 7, 1), cursor, y, VALUE, size=13)
        cursor += 8
        right = self.font_sm.render("П", True, LABEL)
        screen.blit(right, (cursor, y))
        cursor += right.get_width() + 4
        self._cells(screen, signed(view.drive_r, 7, 1), cursor, y, VALUE, size=13)

    def _draw_journal_block(self, screen, view: MonitorView, content) -> None:
        from .tabnum import signed

        journal = self.journal_rect
        stage_rects, side_rects, sample_y, switch_h = self._run_marker_layout(content, view)
        self.run_stage_rects = stage_rects
        self.run_side_rects = side_rects
        floor = 3 if view.learner == "mb" and self.flash_open else 5
        room = (journal.h - 34 - switch_h) if journal is not None else floor * 16
        log_n = max(floor, room // 16)
        self.log_rows = int(log_n)
        lines = tuple(view.log_lines[-log_n:])
        stage = str(view.run_stage or "до")
        side = str(view.run_side or "слева")
        drives = (signed(view.drive_l, 7, 1), signed(view.drive_r, 7, 1))
        key = (int(content.w), int(content.h), lines, theme_name(), stage, side, drives, int(switch_h))
        # The drive sample registers monospace cells every frame. A cached blit
        # would skip that registration and the column check would see fewer rows.
        cached = None if no_snap() or switch_h else self._journal_cache
        if cached is not None and cached[0] == key:
            screen.blit(cached[1], content.topleft)
            return
        y = content.y + 2 + switch_h
        if switch_h:
            self._paint_run_chips(screen, stage_rects, stage)
            self._paint_run_chips(screen, side_rects, side)
            self._paint_run_sample(screen, view, content.x + 4, int(sample_y))
        width_chars = max(24, content.w // 8)
        for line in lines:
            if y + 16 > content.bottom:
                break
            screen.blit(self.font_sm.render(line[:width_chars], True, LABEL), (content.x + 4, y))
            y += 16
        if switch_h:
            self._journal_cache = None
        elif not no_snap() and screen.get_rect().contains(content):
            self._journal_cache = (key, _snap(screen, content))

    def _draw_controls_block(self, screen, view: MonitorView) -> None:
        self._mode_buttons(screen, view)
        if view.learner == "mb":
            self._treat_button(screen, view, self._mouse())
        self._button(
            screen,
            self.estop_rect,
            "E-STOP",
            None,
            self.estop_rect.collidepoint(self._mouse()),
            label_color=ESTOP_LABEL,
            fill=ESTOP,
            key="E",
        )

    def _keep_glyphs(self, glyphs) -> None:
        group = getattr(self, "_glyph_group", "") or ""
        store = getattr(self, "_glyph_rects", None)
        if store is None:
            self._glyph_rects = []
            store = self._glyph_rects
        for rect in glyphs:
            store.append((rect, group))

    def _draw_clipped(self, screen, rect, group: str, paint) -> None:
        """Draw one block inside its content. Nothing from the block leaves that rect."""
        previous = screen.get_clip()
        clip = rect if previous is None else rect.clip(previous)
        if clip.width <= 0 or clip.height <= 0:
            return
        saved = getattr(self, "_glyph_group", "")
        self._glyph_group = group
        screen.set_clip(clip)
        try:
            paint()
        finally:
            screen.set_clip(previous)
            self._glyph_group = saved

    def _cells(self, screen, text: str, x: int, y: int, color, size: int = 14) -> int:
        from .tabnum import blit_cells

        glyphs = []
        end = blit_cells(screen, text, x, y, color, size=size, rows=self._fixed_rows, glyphs=glyphs)
        self._keep_glyphs(glyphs)
        return end

    def _mixed(self, screen, line: str, x: int, y: int, pattern: str, value_color, label_color=None, status_samples=None, limit=None) -> int:
        glyphs = []
        end = paint_mixed(
            screen,
            x,
            y,
            line,
            pattern,
            self.font_sm,
            value_color,
            LABEL if label_color is None else label_color,
            rows=self._fixed_rows,
            anchors=self._anchors,
            status_samples=status_samples,
            limit=limit,
            glyphs=glyphs,
        )
        self._keep_glyphs(glyphs)
        return end

    def _paint_wrapped(
        self,
        screen,
        line: str,
        x: int,
        y: int,
        bottom: int,
        pattern: str,
        value_color,
        label_color=None,
        status_samples=None,
        limit=None,
    ):
        """Column grid. The whole line wraps or stays hidden; a number is never sliced."""
        glyphs = []
        nxt = paint_wrapped(
            screen,
            x,
            y,
            bottom,
            self._brain_row_h(),
            line,
            pattern,
            self.font_sm,
            value_color,
            LABEL if label_color is None else label_color,
            rows=self._fixed_rows,
            anchors=self._anchors,
            glyphs=glyphs,
            status_samples=status_samples,
            limit=limit,
        )
        if nxt is None:
            return None
        self._keep_glyphs(glyphs)
        return nxt

    def _paint_readout(self, screen, panel, y: int, tag: str, line: str, pattern: str, value_color, label_color=None, status_samples=None) -> int:
        """One brain readout. Every wrapped row is its own band, or the line is skipped."""
        from .tabnum import slots_of, wrap_slots

        row_h = self._brain_row_h()
        limit = panel.right - 4
        try:
            slots = slots_of(line, pattern, status_samples, split_words=True)
            wrapped = wrap_slots(slots, limit - panel.x, self.font_sm)
        except (ValueError, IndexError):
            wrapped = None
        if not wrapped or y + row_h * len(wrapped) > panel.bottom:
            return y
        cursor = y
        for index, row_slots in enumerate(wrapped):
            row_tag = tag if len(wrapped) == 1 else "%s-%d" % (tag, index)

            def paint(row_slots=row_slots, yy=cursor) -> None:
                self._paint_slot_row(screen, row_slots, panel.x, yy, limit, value_color, label_color)

            nxt = self._paint_line(screen, panel, cursor, row_tag, paint)
            if nxt is None:
                return y
            cursor = nxt
        return cursor

    def _paint_slot_row(self, screen, slots, x: int, y: int, limit, value_color, label_color) -> None:
        from .tabnum import paint_slots

        glyphs = []
        paint_slots(
            screen,
            x,
            y,
            slots,
            self.font_sm,
            value_color,
            LABEL if label_color is None else label_color,
            rows=self._fixed_rows,
            anchors=self._anchors,
            limit=limit,
            glyphs=glyphs,
        )
        self._keep_glyphs(glyphs)

    def _brain_row_h(self) -> int:
        from .tabnum import baseline_shift, load_font

        mono = load_font(14)
        ui_dy, mono_dy = baseline_shift(self.font_sm, mono)
        bottom = max(ui_dy + int(self.font_sm.get_height()), mono_dy + int(mono.get_height()))
        return bottom + 4

    def _note_brain_row(self, rect, tag: str) -> None:
        import pygame

        rows = getattr(self, "_brain_rows", None)
        if rows is None:
            self._brain_rows = []
            rows = self._brain_rows
        rows.append((pygame.Rect(rect), tag))

    def _with_row_clip(self, screen, band, paint) -> None:
        prev = screen.get_clip()
        clip = band if prev is None else band.clip(prev)
        if clip.w <= 0 or clip.h <= 0:
            return
        screen.set_clip(clip)
        try:
            paint()
        finally:
            screen.set_clip(prev)

    def _paint_line(self, screen, panel, y: int, tag: str, paint) -> int | None:
        """Draw one full row, or skip it when the block is too short."""
        import pygame

        row = self._brain_row_h()
        if y + row > panel.bottom:
            return None
        band = pygame.Rect(panel.x, y, panel.w, row)
        self._note_brain_row(band, tag)
        self._with_row_clip(screen, band, paint)
        return y + row

    def _sound_label(self) -> str:
        if self.beep_on and self.audio_ok is False:
            return "звук недоступен"
        if self.beep_on:
            return "звук вкл"
        return "звук выкл"

    def _gradient(self, surface) -> None:
        import pygame

        width, height = surface.get_size()
        for row in range(height):
            tone = _mix(BG_TOP, BG_BOT, row / max(height - 1, 1))
            pygame.draw.line(surface, tone, (0, row), (width, row))

    def _window_bg(self, size):
        import pygame

        require_main_thread()
        if self._win_bg is None or self._win_bg.get_size() != tuple(size):
            bg = pygame.Surface((int(size[0]), int(size[1])))
            self._gradient(bg)
            self._win_bg = bg
        return self._win_bg

    def _fill_bg(self, screen) -> None:
        import pygame

        require_main_thread()
        size = screen.get_size()
        if self._bg is None or self._bg.get_size() != size:
            bg = pygame.Surface(size)
            self._gradient(bg)
            self._bg = bg
        screen.blit(self._bg, (0, 0))

    def _card(self, screen, rect, radius: int = 16) -> None:
        import pygame

        require_main_thread()
        if rect.w < 2 or rect.h < 2:
            return
        radius = min(int(radius), rect.w // 2, rect.h // 2)
        key = (int(rect.w), int(rect.h), int(radius), theme_name())
        sprite = self._card_cache.get(key)
        if sprite is None:
            sprite = pygame.Surface((rect.w + 28, rect.h + 32), pygame.SRCALPHA)
            for spread, alpha in ((12, 16), (7, 24), (3, 32)):
                pygame.draw.rect(
                    sprite,
                    (*SHADOW, alpha),
                    (14 - spread, 16 - spread, rect.w + spread * 2, rect.h + spread * 2),
                    border_radius=radius + 4,
                )
            face = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            pygame.draw.rect(face, (*CARD, CARD_ALPHA), face.get_rect(), border_radius=radius)
            pygame.draw.rect(face, (*LINE, 255), face.get_rect(), width=1, border_radius=radius)
            sprite.blit(face, (14, 10))
            if not no_card_cache():
                if len(self._card_cache) > 64:
                    self._card_cache.clear()
                self._card_cache[key] = sprite
        screen.blit(sprite, (rect.x - 14, rect.y - 10))

    def _inset(self, screen, rect, radius: int = 12) -> None:
        """White nested card. No shadow, so it stays flat inside an outer card."""
        import pygame

        if rect.w < 2 or rect.h < 2:
            return
        radius = max(0, min(int(radius), rect.w // 2, rect.h // 2))
        pygame.draw.rect(screen, INSET, rect, border_radius=radius)
        pygame.draw.rect(screen, LINE, rect, 1, border_radius=radius)

    def _lamp(self, screen, center, color, radius: int = 6) -> None:
        import pygame

        require_main_thread()
        x, y = int(center[0]), int(center[1])
        if color == LAMP_OFF or color is None:
            pygame.draw.circle(screen, RING, (x, y), radius, 1)
            return
        if color == PEACH:
            top, bottom = PEACH_SOFT, PEACH
        elif color == AMBER:
            top, bottom = AMBER_SOFT, AMBER
        elif color == TEACHER:
            top, bottom = TEACHER_SOFT, TEACHER
        elif color == GREEN:
            top, bottom = LAMP_GREEN_HI, GREEN
        elif color == RED:
            top, bottom = LAMP_RED_HI, RED
        elif color == ESTOP:
            top, bottom = LAMP_ESTOP_HI, ESTOP
        else:
            top, bottom = _mix(LAMP_WASH, color, 0.35), color
        diameter = radius * 2
        disc = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        for row in range(diameter):
            pygame.draw.line(disc, (*_mix(top, bottom, row / max(diameter - 1, 1)), 255), (0, row), (diameter, row))
        mask = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), (radius, radius), radius)
        disc.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(disc, (x - radius, y - radius))

    def _draw_key_badge(self, screen, rect, key: str):
        """Small gray plaque at the button's right edge. The key itself is unchanged."""
        import pygame

        from .tabnum import load_font

        if not key or rect.w < 28 or rect.h < 12:
            return None
        glyph = load_font(13).render(str(key), True, KEY_BADGE_INK)
        pad_x = 4
        badge = pygame.Rect(0, 0, max(16, glyph.get_width() + pad_x * 2), max(14, min(rect.h - 8, glyph.get_height() + 2)))
        badge.centery = rect.centery
        badge.right = rect.right - 6
        pygame.draw.rect(screen, KEY_BADGE, badge, border_radius=4)
        screen.blit(glyph, (badge.centerx - glyph.get_width() // 2, badge.centery - glyph.get_height() // 2))
        return badge

    def _blit_clipped(self, screen, surface, pos, rect) -> None:
        previous = screen.get_clip()
        clip = rect if previous is None else rect.clip(previous)
        if clip.width <= 0 or clip.height <= 0:
            return
        screen.set_clip(clip)
        try:
            screen.blit(surface, pos)
        finally:
            screen.set_clip(previous)

    def _button(self, screen, rect, text: str, lamp, hot: bool, label_color=None, fill=None, key: str = "") -> None:
        import pygame

        if rect.w < 8 or rect.h < 8:
            return
        radius = min(16, max(1, rect.h // 2))
        if fill is not None:
            pygame.draw.rect(screen, fill, rect, border_radius=radius)
            if hot:
                pygame.draw.rect(screen, ESTOP_EDGE, rect, 2, border_radius=radius)
        else:
            self._card(screen, rect, radius=radius)
            if hot:
                pygame.draw.rect(screen, PEACH, rect, 1, border_radius=radius)
        x = rect.x + 12
        if lamp is not None:
            self._lamp(screen, (rect.x + 18, rect.centery), lamp, 6)
            x = rect.x + 32
        if not text:
            return
        label = self.font_sm.render(text, True, label_color or VALUE)
        if fill is not None:
            self._blit_clipped(screen, label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2), rect)
        else:
            self._blit_clipped(screen, label, (x, rect.centery - label.get_height() // 2), rect)
        if key:
            self._draw_key_badge(screen, rect, key)

    def _link_color(self, view: MonitorView):
        text = view.udp_status or ""
        low = text.lower()
        if "error" in low or "ошиб" in low:
            return RED
        if "нет связи" in text:
            return AMBER
        if " ok" in low or low.endswith("ok"):
            return PEACH
        if view.onboard:
            return AMBER
        return LAMP_OFF

    def _link_lamp(self, screen, view: MonitorView, x: int = 400) -> None:
        color = self._link_color(view)
        ink = INK_DIM if color == LAMP_OFF else color
        label = self.font_sm.render("борт", True, ink)
        stop = self.brain_rect.x - 16
        bar = getattr(self, "help_bar_rect", None)
        if bar is not None and bar.w > 0:
            stop = min(stop, bar.x - 8)
        if x + 14 > stop:
            return
        cy = MENU_H // 2
        self._lamp(screen, (x, cy), color, 6)
        if x + 14 + label.get_width() <= stop:
            screen.blit(label, (x + 14, (MENU_H - label.get_height()) // 2))

    def show_toast(self, text: str, seconds: float = 3.0) -> None:
        import time

        self._toast_text = str(text)
        self._toast_until = time.monotonic() + float(seconds)

    def uptime_s(self) -> float:
        import time

        return max(0.0, time.monotonic() - float(getattr(self, "_booted", time.monotonic())))

    def _draw_toast(self, screen) -> None:
        import time

        import pygame

        text = getattr(self, "_toast_text", "")
        if not text or time.monotonic() > float(getattr(self, "_toast_until", 0.0)):
            return
        label = self.font_sm.render(text, True, VALUE)
        pad = 10
        rect = pygame.Rect(0, 0, label.get_width() + pad * 2, label.get_height() + 8)
        rect.centerx = WIN_W // 2
        rect.y = MENU_H + 10
        pygame.draw.rect(screen, CARD, rect, border_radius=8)
        pygame.draw.rect(screen, LABEL, rect, 1, border_radius=8)
        screen.blit(label, (rect.x + pad, rect.centery - label.get_height() // 2))

    def _mode_buttons(self, screen, view: MonitorView) -> None:
        import pygame

        mouse = self._mouse()
        learn_on = bool(view.learning_on)
        auto_on = bool(view.autonomy_on)
        grabbed = "ПЕРЕХВАТ" in (view.pilot_mode or "")
        self._button(
            screen,
            self.learn_rect,
            "СТОП ОБУЧЕНИЯ" if learn_on else "СТАРТ ОБУЧЕНИЯ",
            PEACH if learn_on else LAMP_OFF,
            self.learn_rect.collidepoint(mouse),
            key="P",
        )
        self._button(
            screen,
            self.auto_rect,
            "СТОП АВТО" if auto_on else "СТАРТ АВТО",
            PEACH if auto_on else LAMP_OFF,
            self.auto_rect.collidepoint(mouse),
            key="A",
        )
        self._button(
            screen,
            self.take_rect,
            "ПЕРЕХВАТ",
            PEACH if grabbed else LAMP_OFF,
            self.take_rect.collidepoint(mouse),
            key="M",
        )
        if view.learner == "mb":
            bilateral = view.steer != "sectors"
            self._button(
                screen,
                self.steer_rect,
                "БИЛАТЕРАЛЬНО" if bilateral else "СЕКТОРЫ",
                None,
                self.steer_rect.collidepoint(mouse),
                key="K",
            )

    def _go2_button_width(self, short: bool = False) -> int:
        """Lamp, tiny caption, lamp, tiny caption, then the title."""
        server = self.font_tiny.size("сервер")[0]
        panel = self.font_tiny.size("панель")[0]
        title = self.font_sm.size("УПР" if short else "УПРАВЛЕНИЕ")[0]
        return 12 + 10 + 4 + server + 10 + 10 + 4 + panel + 8 + title + 12

    def _go2_button(self, screen) -> None:
        missing = bool(getattr(self, "go2_missing", False))
        self._button(
            screen,
            self.go2_rect,
            "",
            None,
            self.go2_rect.collidepoint(self._mouse()) and not missing,
        )
        if self.go2_rect.w < 8 or self.go2_rect.h < 8:
            self.go2_server_lamp = (0, 0)
            self.go2_panel_lamp = (0, 0)
            return
        server_on = bool(getattr(self, "go2_server_open", False))
        panel_on = bool(self.go2_open) and not missing
        ink = INK_DIM if missing else LABEL
        x = self.go2_rect.x + 14
        cy = self.go2_rect.centery
        self.go2_server_lamp = (x, cy)
        self._lamp(screen, self.go2_server_lamp, PEACH if server_on else LAMP_OFF, 5)
        server = self.font_tiny.render("сервер", True, ink)
        screen.blit(server, (x + 8, cy - server.get_height() // 2))
        x = x + 8 + server.get_width() + 12
        self.go2_panel_lamp = (x, cy)
        self._lamp(screen, self.go2_panel_lamp, PEACH if panel_on else LAMP_OFF, 5)
        panel = self.font_tiny.render("панель", True, ink)
        screen.blit(panel, (x + 8, cy - panel.get_height() // 2))
        x = x + 8 + panel.get_width() + 8
        word = "УПР" if getattr(self, "_header_short", False) else "УПРАВЛЕНИЕ"
        label = self.font_sm.render(word, True, INK_DIM if missing else VALUE)
        self._blit_clipped(screen, label, (x, cy - label.get_height() // 2), self.go2_rect)

    def _brain_button(self, screen, view: MonitorView) -> None:
        if view.learner != "mb":
            return
        self._button(
            screen,
            self.brain_rect,
            "МОЗГ" if getattr(self, "_header_short", False) else "МОЗГ 3D",
            PEACH if self.brain_open else LAMP_OFF,
            self.brain_rect.collidepoint(self._mouse()),
            key="J",
        )

    def _teacher_button(self, screen, view: MonitorView) -> None:
        """Title stays «УЧИТЕЛЬ». The state slot is as wide as «смотрит»."""
        if view.learner != "mb":
            return
        state = view.yolo_state or "выкл (нет обучения)"
        if state == "учит":
            lamp, status, ink = PEACH, "учит", VALUE
        elif state == "смотрит":
            lamp, status, ink = TEACHER, "смотрит", VALUE
        elif state == "учитель не запущен":
            lamp, status, ink = AMBER, "выкл", AMBER
        else:
            lamp, status, ink = LAMP_OFF, "выкл", INK_DIM
        rect = self.teacher_rect
        hot = rect.collidepoint(self._mouse())
        self._button(screen, rect, "", lamp, hot)
        if rect.w < 8 or rect.h < 8:
            return
        badge = self._draw_key_badge(screen, rect, "Y")
        word = "УЧ" if getattr(self, "_header_short", False) else "УЧИТЕЛЬ"
        title = self.font_sm.render(word, True, VALUE)
        self._blit_clipped(screen, title, (rect.x + 32, rect.centery - title.get_height() // 2), rect)
        if not getattr(self, "_header_short", False):
            slot = self.font_sm.size("смотрит")[0]
            glyph = self.font_sm.render(status, True, ink)
            right = badge.left - 8 if badge is not None else rect.right - 8
            self._blit_clipped(screen, glyph, (right - slot, rect.centery - glyph.get_height() // 2), rect)

    def _draw_det_boxes(self, screen, inner, boxes) -> None:
        """Screen-only YOLO frames. The camera array is not written."""
        import pygame

        if not boxes:
            return
        color = PEACH
        for box in boxes:
            x0 = inner.x + int(round(float(box.x0) * inner.w))
            x1 = inner.x + int(round(float(box.x1) * inner.w))
            y0 = inner.y + int(round(float(box.y0) * inner.h))
            y1 = inner.y + int(round(float(box.y1) * inner.h))
            rect = pygame.Rect(min(x0, x1), min(y0, y1), max(1, abs(x1 - x0)), max(1, abs(y1 - y0)))
            pygame.draw.rect(screen, color, rect, 1)
            kind = getattr(box, "kind", "") or ""
            tag = format_box_tag("", float(box.conf), kind)
            from .tabnum import cell_px

            need = cell_px(14) * len(tag)
            if rect.w >= need and rect.h >= 16:
                self._mixed(
                    screen,
                    tag,
                    rect.x + 2,
                    max(inner.y, rect.y - 16),
                    "[8] {4}%",
                    color,
                    label_color=color,
                    limit=rect.right - 2,
                )

    def _draw_robot_block(self, screen) -> None:
        mouse = self._mouse()
        self._button(screen, self.stand_up_rect, "ВСТАТЬ", None, self.stand_up_rect.collidepoint(mouse), key="+")
        self._button(screen, self.stand_down_rect, "ЛЕЧЬ", None, self.stand_down_rect.collidepoint(mouse), key="-")
        self._button(screen, self.recovery_rect, "ПОДЪЁМ", None, self.recovery_rect.collidepoint(mouse))
        self._lidar_reset_button(screen)

    def _record_button(self, screen, view: MonitorView) -> None:
        import pygame

        hot = self.record_rect.collidepoint(self._mouse())
        if view.record_on:
            text = format_record(view.record_saved, view.record_bytes, view.record_idle)
            lamp = PEACH
            self._button(screen, self.record_rect, "", lamp, hot)
            self._mixed(
                screen,
                text,
                self.record_rect.x + 32,
                self.record_rect.centery - 8,
                "ЗАПИСЬ {5} {10}",
                VALUE,
            )
            self._draw_key_badge(screen, self.record_rect, "U")
        else:
            word = "ЗАП" if getattr(self, "_header_short", False) else "ЗАПИСЬ КАДРОВ"
            self._button(screen, self.record_rect, word, LAMP_OFF, hot, key="U")

    def _ensure_audio(self) -> bool:
        if self.audio_ok is not None:
            return self.audio_ok
        try:
            import pygame

            require_main_thread()
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            spec = pygame.mixer.get_init()
            if not spec:
                raise RuntimeError("mixer silent")
            sr = int(spec[0])
            channels = int(spec[2])
            n = int(sr * 0.09)
            t = np.arange(n, dtype=np.float32) / float(sr)
            env = np.hanning(n).astype(np.float32)
            mono = (0.35 * np.sin(2 * np.pi * 880 * t) * env * 32767).astype(np.int16)
            if channels >= 2:
                wave = np.column_stack([mono, mono])
            else:
                wave = mono
            self._beep_samples = wave
            self._beep_sound = pygame.sndarray.make_sound(self._beep_samples)
            self.audio_ok = True
        except Exception:
            self.audio_ok = False
            self._beep_sound = None
        return self.audio_ok

    def _play_beep(self) -> None:
        if not self._ensure_audio() or self._beep_sound is None:
            return
        try:
            self._beep_sound.play()
        except Exception:
            self.audio_ok = False

    def _treat_button(self, screen, view: MonitorView, mouse) -> None:
        lamp = GREEN if view.treat_flash else GREEN
        self._button(screen, self.treat_rect, "ЛАКОМСТВО", lamp, self.treat_rect.collidepoint(mouse), key="T")
        if view.treat_flash:
            self._lamp(screen, (self.treat_rect.x + 16, self.treat_rect.centery), GREEN, 9)

    def _lidar_reset_button(self, screen) -> None:
        import pygame

        hot = self.lidar_reset_rect.collidepoint(self._mouse())
        self._button(screen, self.lidar_reset_rect, "СБРОС ЛИДАРА", None, hot, key="C")

    def _lidar_fresh_button(self, screen, view: MonitorView) -> None:
        import pygame

        hot = self.lidar_fresh_rect.collidepoint(self._mouse())
        if view.lidar_fresh_on:
            caption = view.lidar_mode or "свежий лидар"
            lamp = PEACH
        else:
            caption = view.lidar_mode or "лидар копится"
            lamp = LAMP_OFF
        self._button(screen, self.lidar_fresh_rect, "", lamp, hot)
        cap_x = self.lidar_fresh_rect.x + 32
        cap_y = self.lidar_fresh_rect.centery - 8
        if view.lidar_fresh_on:
            self._mixed(screen, caption, cap_x, cap_y, "свежий лидар {5} с", VALUE)
        else:
            screen.blit(self.font_sm.render(caption.strip(), True, VALUE), (cap_x, cap_y))
        self._draw_key_badge(screen, self.lidar_fresh_rect, "V")

    def _draw_marks(self, screen, inner, marks) -> None:
        import pygame

        require_main_thread()
        overlay = pygame.Surface((inner.w, inner.h), pygame.SRCALPHA)

        def pt(nx: float, ny: float) -> tuple[int, int]:
            return (
                int(np.clip(nx, -0.05, 1.05) * inner.w),
                int(np.clip(ny, -0.05, 1.05) * inner.h),
            )

        drawn = []
        for mark in marks:
            alpha = int(np.clip(float(getattr(mark, "alpha", 1.0)), 0.0, 1.0) * 210)
            if alpha < 8:
                continue
            origin = pt(mark.ox, mark.oy)
            tip = pt(mark.nx, mark.ny)
            if mark.kind == "cone":
                left = pt(mark.nx_l, mark.ny_l)
                right = pt(mark.nx_r, mark.ny_r)
                pygame.draw.polygon(overlay, (*GREEN, max(alpha // 3, 1)), (origin, left, right))
                pygame.draw.line(overlay, (*GREEN, alpha), origin, tip, 1)
            else:
                pygame.draw.line(overlay, (*GREEN, alpha), origin, tip, 1)
                pygame.draw.circle(overlay, (*GREEN, alpha), tip, 6)
                pygame.draw.circle(overlay, (*LAMP_GREEN_HI, alpha), tip, 6, 1)
            drawn.append((tip, alpha, mark))
        from .tabnum import cell_px, load_font, signed as _signed

        num_font = load_font(14)
        cell = cell_px(14)
        for i, (tip, _alpha, mark) in enumerate(drawn):
            text = _signed(round(float(mark.percent)), 4)
            ox = tip[0] + 12
            oy = tip[1] - 18 + i * 16
            for n, ch in enumerate(text):
                if ch == " ":
                    continue
                overlay.blit(num_font.render(ch, True, PEACH), (ox + n * cell, oy))
            overlay.blit(self.font_sm.render("%", True, PEACH), (ox + 4 * cell, oy))
        screen.blit(overlay, inner.topleft)

    def _eye_lamp_color(self, recognized: bool, ready: bool, percent: float) -> tuple:
        """Recognition only. Teacher PAM/PPL1 is a separate lamp on the camera."""
        if recognized:
            return GREEN
        if ready and float(percent) > 0.0:
            return AMBER
        return LAMP_OFF

    def _teacher_sees(self, view: MonitorView) -> tuple:
        """``(sees_L, sees_R)`` from box centres. Image left is Л, image right is П."""
        from .yolo_teacher import DEFAULT_CONF, hemisphere_hits

        return hemisphere_hits(
            view.teacher_boxes,
            view.overlap,
            conf_min=DEFAULT_CONF,
            center_only=True,
        )

    def _teacher_lamp_color(self, flash: str, sees: bool) -> tuple:
        if flash == "pam":
            return GREEN
        if flash == "ppl1":
            return RED
        if sees:
            return TEACHER
        return LAMP_OFF

    def _teacher_caption(self, flash: str, sees: bool) -> str:
        if flash == "pam":
            return "уч PAM"
        if flash == "ppl1":
            return "уч PPL1"
        if sees:
            return "уч видит"
        return "уч"

    def _eye_plaques(self, screen, inner, view: MonitorView) -> None:
        """Recognition lamp and a teacher lamp on the preview.

        Image left is Л, image right is П. The teacher lamp sits beside
        «узнаю»: green flash is PAM, red is PPL1, blue is a dog in that
        eye's field, an empty ring is no dog.
        """
        import pygame

        require_main_thread()
        height = 36
        mid = inner.x + inner.w // 2
        veil = pygame.Surface((inner.w, height), pygame.SRCALPHA)
        veil.fill((*CARD, 210))
        screen.blit(veil, (inner.x, inner.y))
        sees_l, sees_r = self._teacher_sees(view)
        self.eye_lamps = {
            "Л": self._eye_plaque(
                screen,
                pygame.Rect(inner.x, inner.y, mid - inner.x, height),
                "Л",
                view.eye_l_recognized,
                view.eye_l_ready,
                view.eye_l_confidence,
                view.teacher_flash_l,
                sees_l,
            ),
            "П": self._eye_plaque(
                screen,
                pygame.Rect(mid, inner.y, inner.right - mid, height),
                "П",
                view.eye_r_recognized,
                view.eye_r_ready,
                view.eye_r_confidence,
                view.teacher_flash_r,
                sees_r,
            ),
        }

    def _eye_plaque(
        self,
        screen,
        rect,
        name: str,
        recognized: bool,
        ready: bool,
        percent: float,
        flash: str = "",
        sees: bool = False,
    ) -> dict:
        from .tabnum import cell_px

        recog = (rect.x + 18, rect.centery)
        self._lamp(screen, recog, self._eye_lamp_color(recognized, ready, percent), 7)
        word = "УЗНАЮ" if recognized else "—"
        text_x = rect.x + 34
        self._cells(screen, format_plaque(name, word), text_x, rect.centery - 8, VALUE)
        teacher_x = text_x + cell_px(14) * len(format_plaque(name, word)) + 16
        teacher = (teacher_x, rect.centery)
        flashing = flash in ("pam", "ppl1")
        self._lamp(screen, teacher, self._teacher_lamp_color(flash, sees), 8 if flashing else 6)
        caption = self.font_tiny.render(self._teacher_caption(flash, sees), True, INK_DIM)
        screen.blit(caption, (teacher_x + 12, rect.centery - caption.get_height() // 2))
        return {"recog": recog, "teacher": teacher}

    def _eyes_banner(self, screen, view: MonitorView, rx: int, col_w: int, y: int, bottom: int | None = None) -> int:
        import pygame

        if bottom is None:
            bottom = y + 4096
        panel = pygame.Rect(rx, y, max(1, col_w), max(0, int(bottom) - int(y)))
        limit = panel.right - 4

        def eyes() -> None:
            cy = y + self._brain_row_h() // 2
            self._lamp(
                screen,
                (rx + 10, cy),
                self._eye_lamp_color(view.eye_r_recognized, view.eye_r_ready, view.eye_r_confidence),
                5,
            )
            from .tabnum import baseline_shift, load_font

            ui_dy, _mono_dy = baseline_shift(self.font_sm, load_font(14))
            screen.blit(self.font_sm.render("П", True, PEACH), (rx + 20, y + ui_dy))
            self._lamp(
                screen,
                (rx + 52, cy),
                self._eye_lamp_color(view.eye_l_recognized, view.eye_l_ready, view.eye_l_confidence),
                5,
            )
            screen.blit(self.font_sm.render("Л", True, PEACH), (rx + 62, y + ui_dy))
            if view.eyes_line:
                self._mixed(
                    screen,
                    view.eyes_line,
                    rx + 88,
                    y,
                    "R_L {7}  R_R {7}  [22]",
                    VALUE,
                    status_samples=EYES_STATUS,
                    limit=limit,
                )

        nxt = self._paint_line(screen, panel, y, "eyes", eyes)
        if nxt is None:
            return y
        y = nxt
        if view.recog_line:
            def recog() -> None:
                screen.blit(self.font_sm.render(view.recog_line, True, INK_DIM), (rx + 8, y))

            nxt = self._paint_line(screen, panel, y, "recog", recog)
            if nxt is None:
                return y
            y = nxt
        word = view.pilot_mode or "РУЧНОЕ"
        if word == "АВТОНОМИЯ" or "ПЕРЕХВАТ" in word:
            tone = PEACH
        elif word == "СТОП":
            tone = ESTOP
        else:
            tone = VALUE
        mode = format_pilot_mode(word)
        rest = format_pilot_rest(view.pilot_who, view.phase_ru, view.last_seen_side)

        def mode_row() -> None:
            from .tabnum import cell_px

            self._cells(screen, mode, rx, y, tone)
            rest_x = rx + cell_px(14) * MODE_W
            if rest_x + cell_px(14) * len(rest) <= limit:
                self._cells(screen, rest, rest_x, y, INK_DIM)

        nxt = self._paint_line(screen, panel, y, "mode", mode_row)
        if nxt is None:
            return y
        y = nxt
        if view.pilot_hint:
            def hint() -> None:
                screen.blit(self.font_sm.render(view.pilot_hint, True, AMBER), (rx, y))

            nxt = self._paint_line(screen, panel, y, "hint", hint)
            if nxt is None:
                return y
            y = nxt
        return y

    def _draw_mb_head(self, screen, view: MonitorView, rx: int, y: int | None = None) -> None:
        col_w = self.panel_rect.w if hasattr(self, "panel_rect") else WIN_W - rx - 16
        if y is None:
            y = self.panel_rect.y if hasattr(self, "panel_rect") else 56
        if view.eyes_line or view.recog_line or view.learner == "mb":
            self._eyes_banner(screen, view, rx, col_w, y, bottom=y + 76)
            return
        if view.recognized:
            phrase, tone = "УЗНАЮ СОРОДИЧА", GREEN
        else:
            phrase, tone = "НЕ УЗНАЮ", INK_DIM
        screen.blit(self.font_ind.render(phrase, True, tone), (rx, y))
        from .tabnum import cell_px, signed

        pct = (signed(view.confidence, 4) + "%") if view.confidence_ready else "    …"
        self._cells(screen, pct, rx + col_w - cell_px(14) * len(pct), y, tone)

    def _draw_mb(self, screen, view: MonitorView, panel) -> int:
        import pygame

        self.lead_slider_rect = pygame.Rect(0, 0, 0, 0)
        self.approach_rect = pygame.Rect(0, 0, 0, 0)
        self.search_lamp_at = (0, 0)
        sm = self.font_sm
        rx, col_w = panel.x, panel.w
        limit = panel.right - 4
        y = self._eyes_banner(screen, view, rx, col_w, panel.y, bottom=panel.bottom)
        y = self._approach_block(screen, view, panel, y, limit)
        y = self._progress_box(screen, view, panel, y)
        row = self._brain_row_h()
        # Curves are optional. The readout lines below them stay on their own rows.
        keep_after_plots = row * 5 + 8
        plot_h = 72
        if y + plot_h + keep_after_plots <= panel.bottom and col_w >= 180:
            plot_w = (col_w - 8) // 2
            left = pygame.Rect(rx, y, plot_w, plot_h)
            right = pygame.Rect(rx + plot_w + 8, y, col_w - plot_w - 8, plot_h)
            for plot_rect, series, color, caption in (
                (left, view.peer_curve, GREEN, "собака в кадре"),
                (right, view.other_curve, AMBER, "нет собаки"),
            ):
                label_rect, graph_rect = _plot(screen, plot_rect, series, 0.0, 1.0, color, sm, caption)
                if label_rect is not None and label_rect.w > 0 and label_rect.h > 0:
                    self._note_brain_row(label_rect, "plot-label")
                if graph_rect.w > 0 and graph_rect.h > 0:
                    self._note_brain_row(graph_rect, "plot")
            y += plot_h + 6
        series = [(float(t), float(v)) for t, v in view.drift_curve]
        if view.session_drift is not None:
            base = float(view.drift) - float(view.session_drift)
            series = [(t, v - base) for t, v in series]
            if series:
                series[-1] = (series[-1][0], float(view.session_drift))
            elif view.drift or view.session_drift:
                series = [(float(view.t), float(view.session_drift))]
        else:
            if series:
                series[-1] = (series[-1][0], float(view.drift))
            elif view.drift:
                series = [(float(view.t), float(view.drift))]
        from .brain_panel import drift_axis

        drift_lo, drift_hi = drift_axis(series)
        drift_line = format_drift_span(view.session_drift, view.drift)
        drift_pattern = "дрейф: сессия {7} · с начала {7}"

        def drift_row() -> None:
            self._mixed(screen, drift_line, rx, y, drift_pattern, VALUE, limit=limit - 78)
            lamp_x = limit - 62
            cy = y + self._brain_row_h() // 2
            color = AMBER if view.search_lamp else LAMP_OFF
            self._lamp(screen, (lamp_x, cy), color, 5)
            self.search_lamp_at = (lamp_x, cy)
            word = self.font_sm.render("поиск", True, AMBER if view.search_lamp else INK_DIM)
            screen.blit(word, (lamp_x + 12, y))
            self._keep_glyphs([pygame.Rect(lamp_x + 12, y, word.get_width(), word.get_height())])

        nxt = self._paint_line(screen, panel, y, "drift", drift_row)
        if nxt is None:
            return y
        y = nxt
        keep_after_drift_plot = row * 3 + 8
        drift_plot_h = 48
        if y + drift_plot_h + keep_after_drift_plot <= panel.bottom and series:
            drift_rect = pygame.Rect(rx, y, col_w, drift_plot_h)
            _label_rect, graph_rect = _plot(screen, drift_rect, series, drift_lo, drift_hi, INK_DIM, sm, "")
            if graph_rect.h > 0:
                self._note_brain_row(graph_rect, "drift-plot")
            y += drift_plot_h + 4
        y = self._dan_timeline(screen, view, panel, y)
        y = self._kc_row(screen, view, panel, y)
        if view.teacher_skips:
            quiet = skips_are_quiet(view.teacher_skips)
            tone = LABEL if quiet else AMBER
            y = self._paint_readout(
                screen,
                panel,
                y,
                "skips",
                view.teacher_skips,
                "ложных узнаваний без наказания: {7}  [36]",
                tone,
                label_color=tone,
                status_samples=SKIP_STATUS,
            )
        y = self._draw_lead_row(screen, view, panel, y)
        return y

    def _lead_at(self, win_pos, inp, drag: bool) -> bool:
        """Map a pointer x onto the teacher weight. Drag keeps the track."""
        import pygame

        rect = getattr(self, "lead_slider_rect", None)
        if rect is None or rect.w < 8:
            return False
        logical = self._logical_pos(win_pos)
        point = pygame.Rect(rect)
        if drag:
            point.inflate_ip(0, 28)
        if not point.collidepoint(logical[0], logical[1]):
            return False
        span = float(rect.w)
        share = (float(logical[0]) - float(rect.x)) / span
        if share < 0.0:
            share = 0.0
        elif share > 1.0:
            share = 1.0
        inp.teacher_weight = share
        return True

    def _draw_lead_row(self, screen, view, panel, y: int) -> int:
        """Plaque and weight track. Omitted whole when the block is short."""
        import pygame

        from .teacher_lead import lead_plaque

        self.lead_slider_rect = pygame.Rect(0, 0, 0, 0)
        if getattr(view, "learner", "") != "mb":
            return y
        weight = float(getattr(view, "teacher_weight", 0.0) or 0.0)
        if weight < 0.0:
            weight = 0.0
        elif weight > 1.0:
            weight = 1.0
        plaque = str(getattr(view, "lead_plaque", "") or "") or lead_plaque(weight)

        def paint_plaque() -> None:
            screen.blit(self.font_sm.render(plaque, True, LABEL), (panel.x, y))

        nxt = self._paint_line(screen, panel, y, "lead", paint_plaque)
        if nxt is None:
            return y
        y = nxt
        row = self._brain_row_h()
        if y + row > panel.bottom:
            return y
        band = pygame.Rect(panel.x, y, panel.w, row)
        track = pygame.Rect(panel.x + 8, band.centery - 3, max(1, panel.w - 16), 6)

        def paint_track() -> None:
            pygame.draw.rect(screen, INSET, track, border_radius=3)
            fill_w = int(round(track.w * weight))
            if fill_w > 0:
                pygame.draw.rect(screen, PEACH, pygame.Rect(track.x, track.y, fill_w, track.h), border_radius=3)
            pygame.draw.rect(screen, LINE, track, width=1, border_radius=3)
            thumb_x = track.x + int(round((track.w - 1) * weight))
            thumb = pygame.Rect(0, 0, 8, 14)
            thumb.center = (thumb_x, track.centery)
            if thumb.x < track.x:
                thumb.x = track.x
            if thumb.right > track.right:
                thumb.right = track.right
            pygame.draw.rect(screen, PEACH, thumb, border_radius=2)

        nxt = self._paint_line(screen, panel, y, "lead-slider", paint_track)
        if nxt is None:
            return y
        self.lead_slider_rect = track
        return nxt

    def _progress_line(self, screen, line: str, x: int, y: int, limit=None) -> None:
        if line.startswith("метки "):
            screen.blit(self.font_sm.render(line, True, LABEL), (x, y))
            return
        if line.startswith("Всего"):
            self._mixed(screen, line, x, y, "Всего за всё время: PAM {7} / PPL1 {7} (всего {7})", VALUE, limit=limit)
            return
        if line.startswith("За этот"):
            self._mixed(
                screen,
                line,
                x,
                y,
                "За этот запуск: PAM {7} / PPL1 {7}  {12}  [15]",
                VALUE,
                status_samples=SOUND_STATUS,
                limit=limit,
            )
            return
        if "знакомство" in line:
            self._mixed(screen, line, x, y, "[8]{12}  знакомство {7}", VALUE, limit=limit)
            return
        if line == "нет данных":
            screen.blit(self.font_sm.render(line, True, LABEL), (x, y))
            return
        if line.startswith("D"):
            self._mixed(screen, line, x, y, "D−N сессия {6} {4}% {5}  всего {6} {4}% {5}", VALUE, limit=limit)
            return
        from .tabnum import cell_px

        width = cell_px(14) * len(line)
        if limit is not None and x + width > int(limit):
            return
        self._cells(screen, line, x, y, VALUE)

    def _progress_box(self, screen, view: MonitorView, panel, y: int) -> int:
        import pygame

        row = self._brain_row_h()
        pad = 6
        if y + pad + row + pad > panel.bottom:
            return y
        if view.dan_mode == "familiarity":
            lines = [
                format_familiar("сессия", view.session_time, view.session_novelty),
                format_familiar("всего", view.total_time, view.total_novelty),
            ]
        else:
            lines = [
                format_lifetime(view.total_pam, view.total_ppl1, view.lifetime_known),
                format_session_counts(view.n_pam, view.n_ppl1, view.session_time, self._sound_label()),
            ]
        lines.append(
            format_metrics_line(
                view.session_sep,
                view.session_acc,
                view.session_labeled,
                view.total_sep,
                view.total_acc,
                view.total_labeled,
                bool(view.session_labeled or view.total_labeled),
            )
        )
        n = 0
        used = pad + row + pad
        while n < len(lines) and y + used + row <= panel.bottom:
            n += 1
            used += row
        rect = pygame.Rect(panel.x, y, panel.w, used)
        self._inset(screen, rect)
        limit = panel.right - 8
        title_y = y + pad

        def title() -> None:
            screen.blit(self.font_sm.render("насколько обучен", True, PEACH), (panel.x + 8, title_y))

        self._paint_line(screen, panel, title_y, "progress-title", title)
        cursor = title_y + row
        for index in range(n):
            line = lines[index]
            line_y = cursor

            def paint(line=line, line_y=line_y) -> None:
                self._progress_line(screen, line, panel.x + 8, line_y, limit=limit)

            self._paint_line(screen, panel, line_y, "progress-%d" % index, paint)
            cursor += row
        return y + used + 4

    def _approach_block(self, screen, view: MonitorView, panel, y: int, limit: int) -> int:
        import pygame

        title = "подход − избегание"

        def title_row() -> None:
            screen.blit(self.font_sm.render(title, True, PEACH), (panel.x, y))

        nxt = self._paint_line(screen, panel, y, "drive-title", title_row)
        if nxt is None:
            return y
        y = nxt
        drive = format_drive_line(view.drive_l, view.drive_r, view.turn)

        def drive_row() -> None:
            self._mixed(screen, drive, panel.x, y, "Л {7}  {1}  П {7}", VALUE, limit=limit)

        nxt = self._paint_line(screen, panel, y, "drive", drive_row)
        if nxt is None:
            return y
        y = nxt
        track_h = 8
        gap = 4
        block_h = track_h + gap + track_h
        if y + block_h <= panel.bottom and panel.w >= 80:
            bars = pygame.Rect(panel.x, y, panel.w, block_h)
            self._note_brain_row(bars, "drive-bars")
            scale = 1.0
            for value in (view.drive_l, view.drive_r):
                if value is not None:
                    scale = max(scale, abs(float(value)))
            self._mirror_bars(screen, pygame.Rect(panel.x, y, panel.w, track_h), view.drive_l, view.drive_r, scale)
            self._zero_bar(screen, pygame.Rect(panel.x, y + track_h + gap, panel.w, track_h), view.turn)
            y += block_h
        stim = format_stim_line(view.stim_dog, view.stim_cube)

        def stim_row() -> None:
            self._mixed(screen, stim, panel.x, y, "сородич {7}  контроль (куб) {7}", VALUE, limit=limit)

        nxt = self._paint_line(screen, panel, y, "stim", stim_row)
        if nxt is None:
            return y
        y = nxt
        pref = format_pref_line(view.pref_index, view.pref_seconds)
        pattern = "сородич−куб {12}  {6}" if view.pref_index is None else "сородич−куб {7}  {6}"

        def pref_row() -> None:
            end = self._mixed(screen, pref, panel.x, y, pattern, VALUE, limit=limit)
            word = "новый заход"
            glyph = self.font_sm.render(word, True, LABEL)
            width = int(glyph.get_width()) + 16
            if end + 8 + width <= limit:
                rect = pygame.Rect(end + 8, y, width, max(1, self._brain_row_h() - 2))
                pygame.draw.rect(screen, INSET, rect, border_radius=3)
                pygame.draw.rect(screen, LINE, rect, width=1, border_radius=3)
                screen.blit(glyph, (rect.x + 8, rect.centery - glyph.get_height() // 2))
                self._keep_glyphs([pygame.Rect(rect.x + 8, rect.y, glyph.get_width(), glyph.get_height())])
                self.approach_rect = rect

        nxt = self._paint_line(screen, panel, y, "pref", pref_row)
        if nxt is None:
            return y
        return nxt

    def _dan_timeline(self, screen, view: MonitorView, panel, y: int) -> int:
        import pygame

        title = "шкала DAN"

        def title_row() -> None:
            screen.blit(self.font_sm.render(title, True, PEACH), (panel.x + 8, y))

        nxt = self._paint_line(screen, panel, y, "dan-title", title_row)
        if nxt is None:
            return y
        y = nxt
        line = format_lamps(view.pam_l, view.pam_r, view.ppl1_l, view.ppl1_r)
        limit = panel.right - 4

        def lamps() -> None:
            x = panel.x + 8
            cy = y + self._brain_row_h() // 2
            for color in (GREEN, GREEN, RED, RED):
                pygame.draw.circle(screen, color, (x + 4, cy), 4)
                x += 12
            self._mixed(screen, line, x + 4, y, "PAM Л {7}  PAM П {7}  PPL1 Л {7}  PPL1 П {7}", VALUE, limit=limit)

        nxt = self._paint_line(screen, panel, y, "dan-lamps", lamps)
        if nxt is None:
            return y
        return nxt

    def _kc_row(self, screen, view: MonitorView, panel, y: int) -> int:
        import pygame

        from .tabnum import SLOT_GAP, slot_width, slots_of

        line = format_kc_scales(view.kc_gd_frac, view.kc_abp_frac)
        row_h = self._brain_row_h()
        if y + row_h > panel.bottom:
            return y
        try:
            slots = slots_of(line, "γ-d {4}%  α/β-p {4}%")
        except (ValueError, IndexError):
            slots = []
        label_w = 0
        for slot in slots:
            if label_w:
                label_w += SLOT_GAP
            label_w += slot_width(slot, self.font_sm)
        label_w += 4
        room = panel.w - label_w - 8
        if room >= 96:
            label_rect = pygame.Rect(panel.x, y, label_w, row_h)
            scale_rect = pygame.Rect(label_rect.right + 8, y, panel.right - label_rect.right - 8, row_h)
            self._note_brain_row(label_rect, "kc")
            self._note_brain_row(scale_rect, "kc-scale")
            self._with_row_clip(
                screen,
                label_rect,
                lambda: self._mixed(
                    screen,
                    line,
                    panel.x,
                    y,
                    "γ-d {4}%  α/β-p {4}%",
                    VALUE,
                    limit=label_rect.right,
                ),
            )
            self._kc_tracks(screen, scale_rect, view.kc_gd_frac, view.kc_abp_frac)
            return y + row_h
        nxt = self._paint_line(
            screen,
            panel,
            y,
            "kc",
            lambda: self._mixed(screen, line, panel.x, y, "γ-d {4}%  α/β-p {4}%", VALUE, limit=panel.right - 4),
        )
        if nxt is None:
            return y
        y = nxt
        band_h = 14
        if y + band_h > panel.bottom or panel.w < 80:
            return y
        band = pygame.Rect(panel.x, y, panel.w, band_h)
        self._note_brain_row(band, "kc-scale")
        self._kc_tracks(screen, band, view.kc_gd_frac, view.kc_abp_frac)
        return y + band_h

    def _kc_tracks(self, screen, band, gd, ab) -> None:
        import pygame

        gap = 12
        track_h = 8
        top = band.y + max(0, (band.h - track_h) // 2)
        track_w = max(8, (band.w - gap) // 2)
        self._kc_scale(screen, pygame.Rect(band.x, top, track_w, track_h), gd)
        self._kc_scale(
            screen,
            pygame.Rect(band.x + track_w + gap, top, band.right - (band.x + track_w + gap), track_h),
            ab,
        )

    def _mirror_bars(self, screen, rect, left, right, scale: float) -> None:
        """One centre. Л grows left, П grows right. Approach is peach."""
        import pygame

        if rect.w < 8 or rect.h < 4:
            return
        pygame.draw.rect(screen, INSET, rect, border_radius=3)
        pygame.draw.rect(screen, LINE, rect, width=1, border_radius=3)
        mid = rect.x + rect.w // 2
        half = max(1, rect.w // 2)
        self._outward(screen, rect, mid, half, left, scale, -1)
        self._outward(screen, rect, mid, half, right, scale, 1)
        pygame.draw.line(screen, LINE, (mid, rect.y), (mid, rect.bottom - 1))

    def _outward(self, screen, rect, mid: int, half: int, value, scale: float, sign: int) -> None:
        """Fill away from ``mid``. ``sign`` −1 is the left half."""
        import pygame

        if value is None or scale <= 0:
            return
        share = abs(float(value)) / float(scale)
        if share > 1.0:
            share = 1.0
        width = int(round(half * share))
        if width <= 0:
            return
        color = PEACH if float(value) >= 0.0 else INK_DIM
        if sign < 0:
            box = pygame.Rect(mid - width, rect.y, width, rect.h)
        else:
            box = pygame.Rect(mid, rect.y, width, rect.h)
        pygame.draw.rect(screen, color, box)

    def _zero_bar(self, screen, rect, value) -> None:
        """Yaw with zero in the middle. Left (positive) fills toward Л."""
        import pygame

        if rect.w < 8 or rect.h < 4:
            return
        pygame.draw.rect(screen, INSET, rect, border_radius=3)
        pygame.draw.rect(screen, LINE, rect, width=1, border_radius=3)
        mid = rect.x + rect.w // 2
        if value is not None:
            share = float(value)
            if share > 1.0:
                share = 1.0
            elif share < -1.0:
                share = -1.0
            width = int(round((rect.w // 2) * abs(share)))
            if width > 0:
                if share >= 0.0:
                    box = pygame.Rect(mid - width, rect.y, width, rect.h)
                else:
                    box = pygame.Rect(mid, rect.y, width, rect.h)
                pygame.draw.rect(screen, PEACH, box)
        pygame.draw.line(screen, LINE, (mid, rect.y), (mid, rect.bottom - 1))

    def _kc_scale(self, screen, track, frac) -> None:
        """0–20% track. The green band is the 5–10% sparsity the eye keeps."""
        import pygame

        if track.w < 8 or track.h < 4:
            return
        from .mb_constants import KC_FRAC_MAX, KC_FRAC_MIN

        pygame.draw.rect(screen, INSET, track, border_radius=3)
        pygame.draw.rect(screen, LINE, track, width=1, border_radius=3)
        span = float(KC_FRAC_MAX) * 2.0
        z0 = track.x + int(round(track.w * (float(KC_FRAC_MIN) / span)))
        z1 = track.x + int(round(track.w * (float(KC_FRAC_MAX) / span)))
        if z1 > z0:
            zone = pygame.Rect(z0, track.y, max(1, z1 - z0), track.h)
            pygame.draw.rect(screen, GREEN, zone)
        if frac is not None:
            share = float(frac) / span
            if share < 0.0:
                share = 0.0
            elif share > 1.0:
                share = 1.0
            thumb_x = track.x + int(round((track.w - 1) * share))
            pygame.draw.line(screen, PEACH, (thumb_x, track.y - 2), (thumb_x, track.bottom + 2), 2)

    def _draw_hebb(self, screen, view: MonitorView, panel) -> int:
        import pygame

        sm = self.font_sm
        rx, col_w = panel.x, panel.w
        y = panel.y
        from .tabnum import cell_px, signed

        self._cells(screen, signed(view.likeness, 7, 2), rx, y, INK, size=26)
        screen.blit(sm.render("похожесть    слой сравнения, не KC→MBON", True, INK_DIM), (rx + cell_px(26) * 8, y + 8))
        self._mixed(
            screen,
            format_hebb_proto(view.learned_match, view.n_self, view.n_other),
            rx,
            y + 36,
            "к прототипу {7}  свой {7}  прочее {7}",
            VALUE,
        )
        screen.blit(sm.render(view.operator_label[:86], True, INK_DIM), (rx, y + 52))
        plot_w = (col_w - 8) // 2
        _plot(screen, pygame.Rect(rx, y + 76, plot_w, 120), view.peer_curve, 0.0, 1.0, GREEN, sm, "собака в кадре")
        _plot(
            screen,
            pygame.Rect(rx + plot_w + 8, y + 76, col_w - plot_w - 8, 120),
            view.other_curve,
            0.0, 1.0,
            AMBER,
            sm,
            "нет собаки",
        )
        inv = [(float(row.get("t", 0.0)), float(row.get("invariance", 0.0))) for row in view.metric_curve]
        _plot(screen, pygame.Rect(rx, y + 204, col_w, 80), inv, -1.0, 1.0, INK_DIM, sm, "инвариантность  собака стоит − дистрактор быстрый")
        half = (col_w - 12) // 2
        _bars(screen, sm, (rx, y + 292), view.proto_self, half, "прототип сородича", INK_DIM)
        _bars(screen, sm, (rx + half + 12, y + 292), view.proto_other, half, "прототип прочего", INK_DIM)
        purity = view.metric_curve[-1].get("purity") if view.metric_curve else None
        if purity is not None:
            self._mixed(screen, format_purity(float(purity)), rx, y + 400, "чистота {4} / {2}", VALUE)
        return min(y + 430, panel.bottom - 96)

    def _frame(self, screen, rect, image, title, error: str, chrome: bool = True):
        import pygame

        require_main_thread()
        if rect.w < 4 or rect.h < 4:
            return None
        if chrome:
            self._card(screen, rect)
            if title:
                screen.blit(self.font_sm.render(title, True, PEACH), (rect.x + 16, rect.y + 12))
            avail = pygame.Rect(rect.x + 10, rect.y + 36, max(1, rect.w - 20), max(1, rect.h - 48))
        else:
            avail = pygame.Rect(rect.x + 4, rect.y + 4, max(1, rect.w - 8), max(1, rect.h - 8))
        inner = None
        if image is not None and getattr(image, "size", 0):
            surf = _surf_from_rgb(image)
            iw, ih = surf.get_size()
            scale = min(avail.w / float(max(iw, 1)), avail.h / float(max(ih, 1)))
            tw = max(1, int(round(iw * scale)))
            th = max(1, int(round(ih * scale)))
            inner = pygame.Rect(
                avail.x + (avail.w - tw) // 2,
                avail.y + (avail.h - th) // 2,
                tw,
                th,
            )
            scaled = pygame.transform.smoothscale(surf, (inner.w, inner.h))
            screen.blit(scaled, inner.topleft)
        if error:
            shown, quiet = _camera_caption(error)
            ink = INK_DIM if quiet else RED
            y = rect.y + 40
            for chunk in _wrap(shown, 64):
                screen.blit(self.font_sm.render(chunk, True, ink), (rect.x + 10, y))
                y += 16
        return inner

    def _camera_overlap(self, screen, inner, view: MonitorView) -> None:
        """Binocular zone and one «УЗНАЮ» per eye. The camera array is not written.

        Image left is the robot's left eye (Л). Image right is the right eye (П).
        The shared band has a thin peach boundary on each side.
        """
        import pygame

        require_main_thread()
        from .hemifield import overlap_bands

        lo, hi = overlap_bands(view.overlap)
        x0 = inner.x + int(round(lo * inner.w))
        x1 = inner.x + int(round(hi * inner.w))
        if x1 > x0 + 1:
            pygame.draw.line(screen, PEACH, (x0, inner.y + 1), (x0, inner.bottom - 1), 1)
            pygame.draw.line(screen, PEACH, (x1, inner.y + 1), (x1, inner.bottom - 1), 1)
        else:
            mid = inner.x + inner.w // 2
            pygame.draw.line(screen, PEACH, (mid, inner.y + 1), (mid, inner.bottom - 1), 1)
            x0 = mid
            x1 = mid
        self._zone_tag(screen, inner.x, x0, inner, "Л")
        if x1 > x0 + 18:
            self._zone_tag(screen, x0, x1, inner, "Л+П")
        self._zone_tag(screen, x1, inner.right, inner, "П")
        from .tabnum import cell_px

        pct = format_overlap(view.overlap)
        cell = cell_px(14)
        pad_w = cell * len(pct) + 10
        pad_h = 22
        pad = pygame.Surface((pad_w, pad_h), pygame.SRCALPHA)
        pad.fill((*CARD, 230))
        px = inner.right - pad_w - 6
        py = inner.bottom - pad_h - 6
        screen.blit(pad, (px, py))
        self._mixed(screen, pct, px + 5, py + 2, "перекрытие {4}%", VALUE)

    def _zone_tag(self, screen, x_lo: int, x_hi: int, inner, name: str) -> None:
        import pygame

        require_main_thread()
        if x_hi - x_lo < 16:
            return
        label = self.font_sm.render(name, True, PEACH)
        pad = pygame.Surface((label.get_width() + 10, label.get_height() + 4), pygame.SRCALPHA)
        pad.fill((*CARD, 220))
        x = x_lo + max(4, (x_hi - x_lo - pad.get_width()) // 2)
        y = inner.bottom - pad.get_height() - 28
        screen.blit(pad, (x, y))
        screen.blit(label, (x + 5, y + 2))

    def _camera_halves(self, screen, inner, view: MonitorView) -> None:
        """Centre line on the displayed camera. The JPEG is not touched.

        Sector 0 is the right columns and the robot's right, so the picture's
        left half is Л / R_L and the right half is П / R_R.
        """
        import pygame

        mid = inner.x + inner.w // 2
        left = pygame.Rect(inner.x, inner.y, mid - inner.x, inner.h)
        right = pygame.Rect(mid, inner.y, inner.right - mid, inner.h)
        show = (
            view.steer == "bilateral"
            and view.hemi_l is not None
            and view.hemi_r is not None
        )
        pygame.draw.line(screen, PEACH, (mid, inner.y + 1), (mid, inner.bottom - 1), 1)
        self._half_tag(screen, left, "Л", view.hemi_l if show else None)
        self._half_tag(screen, right, "П", view.hemi_r if show else None)

    def _wash(self, screen, rect, color: tuple[int, int, int]) -> None:
        import pygame

        require_main_thread()
        veil = pygame.Surface((max(1, rect.w), max(1, rect.h)), pygame.SRCALPHA)
        veil.fill((color[0], color[1], color[2], 52))
        screen.blit(veil, rect.topleft)

    def _half_tag(self, screen, rect, name: str, value: float | None) -> None:
        import pygame

        require_main_thread()
        from .tabnum import cell_px

        text = format_half(name, value)
        cell = cell_px(14)
        pad = pygame.Surface((cell * len(text) + 10, 22), pygame.SRCALPHA)
        pad.fill((*CARD, 220))
        x = rect.x + 6
        y = rect.y + 6
        screen.blit(pad, (x, y))
        self._mixed(screen, text, x + 5, y + 2, "[3] {7}", VALUE)


def _series_mark(rows) -> tuple:
    if not rows:
        return (0, 0.0, 0.0)
    t, v = rows[-1]
    return (len(rows), round(float(t), 3), round(float(v), 3))


def _bins_mark(bins) -> tuple:
    if bins is None:
        return (0, 0.0)
    vals = np.asarray(bins, dtype=float).ravel()
    if vals.size == 0:
        return (0, 0.0)
    return (int(vals.size), round(float(vals.sum()), 3))


def _events_mark(events) -> tuple:
    if not events:
        return (0, "")
    last = events[-1]
    kind = last[1] if len(last) > 1 else ""
    return (len(events), str(kind))


def _camera_caption(error: str) -> tuple:
    """Short grey line for a dead onboard camera. The long text stays in the log."""
    text = "" if error is None else str(error)
    if not text:
        return "", False
    if text.startswith("камера борта"):
        return text, True
    if "Cannot reach" in text or "/camera.jpg" in text:
        return "камера борта недоступна (8088)", True
    return text, False


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        nxt = word if not cur else cur + " " + word
        if len(nxt) > width:
            if cur:
                lines.append(cur)
            cur = word
        else:
            cur = nxt
    if cur:
        lines.append(cur)
    return lines or [""]
