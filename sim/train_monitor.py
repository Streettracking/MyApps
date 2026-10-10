"""Live monitor for conspecific-recognition training.

Operator labels and ground truth drawn here never go back into the learner.
The window also owns the drive keys. They are reported to the caller; this
module does not send UDP itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hemifield import DEFAULT_OVERLAP
from .tabnum import (
    EYES_STATUS,
    FLY_STATUS,
    MODE_W,
    RAW_STATUS,
    SKIP_STATUS,
    SOUND_STATUS,
    format_box_tag,
    format_command,
    format_drift,
    format_familiar,
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
    skips_are_quiet,
)

WIN_W = 1920
WIN_H = 1080
MIN_SCALE = 0.75
MIN_CONTENT_W = int(round(WIN_W * MIN_SCALE))  # 1440
SCROLL_BAR = 14
SCROLL_TRACK = (214, 210, 204)
SCROLL_THUMB = (90, 82, 74)

# Light warm chrome. Peach is the only accent. Green and red stay semantic.
BG_TOP = (236, 234, 230)  # #ECEAE6
BG_BOT = (244, 242, 239)  # #F4F2EF
CARD = (247, 246, 244)  # #F7F6F4
CARD_ALPHA = 217  # ~0.85
LINE = (226, 224, 220)  # #E2E0DC
LABEL = (34, 34, 34)  # #222222
VALUE = (17, 17, 17)  # #111111
INK = VALUE
INK_DIM = (68, 68, 68)  # #444444 secondary text, no lighter
PEACH = (201, 120, 91)  # #C9785B
PEACH_SOFT = (227, 164, 135)  # #E3A487
GREEN = (141, 181, 150)  # #8DB596 узнаю / PAM
RED = (217, 136, 128)  # #D98880 PPL1
ESTOP = (184, 40, 36)
AMBER = PEACH
RING = (176, 174, 170)
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
    fullscreen_toggle: bool = False
    record_toggle: bool = False
    takeover: bool = False
    treat_down: bool = False
    punish_down: bool = False
    focused: bool = True


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


def scroll_limits(ww: int, wh: int) -> tuple:
    _scale, _lw, _lh, content_w, content_h = fit_window(ww, wh)
    return max(0, content_w - int(ww)), max(0, content_h - int(wh))


def scrollbar_geom(ww: int, wh: int, scroll_x: int = 0, scroll_y: int = 0) -> dict:
    """Tracks and thumbs in window pixels. Missing bars are ``None``."""
    import pygame

    ww = int(ww)
    wh = int(wh)
    max_x, max_y = scroll_limits(ww, wh)
    _scale, _lw, _lh, content_w, content_h = fit_window(ww, wh)
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


def logical_from_window(pos, ww: int, wh: int, scroll=(0, 0)) -> tuple:
    """Window pixels → canvas pixels, including scroll. A bar click is ``(-1, -1)``."""
    scale, logical_w, logical_h, content_w, content_h = fit_window(ww, wh)
    geom = scrollbar_geom(ww, wh, scroll[0], scroll[1])
    px, py = int(pos[0]), int(pos[1])
    for bar in (geom["track_x"], geom["track_y"]):
        if bar is not None and bar.collidepoint(px, py):
            return (-1, -1)
    sx, sy = geom["scroll"]
    x = px + sx
    y = py + sy
    if x < 0 or y < 0 or x >= content_w or y >= content_h:
        return (-1, -1)
    lx = int(round(x / scale))
    ly = int(round(y / scale))
    return (max(0, min(logical_w - 1, lx)), max(0, min(logical_h - 1, ly)))


def window_from_logical(pos, ww: int, wh: int, scroll=(0, 0)) -> tuple:
    """Canvas pixels → window pixels. Scroll shifts the point toward the origin."""
    scale, _lw, _lh, _cw, _ch = fit_window(ww, wh)
    geom = scrollbar_geom(ww, wh, scroll[0], scroll[1])
    sx, sy = geom["scroll"]
    wx = int(round(float(pos[0]) * scale)) - sx
    wy = int(round(float(pos[1]) * scale)) - sy
    return wx, wy


def monitor_layout(w: int, h: int) -> dict:
    """Rects for one logical frame. Width stays 1920; height grows with the window.

    The camera keeps a 16:9 slot. Leftover height goes to the lidar card and
    the right-hand panel. The OS window scales that frame uniformly.
    """
    m = 16
    header = 48
    footer = 56
    controls_h = 44
    controls_y = h - footer - controls_h - 10
    top = header + 8
    bottom = controls_y - 10
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
    buttons.update(
        place(
            [
                ("learn", 220),
                ("auto", 190),
                ("take", 180),
                ("steer", 210),
                ("treat", 190),
                ("estop", 150),
            ],
            m,
            controls_y,
            controls_h,
            w - m,
        )
    )
    buttons.update(
        {
            "cam": cam,
            "lid": lid,
            "panel": panel,
            "controls_y": controls_y,
            "footer_y": h - footer,
            "lidar_reset": (lid[0] + lid[2] + 12, lid[1], 200, 32),
            "lidar_fresh": (lid[0] + lid[2] + 12, lid[1] + 40, 240, 32),
            "stand_up": (lid[0] + lid[2] + 12, lid[1] + 84, 140, 32),
            "stand_down": (lid[0] + lid[2] + 160, lid[1] + 84, 120, 32),
            "recovery": (lid[0] + lid[2] + 12, lid[1] + 124, 160, 32),
        }
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

    arr = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
    return pygame.surfarray.make_surface(arr)


def _plot(screen, rect, series, y0, y1, color, font, label):
    import pygame

    pygame.draw.rect(screen, (255, 255, 255), rect, border_radius=12)
    pygame.draw.rect(screen, LINE, rect, 1, border_radius=12)
    if y1 <= y0:
        y1 = y0 + 1.0
    pts = []
    for i, (x, y) in enumerate(series):
        if len(series) == 1:
            u = 0.5
        else:
            u = i / (len(series) - 1)
        px = rect.x + 8 + int(u * (rect.w - 16))
        v = (float(y) - y0) / (y1 - y0)
        py = rect.bottom - 8 - int(np.clip(v, 0.0, 1.0) * (rect.h - 20))
        pts.append((px, py))
    if len(pts) >= 2:
        pygame.draw.lines(screen, color, False, pts, 2)
    elif pts:
        pygame.draw.circle(screen, color, pts[0], 3)
    screen.blit(font.render(label, True, PEACH), (rect.x + 10, rect.y + 6))


def _bars(screen, font, origin, values, width, title, color):
    import pygame

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
        pygame.draw.rect(screen, (236, 234, 230), pygame.Rect(rx, y, cell, 58), border_radius=3)
        pygame.draw.rect(screen, color, pygame.Rect(rx, base - h, cell, max(h, 1)), border_radius=3)
        lab = labels[i] if i < len(labels) else str(i)
        screen.blit(font.render(lab, True, LABEL), (rx, base + 2))
    return base + 18


class TrainMonitor:
    def __init__(self, title: str, fullscreen: bool = False):
        import pygame

        pygame.init()
        from .tabnum import reset_fonts

        reset_fonts()
        pygame.display.set_caption(title)
        self.fullscreen = bool(fullscreen)
        self.screen = None
        self.window = None
        self._window_size = None
        self._minimized = False
        self.brain_open = False
        self._open_display()
        self.font = _ui_font(15)
        self.font_sm = _ui_font(14)
        self.font_tiny = _ui_font(11)
        self.font_big = _ui_font(26)
        self.font_ind = _ui_font(18)
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

    def _apply_layout(self, height: int = WIN_H) -> None:
        self.layout = monitor_layout(WIN_W, int(height))
        self.cam_rect = self._rect("cam")
        self.lid_rect = self._rect("lid")
        self.panel_rect = self._rect("panel")
        self.estop_rect = self._rect("estop")
        self.treat_rect = self._rect("treat")
        self.lidar_reset_rect = self._rect("lidar_reset")
        self.lidar_fresh_rect = self._rect("lidar_fresh")
        self.learn_rect = self._rect("learn")
        self.auto_rect = self._rect("auto")
        self.take_rect = self._rect("take")
        self.steer_rect = self._rect("steer")
        self.record_rect = self._rect("record")
        self.teacher_rect = self._rect("teacher")
        self.brain_rect = self._rect("brain")
        self.stand_up_rect = self._rect("stand_up")
        self.stand_down_rect = self._rect("stand_down")
        self.recovery_rect = self._rect("recovery")

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

    def toggle_fullscreen(self) -> bool:
        """F11. Leaving fullscreen restores the window size from before it."""
        self.fullscreen = not self.fullscreen
        self._minimized = False
        self._open_display()
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
        return logical_from_window(pos, ww, wh, (self.scroll_x, self.scroll_y))

    def _mouse(self) -> tuple:
        import pygame

        return self._logical_pos(pygame.mouse.get_pos())

    def _on_resize(self, w: int, h: int) -> None:
        import pygame

        if w < 64 or h < 64:
            self._minimized = True
            return
        self._minimized = False
        if self.fullscreen:
            return
        if self.window is not None and self.window.get_size() == (w, h):
            self._window_size = (w, h)
            return
        self._window_size = (w, h)
        self.window = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self._clamp_scroll(w, h)

    def _clamp_scroll(self, ww: int | None = None, wh: int | None = None) -> None:
        if ww is None or wh is None:
            if self.window is None:
                return
            ww, wh = self.window.get_size()
        max_x, max_y = scroll_limits(int(ww), int(wh))
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
        return scrollbar_geom(int(size[0]), int(size[1]), self.scroll_x, self.scroll_y)

    def _sync_canvas(self, ww: int, wh: int) -> None:
        _scale, logical_w, logical_h, _cw, _ch = fit_window(ww, wh)
        self._clamp_scroll(ww, wh)
        if self.screen is None or self.screen.get_size() != (logical_w, logical_h):
            import pygame

            self.screen = pygame.Surface((logical_w, logical_h))
            self._apply_layout(logical_h)

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

        ww, wh = int(size[0]), int(size[1])
        _scale, logical_w, logical_h, content_w, content_h = fit_window(ww, wh)
        self._content = (0, 0, content_w, content_h)
        self._clamp_scroll(ww, wh)
        source = self.screen
        if source is None:
            source = pygame.Surface((logical_w, logical_h))
        sw, sh = source.get_size()
        target_w = content_w
        target_h = max(1, int(round(sh * (content_w / float(max(sw, 1))))))
        if source.get_size() != (target_w, target_h):
            scaled = pygame.transform.smoothscale(source, (target_w, target_h))
        else:
            scaled = source
        if (target_w, target_h) == (ww, wh) and self.scroll_x == 0 and self.scroll_y == 0:
            out = scaled
        else:
            out = pygame.Surface((ww, wh))
            out.fill(BG_BOT)
            out.blit(scaled, (-self.scroll_x, -self.scroll_y))
        return self._paint_scrollbars(out, ww, wh)

    def _paint_scrollbars(self, surface, ww: int, wh: int):
        import pygame

        geom = scrollbar_geom(ww, wh, self.scroll_x, self.scroll_y)
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
        if shift:
            self._scroll_by(-dy * step - dx * step, 0)
        else:
            self._scroll_by(-dx * step, -dy * step)

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
        geom = scrollbar_geom(ww, wh, self.scroll_x, self.scroll_y)
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

        inp = MonitorInput()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                inp.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
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
                    self.toggle_fullscreen()
                    inp.fullscreen_toggle = True
                elif event.key == pygame.K_u and not getattr(event, "repeat", False):
                    inp.record_toggle = True
                elif event.key in (pygame.K_KP_PLUS,) or getattr(event, "unicode", "") == "+":
                    inp.stand_up = True
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) or getattr(event, "unicode", "") == "-":
                    inp.stand_down = True
            elif event.type == pygame.MOUSEWHEEL:
                self._on_wheel(event)
            elif event.type == pygame.MOUSEMOTION and self._scroll_drag is not None:
                buttons = getattr(event, "buttons", (1, 0, 0))
                if buttons and buttons[0]:
                    self._drag_scroll(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._scroll_drag = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._begin_scroll_drag(event.pos):
                    continue
                pos = self._logical_pos(event.pos)
                if self.estop_rect.collidepoint(pos):
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
                elif self.brain_rect.collidepoint(pos):
                    inp.brain_toggle = True
                elif self.show_stand and self.stand_up_rect.collidepoint(pos):
                    inp.stand_up = True
                elif self.show_stand and self.stand_down_rect.collidepoint(pos):
                    inp.stand_down = True
                elif self.show_stand and self.recovery_rect.collidepoint(pos):
                    inp.recovery = True
            elif event.type == pygame.VIDEORESIZE:
                self._on_resize(int(event.w), int(event.h))
            elif event.type == getattr(pygame, "WINDOWMINIMIZED", -1):
                self._minimized = True
            elif event.type in (
                getattr(pygame, "WINDOWRESTORED", -2),
                getattr(pygame, "WINDOWMAXIMIZED", -3),
                getattr(pygame, "WINDOWSHOWN", -4),
            ):
                self._minimized = False
        inp.focused = bool(pygame.key.get_focused())
        if inp.focused:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                inp.steer_x += 1.0
            if keys[pygame.K_DOWN]:
                inp.steer_x -= 1.0
            if keys[pygame.K_LEFT]:
                inp.steer_z += 1.0
            if keys[pygame.K_RIGHT]:
                inp.steer_z -= 1.0
            inp.treat_down = bool(keys[pygame.K_t])
            inp.punish_down = bool(keys[pygame.K_x])
        return inp

    def save_screenshot(self, path: str) -> None:
        import pygame

        pygame.image.save(self.screen, path)

    def draw(self, view: MonitorView) -> None:
        import pygame

        if self.window is not None and not self._minimized:
            ww, wh = self.window.get_size()
            if ww >= 64 and wh >= 64:
                self._sync_canvas(ww, wh)
        screen = self.screen
        self._fill_bg(screen)
        self._fixed_rows = []
        self._anchors = []
        sm = self.font_sm
        screen.blit(self.font.render(view.title, True, LABEL), (16, 14))
        self._link_lamp(screen, view)
        self._brain_button(screen, view)
        self._teacher_button(screen, view)
        self._record_button(screen, view)

        cam_inner = self._frame(screen, self.cam_rect, view.camera, "камера", view.sensor_error)
        self.cam_inner = cam_inner
        if cam_inner is not None and view.learner == "mb":
            self._camera_overlap(screen, cam_inner, view)
            self._draw_det_boxes(screen, cam_inner, view.teacher_boxes)
            self._eye_plaques(screen, cam_inner, view)
        elif cam_inner is not None:
            self._camera_halves(screen, cam_inner, view)
        lid_title = view.lidar_mode or "карта лидара"
        lid_msg = "" if view.lidar is not None else view.lidar_hold
        inner = self._frame(screen, self.lid_rect, view.lidar, lid_title, lid_msg)
        self.lid_inner = inner
        if inner is not None and view.marks:
            self._draw_marks(screen, inner, view.marks)
        self._lidar_reset_button(screen)
        self._lidar_fresh_button(screen, view)
        note_x = self.lidar_fresh_rect.x
        note_y = (self.recovery_rect.bottom + 8) if view.onboard else (self.lidar_fresh_rect.bottom + 10)
        if view.lidar_warning:
            screen.blit(sm.render(view.lidar_warning[:42], True, PEACH), (note_x, note_y))
            note_y += 18
        if view.fly_line:
            self._mixed(
                screen,
                view.fly_line,
                note_x,
                note_y,
                "муха L {7}  R {7}  Δ {7}  z {7}  [14]",
                VALUE,
                status_samples=FLY_STATUS,
            )
            note_y += 20
        if view.range_line:
            self._mixed(
                screen,
                view.range_line,
                note_x,
                note_y,
                "дальн {7} м  вперёд {7} м  сектор {4}→{4}  {12}",
                VALUE,
            )

        panel = self.panel_rect
        base_panel_h = monitor_layout(WIN_W, WIN_H)["panel"][3]
        extra = max(0, panel.h - base_panel_h)
        journal_h = (108 if panel.h > 420 else 76) + extra // 2
        journal_h = min(journal_h, max(76, panel.h - 160))
        body = pygame.Rect(panel.x, panel.y, panel.w, max(140, panel.h - journal_h - 12))
        journal = pygame.Rect(panel.x, body.bottom + 12, panel.w, max(48, panel.bottom - (body.bottom + 12)))
        self.journal_rect = journal
        self._card(screen, body)
        self._card(screen, journal)
        log_n = max(5, (journal.h - 34) // 16)
        if view.learner == "mb" and self.flash_open:
            self._draw_mb_head(screen, view, body.x + 8)
            from .learn_flash import draw_learn_panel

            draw_learn_panel(
                screen,
                self.font,
                sm,
                pygame.Rect(body.x + 8, body.y + 78, body.w - 16, max(80, body.h - 96)),
                view.mb_layout,
                view.learn_flash,
                view.t,
                view.learn_flash_r,
            )
            log_n = max(3, (journal.h - 34) // 16)
        elif view.learner == "mb":
            self._draw_mb(screen, view, pygame.Rect(body.x + 12, body.y + 12, body.w - 24, body.h - 20))
        else:
            self._draw_hebb(screen, view, pygame.Rect(body.x + 12, body.y + 12, body.w - 24, body.h - 20))
        screen.blit(sm.render("журнал", True, PEACH), (journal.x + 16, journal.y + 10))
        y = journal.y + 30
        width_chars = max(24, journal.w // 8)
        self.log_rows = int(log_n)
        for line in view.log_lines[-log_n:]:
            screen.blit(sm.render(line[:width_chars], True, LABEL), (journal.x + 16, y))
            y += 16

        self._mode_buttons(screen, view)
        self.show_stand = bool(view.onboard)
        if view.onboard:
            self._stand_buttons(screen)
        footer = int(self.layout["footer_y"])
        pygame.draw.line(screen, LINE, (24, footer), (WIN_W - 24, footer), 1)
        focus = "окно в фокусе" if view.focused else "нажмите на окно — клавиши не читаются"
        from .tabnum import phrase as tab_phrase

        self._cells(screen, tab_phrase(view.udp_status, 78) + "  " + tab_phrase(focus, 42), 16, footer + 6, LABEL)
        self._cells(screen, format_command(view.last_command or "—"), 16, footer + 26, VALUE)
        screen.blit(sm.render(view.keys_hint, True, LABEL), (16, footer + 40))

        if view.learner == "mb":
            self._treat_button(screen, view, self._mouse())
            if view.recognized and not self._prev_rec and self.beep_on:
                self._play_beep()
            self._prev_rec = view.recognized
        else:
            self._prev_rec = False
        self._button(screen, self.estop_rect, "E-STOP", ESTOP, self.estop_rect.collidepoint(self._mouse()), label_color=ESTOP)
        self._present()
        self.clock.tick(30)

    def _cells(self, screen, text: str, x: int, y: int, color, size: int = 14) -> int:
        from .tabnum import blit_cells

        return blit_cells(screen, text, x, y, color, size=size, rows=self._fixed_rows)

    def _mixed(self, screen, line: str, x: int, y: int, pattern: str, value_color, label_color=None, status_samples=None) -> int:
        return paint_mixed(
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
        )

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

        if self._win_bg is None or self._win_bg.get_size() != tuple(size):
            bg = pygame.Surface((int(size[0]), int(size[1])))
            self._gradient(bg)
            self._win_bg = bg
        return self._win_bg

    def _fill_bg(self, screen) -> None:
        import pygame

        size = screen.get_size()
        if self._bg is None or self._bg.get_size() != size:
            bg = pygame.Surface(size)
            self._gradient(bg)
            self._bg = bg
        screen.blit(self._bg, (0, 0))

    def _card(self, screen, rect, radius: int = 16) -> None:
        import pygame

        radius = min(int(radius), rect.w // 2, rect.h // 2)
        shadow = pygame.Surface((rect.w + 28, rect.h + 32), pygame.SRCALPHA)
        for spread, alpha in ((12, 16), (7, 24), (3, 32)):
            pygame.draw.rect(
                shadow,
                (140, 120, 100, alpha),
                (14 - spread, 16 - spread, rect.w + spread * 2, rect.h + spread * 2),
                border_radius=radius + 4,
            )
        screen.blit(shadow, (rect.x - 14, rect.y - 10))
        face = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        pygame.draw.rect(face, (*CARD, CARD_ALPHA), face.get_rect(), border_radius=radius)
        pygame.draw.rect(face, (*LINE, 255), face.get_rect(), width=1, border_radius=radius)
        screen.blit(face, rect.topleft)

    def _inset(self, screen, rect, radius: int = 12) -> None:
        """White nested card. No shadow, so it stays flat inside an outer card."""
        import pygame

        if rect.w < 2 or rect.h < 2:
            return
        radius = max(0, min(int(radius), rect.w // 2, rect.h // 2))
        pygame.draw.rect(screen, (255, 255, 255), rect, border_radius=radius)
        pygame.draw.rect(screen, LINE, rect, 1, border_radius=radius)

    def _lamp(self, screen, center, color, radius: int = 6) -> None:
        import pygame

        x, y = int(center[0]), int(center[1])
        if color == LAMP_OFF or color is None:
            pygame.draw.circle(screen, RING, (x, y), radius, 1)
            return
        if color == PEACH:
            top, bottom = PEACH_SOFT, PEACH
        elif color == GREEN:
            top, bottom = (196, 220, 198), GREEN
        elif color == RED:
            top, bottom = (236, 196, 190), RED
        elif color == ESTOP:
            top, bottom = (214, 92, 84), ESTOP
        else:
            top, bottom = _mix((255, 255, 255), color, 0.35), color
        diameter = radius * 2
        disc = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        for row in range(diameter):
            pygame.draw.line(disc, (*_mix(top, bottom, row / max(diameter - 1, 1)), 255), (0, row), (diameter, row))
        mask = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), (radius, radius), radius)
        disc.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(disc, (x - radius, y - radius))

    def _button(self, screen, rect, text: str, lamp, hot: bool, label_color=None) -> None:
        import pygame

        self._card(screen, rect, radius=min(16, rect.h // 2))
        if hot:
            pygame.draw.rect(screen, PEACH, rect, 1, border_radius=min(16, rect.h // 2))
        x = rect.x + 12
        if lamp is not None:
            self._lamp(screen, (rect.x + 18, rect.centery), lamp, 6)
            x = rect.x + 32
        if not text:
            return
        label = self.font_sm.render(text, True, label_color or VALUE)
        screen.blit(label, (x, rect.centery - label.get_height() // 2))

    def _link_color(self, view: MonitorView):
        text = view.udp_status or ""
        low = text.lower()
        if "нет связи" in text or "error" in low:
            return RED
        if " ok" in low or low.endswith("ok"):
            return PEACH
        if view.onboard:
            return RED
        return LAMP_OFF

    def _link_lamp(self, screen, view: MonitorView) -> None:
        color = self._link_color(view)
        self._lamp(screen, (400, 22), color, 6)
        screen.blit(self.font_sm.render("борт", True, PEACH), (414, 12))

    def _mode_buttons(self, screen, view: MonitorView) -> None:
        import pygame

        mouse = self._mouse()
        learn_on = bool(view.learning_on)
        auto_on = bool(view.autonomy_on)
        grabbed = "ПЕРЕХВАТ" in (view.pilot_mode or "")
        self._button(
            screen,
            self.learn_rect,
            "СТОП ОБУЧЕНИЯ P" if learn_on else "СТАРТ ОБУЧЕНИЯ P",
            PEACH if learn_on else LAMP_OFF,
            self.learn_rect.collidepoint(mouse),
        )
        self._button(
            screen,
            self.auto_rect,
            "СТОП АВТО A" if auto_on else "СТАРТ АВТО A",
            PEACH if auto_on else LAMP_OFF,
            self.auto_rect.collidepoint(mouse),
        )
        self._button(
            screen,
            self.take_rect,
            "ПЕРЕХВАТ M",
            PEACH if grabbed else LAMP_OFF,
            self.take_rect.collidepoint(mouse),
        )
        if view.learner == "mb":
            bilateral = view.steer != "sectors"
            self._button(
                screen,
                self.steer_rect,
                "БИЛАТЕРАЛЬНО K" if bilateral else "СЕКТОРЫ K",
                None,
                self.steer_rect.collidepoint(mouse),
            )

    def _brain_button(self, screen, view: MonitorView) -> None:
        if view.learner != "mb":
            return
        self._button(
            screen,
            self.brain_rect,
            "МОЗГ 3D  J",
            PEACH if self.brain_open else LAMP_OFF,
            self.brain_rect.collidepoint(self._mouse()),
        )

    def _teacher_button(self, screen, view: MonitorView) -> None:
        if view.learner != "mb":
            return
        import pygame

        state = view.yolo_state or "выкл (нет обучения)"
        if state == "учит":
            lamp, label = PEACH, "учит  Y"
        elif state == "смотрит":
            lamp, label = PEACH, "смотрит  Y"
        elif state == "учитель не запущен":
            lamp, label = PEACH, "учитель не запущен"
        else:
            lamp, label = LAMP_OFF, "выкл (нет обучения)"
        self._button(screen, self.teacher_rect, label, lamp, self.teacher_rect.collidepoint(self._mouse()))

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
            zone = getattr(box, "zone", "") or ""
            self._mixed(
                screen,
                format_box_tag(zone, float(box.conf)),
                rect.x + 2,
                max(inner.y, rect.y - 16),
                "[3] {6}",
                color,
                label_color=color,
            )

    def _stand_buttons(self, screen) -> None:
        import pygame

        mouse = self._mouse()
        self._button(screen, self.stand_up_rect, "ВСТАТЬ +", None, self.stand_up_rect.collidepoint(mouse))
        self._button(screen, self.stand_down_rect, "ЛЕЧЬ −", None, self.stand_down_rect.collidepoint(mouse))
        self._button(screen, self.recovery_rect, "ПОДЪЁМ", None, self.recovery_rect.collidepoint(mouse))

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
                "ЗАПИСЬ {5} {10} U",
                VALUE,
            )
        else:
            self._button(screen, self.record_rect, "ЗАПИСЬ КАДРОВ  U", LAMP_OFF, hot)

    def _ensure_audio(self) -> bool:
        if self.audio_ok is not None:
            return self.audio_ok
        try:
            import pygame

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
            self._beep_sound = pygame.sndarray.make_sound(wave)
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
        self._button(screen, self.treat_rect, "ЛАКОМСТВО  T", lamp, self.treat_rect.collidepoint(mouse))
        if view.treat_flash:
            self._lamp(screen, (self.treat_rect.x + 16, self.treat_rect.centery), GREEN, 9)

    def _lidar_reset_button(self, screen) -> None:
        import pygame

        hot = self.lidar_reset_rect.collidepoint(self._mouse())
        self._button(screen, self.lidar_reset_rect, "СБРОС ЛИДАРА  C", None, hot)

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
        from .tabnum import SLOT_GAP, cell_px

        cap_x = self.lidar_fresh_rect.x + 32
        cap_y = self.lidar_fresh_rect.centery - 8
        if view.lidar_fresh_on:
            self._mixed(screen, caption, cap_x, cap_y, "свежий лидар {5} с", VALUE)
        else:
            screen.blit(self.font_sm.render(caption.strip(), True, VALUE), (cap_x, cap_y))
        v_x = (
            cap_x
            + self.font_sm.size("свежий лидар")[0]
            + SLOT_GAP
            + 5 * cell_px(14)
            + SLOT_GAP
            + self.font_sm.size("с")[0]
            + SLOT_GAP
        )
        screen.blit(self.font_sm.render("V", True, VALUE), (v_x, cap_y))

    def _draw_marks(self, screen, inner, marks) -> None:
        import pygame

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
                pygame.draw.polygon(overlay, (141, 181, 150, max(alpha // 3, 1)), (origin, left, right))
                pygame.draw.line(overlay, (141, 181, 150, alpha), origin, tip, 1)
            else:
                pygame.draw.line(overlay, (141, 181, 150, alpha), origin, tip, 1)
                pygame.draw.circle(overlay, (141, 181, 150, alpha), tip, 6)
                pygame.draw.circle(overlay, (196, 220, 198, alpha), tip, 6, 1)
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
            return PEACH
        return LAMP_OFF

    def _teacher_sees(self, view: MonitorView) -> tuple:
        """``(sees_L, sees_R)`` from box centres. Image right is Л, image left is П."""
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
            return PEACH
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

        Image left is П, image right is Л. The teacher lamp sits beside
        «узнаю»: green flash is PAM, red is PPL1, amber is a dog in that
        eye's field, an empty ring is no dog.
        """
        import pygame

        height = 36
        mid = inner.x + inner.w // 2
        veil = pygame.Surface((inner.w, height), pygame.SRCALPHA)
        veil.fill((*CARD, 210))
        screen.blit(veil, (inner.x, inner.y))
        sees_l, sees_r = self._teacher_sees(view)
        self.eye_lamps = {
            "П": self._eye_plaque(
                screen,
                pygame.Rect(inner.x, inner.y, mid - inner.x, height),
                "П",
                view.eye_r_recognized,
                view.eye_r_ready,
                view.eye_r_confidence,
                view.teacher_flash_r,
                sees_r,
            ),
            "Л": self._eye_plaque(
                screen,
                pygame.Rect(mid, inner.y, inner.right - mid, height),
                "Л",
                view.eye_l_recognized,
                view.eye_l_ready,
                view.eye_l_confidence,
                view.teacher_flash_l,
                sees_l,
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

    def _eyes_banner(self, screen, view: MonitorView, rx: int, col_w: int, y: int) -> int:
        self._lamp(
            screen,
            (rx + 10, y + 12),
            self._eye_lamp_color(view.eye_r_recognized, view.eye_r_ready, view.eye_r_confidence),
            5,
        )
        screen.blit(self.font_sm.render("П", True, PEACH), (rx + 20, y + 2))
        self._lamp(
            screen,
            (rx + 52, y + 12),
            self._eye_lamp_color(view.eye_l_recognized, view.eye_l_ready, view.eye_l_confidence),
            5,
        )
        screen.blit(self.font_sm.render("Л", True, PEACH), (rx + 62, y + 2))
        if view.eyes_line:
            self._mixed(
                screen,
                view.eyes_line,
                rx + 88,
                y,
                "R_L {7}  R_R {7}  [22]",
                VALUE,
                status_samples=EYES_STATUS,
            )
        if view.recog_line:
            screen.blit(self.font_sm.render(view.recog_line, True, INK_DIM), (rx + 88, y + 22))
        word = view.pilot_mode or "РУЧНОЕ"
        if word == "АВТОНОМИЯ" or "ПЕРЕХВАТ" in word:
            tone = PEACH
        elif word == "СТОП":
            tone = ESTOP
        else:
            tone = VALUE
        mode = format_pilot_mode(word)
        rest = format_pilot_rest(view.pilot_who, view.phase_ru, view.last_seen_side)
        self._cells(screen, mode, rx, y + 44, tone)
        from .tabnum import cell_px

        self._cells(screen, rest, rx + cell_px(14) * MODE_W, y + 44, INK_DIM)
        if view.pilot_hint:
            screen.blit(self.font_sm.render(view.pilot_hint, True, PEACH), (rx, y + 68))
            return y + 88
        return y + 70

    def _draw_mb_head(self, screen, view: MonitorView, rx: int) -> None:
        col_w = self.panel_rect.w if hasattr(self, "panel_rect") else WIN_W - rx - 16
        y = self.panel_rect.y if hasattr(self, "panel_rect") else 56
        if view.eyes_line or view.recog_line or view.learner == "mb":
            self._eyes_banner(screen, view, rx, col_w, y)
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

        sm = self.font_sm
        rx, col_w = panel.x, panel.w
        y = self._eyes_banner(screen, view, rx, col_w, panel.y)
        self._mixed(
            screen,
            format_raw(view.likeness, view.readout_caption),
            rx,
            y,
            "сырой MBON {7}  [36]",
            VALUE,
            status_samples=RAW_STATUS,
        )
        y += 20
        self._progress_box(screen, view, pygame.Rect(rx, y, col_w, 88))
        y += 96
        plot_w = (col_w - 8) // 2
        plot_h = 88
        _plot(screen, pygame.Rect(rx, y, plot_w, plot_h), view.peer_curve, 0.0, 1.0, GREEN, sm, "собака в кадре")
        _plot(
            screen,
            pygame.Rect(rx + plot_w + 8, y, col_w - plot_w - 8, plot_h),
            view.other_curve,
            0.0, 1.0,
            AMBER,
            sm,
            "нет собаки",
        )
        y += plot_h + 6
        drift_hi = max((v for _, v in view.drift_curve), default=1.0)
        if drift_hi <= 0:
            drift_hi = 1.0
        _plot(
            screen,
            pygame.Rect(rx, y, col_w, 52),
            view.drift_curve,
            0.0,
            drift_hi,
            INK_DIM,
            sm,
            "",
        )
        self._mixed(screen, format_drift(view.drift), rx + 8, y + 4, "дрейф KC→MBON  {7}", VALUE)
        y += 58
        self._dan_timeline(screen, view, pygame.Rect(rx, y, col_w, 72))
        y += 78
        self._kc_row(screen, view, rx, y, col_w)
        y += 84
        if view.teacher_counts:
            self._mixed(
                screen,
                view.teacher_counts,
                rx,
                y,
                "учитель: PAM_L {7}  PAM_R {7}  PPL1_L {7}  PPL1_R {7}",
                VALUE,
            )
            y += 20
        if view.teacher_skips:
            quiet = skips_are_quiet(view.teacher_skips)
            tone = LABEL if quiet else AMBER
            self._mixed(
                screen,
                view.teacher_skips,
                rx,
                y,
                "ложных узнаваний без наказания: {7}  [36]",
                tone,
                label_color=tone,
                status_samples=SKIP_STATUS,
            )
            y += 20
        return min(y + 4, panel.bottom - 96)

    def _progress_line(self, screen, line: str, x: int, y: int) -> None:
        if line.startswith("метки "):
            screen.blit(self.font_sm.render(line, True, LABEL), (x, y))
            return
        if line.startswith("Всего"):
            self._mixed(screen, line, x, y, "Всего за всё время: PAM {7} / PPL1 {7} (всего {7})", VALUE)
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
            )
            return
        if "знакомство" in line:
            self._mixed(screen, line, x, y, "[8]{12}  знакомство {7}", VALUE)
            return
        if line.startswith("D"):
            self._mixed(screen, line, x, y, "D−N сессия {6} {4}% {5}  всего {6} {4}% {5}", VALUE)
            return
        self._cells(screen, line, x, y, VALUE)

    def _progress_box(self, screen, view: MonitorView, rect) -> None:
        import pygame

        self._inset(screen, rect)
        screen.blit(self.font_sm.render("насколько обучен", True, PEACH), (rect.x + 12, rect.y + 8))
        if view.dan_mode == "familiarity":
            line1 = format_familiar("сессия", view.session_time, view.session_novelty)
            line2 = format_familiar("всего", view.total_time, view.total_novelty)
        else:
            line1 = format_lifetime(view.total_pam, view.total_ppl1, view.lifetime_known)
            line2 = format_session_counts(view.n_pam, view.n_ppl1, view.session_time, self._sound_label())
        line3 = format_metrics_line(
            view.session_sep,
            view.session_acc,
            view.session_labeled,
            view.total_sep,
            view.total_acc,
            view.total_labeled,
            bool(view.session_labeled or view.total_labeled),
        )
        self._progress_line(screen, line1, rect.x + 12, rect.y + 26)
        self._progress_line(screen, line2, rect.x + 12, rect.y + 46)
        self._progress_line(screen, line3, rect.x + 12, rect.y + 66)

    def _dan_timeline(self, screen, view: MonitorView, rect) -> None:
        import pygame

        self._inset(screen, rect)
        sm = self.font_sm
        title = "шкала DAN    T = лакомство PAM    X = наказание PPL1"
        screen.blit(sm.render(title, True, PEACH), (rect.x + 12, rect.y + 6))
        if view.t <= 30.0:
            t0 = 0.0
            t1 = max(view.t, 1.0)
            span_label = format_span(t1, False)
        else:
            t1 = view.t
            t0 = t1 - 30.0
            span_label = format_span(t1, True)
        inner = pygame.Rect(rect.x + 8, rect.y + 24, rect.w - 16, rect.h - 32)
        pygame.draw.line(screen, LINE, (inner.x, inner.centery), (inner.right, inner.centery), 1)
        shown = 0
        for t, kind in view.dan_events:
            if t < t0 or t > t1:
                continue
            u = (float(t) - t0) / max(t1 - t0, 1e-6)
            x = inner.x + int(u * max(inner.w - 1, 1))
            if kind == "PAM":
                pygame.draw.line(screen, GREEN, (x, inner.y + 2), (x, inner.centery - 1), 3)
            elif kind == "PPL1":
                pygame.draw.line(screen, RED, (x, inner.centery + 1), (x, inner.bottom - 2), 3)
            else:
                pygame.draw.line(screen, INK_DIM, (x, inner.y + 8), (x, inner.bottom - 8), 1)
            shown += 1
        if shown == 0:
            hint = "каждое лакомство появится здесь — собака в кадре, клавиша T"
            screen.blit(sm.render(hint, True, INK_DIM), (inner.x, inner.centery - 8))
        from .tabnum import cell_px

        self._cells(screen, span_label, rect.right - 8 - cell_px(14) * len(span_label), rect.bottom - 18, INK_DIM)

    def _kc_row(self, screen, view: MonitorView, x: int, y: int, width: int) -> None:
        import pygame

        sm = self.font_sm
        self._mixed(screen, format_kc(view.kc_on, view.kc_n), x, y, "активность KC  {7} из {7}", VALUE)
        bins = view.kc_bins
        if bins is None or len(bins) == 0:
            bins = np.zeros(48)
        vals = np.asarray(bins, dtype=float).ravel()
        vmax = float(vals.max()) if vals.size else 1.0
        if vmax < 1e-6:
            vmax = 1.0
        n = len(vals)
        gap = 2
        cell = max(4, (width - gap * (n - 1)) // n)
        base = y + 78
        top = y + 20
        for i, value in enumerate(vals):
            h = int(54 * max(float(value) / vmax, 0.0))
            rx = x + i * (cell + gap)
            pygame.draw.rect(screen, BG_TOP, pygame.Rect(rx, top, max(1, cell), 58))
            pygame.draw.rect(screen, PEACH, pygame.Rect(rx, base - h, max(1, cell), max(h, 1)))

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

    def _frame(self, screen, rect, image, title, error: str):
        import pygame

        self._card(screen, rect)
        screen.blit(self.font_sm.render(title, True, PEACH), (rect.x + 16, rect.y + 12))
        inner = None
        if image is not None and getattr(image, "size", 0):
            surf = _surf_from_rgb(image)
            avail = pygame.Rect(rect.x + 10, rect.y + 36, max(1, rect.w - 20), max(1, rect.h - 48))
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
            y = rect.y + 40
            for chunk in _wrap(error, 64):
                screen.blit(self.font_sm.render(chunk, True, RED), (rect.x + 10, y))
                y += 16
        return inner

    def _camera_overlap(self, screen, inner, view: MonitorView) -> None:
        """Binocular zone and one «УЗНАЮ» per eye. The camera array is not written.

        Image left is the robot's right eye (П). Image right is the left eye (Л).
        The shared band has a thin peach boundary on each side.
        """
        import pygame

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
        self._zone_tag(screen, inner.x, x0, inner, "П")
        if x1 > x0 + 18:
            self._zone_tag(screen, x0, x1, inner, "Л+П")
        self._zone_tag(screen, x1, inner.right, inner, "Л")
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

        Sector 0 is the left columns and the robot's right, so the picture's
        left half is П / R_R and the right half is Л / R_L.
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
        self._half_tag(screen, right, "Л", view.hemi_l if show else None)
        self._half_tag(screen, left, "П", view.hemi_r if show else None)

    def _wash(self, screen, rect, color: tuple[int, int, int]) -> None:
        import pygame

        veil = pygame.Surface((max(1, rect.w), max(1, rect.h)), pygame.SRCALPHA)
        veil.fill((color[0], color[1], color[2], 52))
        screen.blit(veil, rect.topleft)

    def _half_tag(self, screen, rect, name: str, value: float | None) -> None:
        import pygame

        from .tabnum import cell_px

        text = format_half(name, value)
        cell = cell_px(14)
        pad = pygame.Surface((cell * len(text) + 10, 22), pygame.SRCALPHA)
        pad.fill((*CARD, 220))
        x = rect.x + 6
        y = rect.y + 6
        screen.blit(pad, (x, y))
        self._mixed(screen, text, x + 5, y + 2, "[3] {7}", VALUE)


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
