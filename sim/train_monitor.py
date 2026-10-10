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
    MODE_W,
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
    skips_are_quiet,
)

WIN_W = 1920
WIN_H = 1080

# Austere chrome. Colour is reserved for a state: green, red, amber.
BG = (34, 34, 36)
PANEL = (42, 42, 44)
INK = (236, 236, 236)
INK_DIM = (168, 168, 170)
LINE = (96, 96, 100)
BTN = (48, 48, 50)
BTN_EDGE = (132, 132, 136)
GREEN = (64, 184, 96)
RED = (204, 62, 56)
AMBER = (214, 168, 64)
LAMP_OFF = (86, 86, 90)


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


def monitor_layout(w: int, h: int) -> dict:
    """Rects for one logical frame. The camera slot is at least half the width.

    pygame.SCALED then fits this 16:9 frame onto a 1920×1080 screen or a
    smaller window without rearranging the controls.
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

    header_x = w - 16 - 320 - 8 - 260
    buttons = place([("teacher", 320), ("record", 260)], header_x, 8, 32, w - 16)
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


def _surf_from_rgb(rgb: np.ndarray):
    import pygame

    arr = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
    return pygame.surfarray.make_surface(arr)


def _plot(screen, rect, series, y0, y1, color, font, label):
    import pygame

    pygame.draw.rect(screen, PANEL, rect)
    pygame.draw.rect(screen, LINE, rect, 1)
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
    screen.blit(font.render(label, True, color), (rect.x + 8, rect.y + 4))


def _bars(screen, font, origin, values, width, title, color):
    import pygame

    x, y = origin
    screen.blit(font.render(title, True, (200, 204, 214)), (x, y))
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
        pygame.draw.rect(screen, (32, 34, 42), pygame.Rect(rx, y, cell, 58))
        pygame.draw.rect(screen, color, pygame.Rect(rx, base - h, cell, max(h, 1)))
        lab = labels[i] if i < len(labels) else str(i)
        screen.blit(font.render(lab, True, (140, 144, 156)), (rx, base + 2))
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
        self._open_display()
        self.font = pygame.font.SysFont("dejavusans", 16)
        self.font_sm = pygame.font.SysFont("dejavusans", 15)
        self.font_big = pygame.font.SysFont("dejavusans", 28)
        self.font_ind = pygame.font.SysFont("dejavusans", 22)
        self._fixed_rows: list = []
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

    def _apply_layout(self) -> None:
        self.layout = monitor_layout(WIN_W, WIN_H)
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
        self.stand_up_rect = self._rect("stand_up")
        self.stand_down_rect = self._rect("stand_down")
        self.recovery_rect = self._rect("recovery")

    def _open_display(self) -> None:
        """Logical 1920×1080. SCALED fits that frame to the window or the screen."""
        import pygame

        flags = getattr(pygame, "SCALED", 0)
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
        else:
            flags |= pygame.RESIZABLE
        try:
            self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags)
        except pygame.error:
            self.fullscreen = False
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))

    def toggle_fullscreen(self) -> bool:
        self.fullscreen = not self.fullscreen
        self._open_display()
        return self.fullscreen

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
                elif event.key == pygame.K_F11 and not getattr(event, "repeat", False):
                    self.toggle_fullscreen()
                    inp.fullscreen_toggle = True
                elif event.key == pygame.K_u and not getattr(event, "repeat", False):
                    inp.record_toggle = True
                elif event.key in (pygame.K_KP_PLUS,) or getattr(event, "unicode", "") == "+":
                    inp.stand_up = True
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) or getattr(event, "unicode", "") == "-":
                    inp.stand_down = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.estop_rect.collidepoint(event.pos):
                    inp.estop = True
                elif self.treat_rect.collidepoint(event.pos):
                    inp.treat = True
                elif self.lidar_reset_rect.collidepoint(event.pos):
                    inp.lidar_reset = True
                elif self.lidar_fresh_rect.collidepoint(event.pos):
                    inp.lidar_toggle = True
                elif self.learn_rect.collidepoint(event.pos):
                    inp.pause_learn = True
                elif self.auto_rect.collidepoint(event.pos):
                    inp.autonomy_toggle = True
                elif self.take_rect.collidepoint(event.pos):
                    inp.takeover = True
                elif self.steer_rect.collidepoint(event.pos):
                    inp.steer_toggle = True
                elif self.record_rect.collidepoint(event.pos):
                    inp.record_toggle = True
                elif self.teacher_rect.collidepoint(event.pos):
                    inp.teacher_toggle = True
                elif self.show_stand and self.stand_up_rect.collidepoint(event.pos):
                    inp.stand_up = True
                elif self.show_stand and self.stand_down_rect.collidepoint(event.pos):
                    inp.stand_down = True
                elif self.show_stand and self.recovery_rect.collidepoint(event.pos):
                    inp.recovery = True
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

        screen = self.screen
        screen.fill(BG)
        self._fixed_rows = []
        sm = self.font_sm
        screen.blit(self.font.render(view.title, True, INK), (16, 12))
        self._link_lamp(screen, view)
        self._teacher_button(screen, view)
        self._record_button(screen, view)

        cam_inner = self._frame(screen, self.cam_rect, view.camera, "камера", view.sensor_error)
        if cam_inner is not None and view.learner == "mb":
            self._camera_overlap(screen, cam_inner, view)
            self._draw_det_boxes(screen, cam_inner, view.teacher_boxes)
            self._eye_plaques(screen, cam_inner, view)
        elif cam_inner is not None:
            self._camera_halves(screen, cam_inner, view)
        lid_title = view.lidar_mode or "карта лидара"
        lid_msg = "" if view.lidar is not None else view.lidar_hold
        inner = self._frame(screen, self.lid_rect, view.lidar, lid_title, lid_msg)
        if inner is not None and view.marks:
            self._draw_marks(screen, inner, view.marks)
        self._lidar_reset_button(screen)
        self._lidar_fresh_button(screen, view)
        note_x = self.lidar_fresh_rect.x
        note_y = (self.recovery_rect.bottom + 8) if view.onboard else (self.lidar_fresh_rect.bottom + 10)
        if view.lidar_warning:
            screen.blit(sm.render(view.lidar_warning[:42], True, AMBER), (note_x, note_y))
            note_y += 18
        if view.fly_line:
            self._cells(screen, view.fly_line, note_x, note_y, INK_DIM)
            note_y += 20
        if view.range_line:
            self._cells(screen, view.range_line, note_x, note_y, INK_DIM)

        panel = self.panel_rect
        log_n = 5
        if view.learner == "mb" and self.flash_open:
            self._draw_mb_head(screen, view, panel.x)
            from .learn_flash import draw_learn_panel

            draw_learn_panel(
                screen,
                self.font,
                sm,
                pygame.Rect(panel.x, panel.y + 72, panel.w, max(120, panel.h - 160)),
                view.mb_layout,
                view.learn_flash,
                view.t,
                view.learn_flash_r,
            )
            log_y = panel.bottom - 78
            log_n = 3
        elif view.learner == "mb":
            log_y = self._draw_mb(screen, view, panel)
        else:
            log_y = self._draw_hebb(screen, view, panel)
        screen.blit(sm.render("журнал", True, INK_DIM), (panel.x, log_y))
        y = log_y + 18
        width_chars = max(24, panel.w // 8)
        for line in view.log_lines[-log_n:]:
            screen.blit(sm.render(line[:width_chars], True, INK_DIM), (panel.x, y))
            y += 16

        self._mode_buttons(screen, view)
        self.show_stand = bool(view.onboard)
        if view.onboard:
            self._stand_buttons(screen)
        footer = int(self.layout["footer_y"])
        pygame.draw.line(screen, LINE, (0, footer), (WIN_W, footer), 1)
        focus = "окно в фокусе" if view.focused else "нажмите на окно — клавиши не читаются"
        from .tabnum import phrase as tab_phrase

        self._cells(screen, tab_phrase(view.udp_status, 78) + "  " + tab_phrase(focus, 42), 16, footer + 6, INK_DIM)
        self._cells(screen, format_command(view.last_command or "—"), 16, footer + 26, INK_DIM)
        screen.blit(sm.render(view.keys_hint, True, INK_DIM), (16, footer + 40))

        if view.learner == "mb":
            self._treat_button(screen, view, pygame.mouse.get_pos())
            if view.recognized and not self._prev_rec and self.beep_on:
                self._play_beep()
            self._prev_rec = view.recognized
        else:
            self._prev_rec = False
        self._button(screen, self.estop_rect, "E-STOP", RED, self.estop_rect.collidepoint(pygame.mouse.get_pos()))
        pygame.display.flip()
        self.clock.tick(30)

    def _cells(self, screen, text: str, x: int, y: int, color, size: int = 14) -> int:
        from .tabnum import blit_cells

        return blit_cells(screen, text, x, y, color, size=size, rows=self._fixed_rows)

    def _sound_label(self) -> str:
        if self.beep_on and self.audio_ok is False:
            return "звук недоступен"
        if self.beep_on:
            return "звук вкл"
        return "звук выкл"

    def _lamp(self, screen, center, color, radius: int = 6) -> None:
        import pygame

        x, y = int(center[0]), int(center[1])
        pygame.draw.circle(screen, (22, 22, 24), (x, y), radius + 2)
        pygame.draw.circle(screen, color, (x, y), radius)

    def _button(self, screen, rect, text: str, lamp, hot: bool) -> None:
        import pygame

        edge = INK if hot else BTN_EDGE
        pygame.draw.rect(screen, BTN, rect, border_radius=4)
        pygame.draw.rect(screen, edge, rect, 1, border_radius=4)
        x = rect.x + 8
        if lamp is not None:
            self._lamp(screen, (rect.x + 16, rect.centery), lamp, 6)
            x = rect.x + 28
        label = self.font_sm.render(text, True, INK)
        if label.get_width() > rect.w - (x - rect.x) - 8:
            label = self.font_sm.render(text, True, INK)
        screen.blit(label, (x, rect.centery - label.get_height() // 2))

    def _link_color(self, view: MonitorView):
        text = view.udp_status or ""
        low = text.lower()
        if "нет связи" in text or "error" in low:
            return AMBER
        if " ok" in low or low.endswith("ok"):
            return GREEN
        if view.onboard:
            return AMBER
        return LAMP_OFF

    def _link_lamp(self, screen, view: MonitorView) -> None:
        color = self._link_color(view)
        self._lamp(screen, (400, 22), color, 6)
        screen.blit(self.font_sm.render("борт", True, INK), (414, 12))

    def _mode_buttons(self, screen, view: MonitorView) -> None:
        import pygame

        mouse = pygame.mouse.get_pos()
        learn_on = bool(view.learning_on)
        auto_on = bool(view.autonomy_on)
        grabbed = "ПЕРЕХВАТ" in (view.pilot_mode or "")
        self._button(
            screen,
            self.learn_rect,
            "СТОП ОБУЧЕНИЯ P" if learn_on else "СТАРТ ОБУЧЕНИЯ P",
            GREEN if learn_on else LAMP_OFF,
            self.learn_rect.collidepoint(mouse),
        )
        self._button(
            screen,
            self.auto_rect,
            "СТОП АВТО A" if auto_on else "СТАРТ АВТО A",
            GREEN if auto_on else LAMP_OFF,
            self.auto_rect.collidepoint(mouse),
        )
        self._button(
            screen,
            self.take_rect,
            "ПЕРЕХВАТ M",
            AMBER if grabbed else LAMP_OFF,
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

    def _teacher_button(self, screen, view: MonitorView) -> None:
        if view.learner != "mb":
            return
        import pygame

        state = view.yolo_state or "выкл (нет обучения)"
        if state == "учит":
            lamp, label = GREEN, "учит  Y"
        elif state == "смотрит":
            lamp, label = AMBER, "смотрит  Y"
        elif state == "учитель не запущен":
            lamp, label = AMBER, "учитель не запущен"
        else:
            lamp, label = LAMP_OFF, "выкл (нет обучения)"
        self._button(screen, self.teacher_rect, label, lamp, self.teacher_rect.collidepoint(pygame.mouse.get_pos()))

    def _draw_det_boxes(self, screen, inner, boxes) -> None:
        """Screen-only YOLO frames. The camera array is not written."""
        import pygame

        if not boxes:
            return
        color = AMBER
        for box in boxes:
            x0 = inner.x + int(round(float(box.x0) * inner.w))
            x1 = inner.x + int(round(float(box.x1) * inner.w))
            y0 = inner.y + int(round(float(box.y0) * inner.h))
            y1 = inner.y + int(round(float(box.y1) * inner.h))
            rect = pygame.Rect(min(x0, x1), min(y0, y1), max(1, abs(x1 - x0)), max(1, abs(y1 - y0)))
            pygame.draw.rect(screen, color, rect, 2)
            zone = getattr(box, "zone", "") or ""
            self._cells(screen, format_box_tag(zone, float(box.conf)), rect.x + 2, max(inner.y, rect.y - 16), color)

    def _stand_buttons(self, screen) -> None:
        import pygame

        mouse = pygame.mouse.get_pos()
        self._button(screen, self.stand_up_rect, "ВСТАТЬ +", None, self.stand_up_rect.collidepoint(mouse))
        self._button(screen, self.stand_down_rect, "ЛЕЧЬ −", None, self.stand_down_rect.collidepoint(mouse))
        self._button(screen, self.recovery_rect, "ПОДЪЁМ", None, self.recovery_rect.collidepoint(mouse))

    def _record_button(self, screen, view: MonitorView) -> None:
        import pygame

        hot = self.record_rect.collidepoint(pygame.mouse.get_pos())
        if view.record_on:
            text = format_record(view.record_saved, view.record_bytes, view.record_idle)
            lamp = AMBER
            self._button(screen, self.record_rect, "", lamp, hot)
            self._cells(screen, text, self.record_rect.x + 28, self.record_rect.centery - 9, INK)
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

        hot = self.lidar_reset_rect.collidepoint(pygame.mouse.get_pos())
        self._button(screen, self.lidar_reset_rect, "СБРОС ЛИДАРА  C", None, hot)

    def _lidar_fresh_button(self, screen, view: MonitorView) -> None:
        import pygame

        hot = self.lidar_fresh_rect.collidepoint(pygame.mouse.get_pos())
        if view.lidar_fresh_on:
            caption = view.lidar_mode or "свежий лидар"
            lamp = GREEN
        else:
            caption = view.lidar_mode or "лидар копится"
            lamp = AMBER
        self._button(screen, self.lidar_fresh_rect, "", lamp, hot)
        self._cells(screen, "%s  V" % caption, self.lidar_fresh_rect.x + 28, self.lidar_fresh_rect.centery - 9, INK)

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
                pygame.draw.polygon(overlay, (80, 210, 120, alpha // 3), (origin, left, right))
                pygame.draw.line(overlay, (180, 255, 190, alpha), origin, tip, 2)
            else:
                pygame.draw.line(overlay, (180, 255, 190, alpha), origin, tip, 2)
                pygame.draw.circle(overlay, (70, 200, 110, alpha), tip, 7)
                pygame.draw.circle(overlay, (230, 255, 230, alpha), tip, 7, 2)
            drawn.append((tip, alpha, mark))
        for i, (tip, _alpha, mark) in enumerate(drawn):
            tag = self.font_sm.render(f"{mark.percent:.0f}%", True, (230, 255, 220))
            overlay.blit(tag, (tip[0] + 12, tip[1] - 18 + i * 16))
        screen.blit(overlay, inner.topleft)

    def _eye_lamp_color(self, recognized: bool, ready: bool, percent: float, flash: str) -> tuple:
        if flash == "ppl1":
            return RED
        if flash == "pam":
            return GREEN
        if recognized:
            return GREEN
        if ready and float(percent) > 0.0:
            return AMBER
        return LAMP_OFF

    def _eye_plaques(self, screen, inner, view: MonitorView) -> None:
        """Lamp and a word on the preview. Image left is П, image right is Л."""
        import pygame

        height = 36
        mid = inner.x + inner.w // 2
        veil = pygame.Surface((inner.w, height), pygame.SRCALPHA)
        veil.fill((28, 28, 30, 170))
        screen.blit(veil, (inner.x, inner.y))
        self._eye_plaque(
            screen,
            pygame.Rect(inner.x, inner.y, mid - inner.x, height),
            "П",
            view.eye_r_recognized,
            view.eye_r_ready,
            view.eye_r_confidence,
            view.teacher_flash_r,
        )
        self._eye_plaque(
            screen,
            pygame.Rect(mid, inner.y, inner.right - mid, height),
            "Л",
            view.eye_l_recognized,
            view.eye_l_ready,
            view.eye_l_confidence,
            view.teacher_flash_l,
        )

    def _eye_plaque(self, screen, rect, name: str, recognized: bool, ready: bool, percent: float, flash: str = "") -> None:
        color = self._eye_lamp_color(recognized, ready, percent, flash)
        radius = 9 if flash in ("pam", "ppl1") else 7
        self._lamp(screen, (rect.x + 18, rect.centery), color, radius)
        word = "УЗНАЮ" if recognized or flash else "—"
        self._cells(
            screen,
            format_plaque(name, word),
            rect.x + 34,
            rect.centery - 9,
            INK,
        )

    def _eyes_banner(self, screen, view: MonitorView, rx: int, col_w: int, y: int) -> int:
        self._lamp(
            screen,
            (rx + 10, y + 12),
            self._eye_lamp_color(view.eye_r_recognized, view.eye_r_ready, view.eye_r_confidence, view.teacher_flash_r),
            5,
        )
        screen.blit(self.font_sm.render("П", True, INK), (rx + 20, y + 2))
        self._lamp(
            screen,
            (rx + 52, y + 12),
            self._eye_lamp_color(view.eye_l_recognized, view.eye_l_ready, view.eye_l_confidence, view.teacher_flash_l),
            5,
        )
        screen.blit(self.font_sm.render("Л", True, INK), (rx + 62, y + 2))
        if view.eyes_line:
            self._cells(screen, view.eyes_line, rx + 88, y, INK)
        if view.recog_line:
            screen.blit(self.font_sm.render(view.recog_line, True, INK_DIM), (rx + 88, y + 22))
        word = view.pilot_mode or "РУЧНОЕ"
        if word == "АВТОНОМИЯ":
            tone = GREEN
        elif "ПЕРЕХВАТ" in word:
            tone = AMBER
        elif word == "СТОП":
            tone = RED
        else:
            tone = INK
        mode = format_pilot_mode(word)
        rest = format_pilot_rest(view.pilot_who, view.phase_ru, view.last_seen_side)
        self._cells(screen, mode, rx, y + 44, tone)
        from .tabnum import cell_px

        self._cells(screen, rest, rx + cell_px(14) * MODE_W, y + 44, INK_DIM)
        if view.pilot_hint:
            screen.blit(self.font_sm.render(view.pilot_hint, True, AMBER), (rx, y + 68))
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
        self._cells(screen, format_raw(view.likeness, view.readout_caption), rx, y, INK_DIM)
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
        self._cells(screen, format_drift(view.drift), rx + 8, y + 4, INK_DIM)
        y += 58
        self._dan_timeline(screen, view, pygame.Rect(rx, y, col_w, 72))
        y += 78
        self._kc_row(screen, view, rx, y, col_w)
        y += 84
        if view.teacher_counts:
            self._cells(screen, view.teacher_counts, rx, y, INK_DIM)
            y += 20
        if view.teacher_skips:
            quiet = skips_are_quiet(view.teacher_skips)
            self._cells(screen, view.teacher_skips, rx, y, INK_DIM if quiet else AMBER)
            y += 20
        return min(y + 4, panel.bottom - 96)

    def _progress_box(self, screen, view: MonitorView, rect) -> None:
        import pygame

        pygame.draw.rect(screen, PANEL, rect)
        pygame.draw.rect(screen, LINE, rect, 1)
        screen.blit(self.font_sm.render("насколько обучен", True, INK_DIM), (rect.x + 8, rect.y + 4))
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
        self._cells(screen, line1, rect.x + 8, rect.y + 22, INK)
        self._cells(screen, line2, rect.x + 8, rect.y + 42, INK_DIM)
        self._cells(screen, line3, rect.x + 8, rect.y + 62, INK_DIM)

    def _dan_timeline(self, screen, view: MonitorView, rect) -> None:
        import pygame

        pygame.draw.rect(screen, PANEL, rect)
        pygame.draw.rect(screen, LINE, rect, 1)
        sm = self.font_sm
        title = "шкала DAN    T = лакомство PAM    X = наказание PPL1"
        screen.blit(sm.render(title, True, INK_DIM), (rect.x + 8, rect.y + 4))
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
        self._cells(screen, format_kc(view.kc_on, view.kc_n), x, y, INK_DIM)
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
            pygame.draw.rect(screen, (28, 28, 30), pygame.Rect(rx, top, cell, 58))
            pygame.draw.rect(screen, INK_DIM, pygame.Rect(rx, base - h, cell, max(h, 1)))

    def _draw_hebb(self, screen, view: MonitorView, panel) -> int:
        import pygame

        sm = self.font_sm
        rx, col_w = panel.x, panel.w
        y = panel.y
        from .tabnum import cell_px, signed

        self._cells(screen, signed(view.likeness, 7, 2), rx, y, INK, size=26)
        screen.blit(sm.render("похожесть    слой сравнения, не KC→MBON", True, INK_DIM), (rx + cell_px(26) * 8, y + 8))
        self._cells(screen, format_hebb_proto(view.learned_match, view.n_self, view.n_other), rx, y + 36, INK_DIM)
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
            self._cells(screen, format_purity(float(purity)), rx, y + 400, INK_DIM)
        return min(y + 430, panel.bottom - 96)

    def _frame(self, screen, rect, image, title, error: str):
        import pygame

        pygame.draw.rect(screen, PANEL, rect)
        pygame.draw.rect(screen, LINE, rect, 1)
        screen.blit(self.font_sm.render(title, True, INK_DIM), (rect.x + 8, rect.y + 4))
        inner = None
        if image is not None and getattr(image, "size", 0):
            surf = _surf_from_rgb(image)
            avail = pygame.Rect(rect.x + 6, rect.y + 24, max(1, rect.w - 12), max(1, rect.h - 30))
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
            y = rect.y + 28
            for chunk in _wrap(error, 64):
                screen.blit(self.font_sm.render(chunk, True, RED), (rect.x + 10, y))
                y += 16
        return inner

    def _camera_overlap(self, screen, inner, view: MonitorView) -> None:
        """Binocular zone and one «УЗНАЮ» per eye. The camera array is not written.

        Image left is the robot's right eye (П). Image right is the left eye (Л).
        The shared band is cyan, with a boundary on each side.
        """
        import pygame

        from .hemifield import overlap_bands

        lo, hi = overlap_bands(view.overlap)
        x0 = inner.x + int(round(lo * inner.w))
        x1 = inner.x + int(round(hi * inner.w))
        if x1 > x0 + 1:
            pygame.draw.line(screen, INK, (x0, inner.y + 1), (x0, inner.bottom - 1), 1)
            pygame.draw.line(screen, INK, (x1, inner.y + 1), (x1, inner.bottom - 1), 1)
        else:
            mid = inner.x + inner.w // 2
            pygame.draw.line(screen, INK, (mid, inner.y + 1), (mid, inner.bottom - 1), 1)
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
        pad.fill((28, 28, 30, 180))
        px = inner.right - pad_w - 6
        py = inner.bottom - pad_h - 6
        screen.blit(pad, (px, py))
        self._cells(screen, pct, px + 5, py + 2, INK)

    def _zone_tag(self, screen, x_lo: int, x_hi: int, inner, name: str) -> None:
        import pygame

        if x_hi - x_lo < 16:
            return
        label = self.font_sm.render(name, True, INK)
        pad = pygame.Surface((label.get_width() + 10, label.get_height() + 4), pygame.SRCALPHA)
        pad.fill((10, 12, 16, 176))
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
        pygame.draw.line(screen, INK, (mid, inner.y + 1), (mid, inner.bottom - 1), 1)
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
        pad.fill((10, 12, 16, 176))
        x = rect.x + 6
        y = rect.y + 6
        screen.blit(pad, (x, y))
        self._cells(screen, text, x + 5, y + 2, INK)


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
