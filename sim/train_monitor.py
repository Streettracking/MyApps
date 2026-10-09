"""Live monitor for conspecific-recognition training.

Operator labels and ground truth drawn here never go back into the learner.
The window also owns the drive keys. They are reported to the caller; this
module does not send UDP itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mb_confidence import fmt_duration

WIN_W = 1280
WIN_H = 860


@dataclass
class MonitorInput:
    quit: bool = False
    steer_x: float = 0.0
    steer_z: float = 0.0
    stop: bool = False
    stand_up: bool = False
    stand_down: bool = False
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
    learning_on: bool = True
    autonomy_on: bool = False


def _surf_from_rgb(rgb: np.ndarray):
    import pygame

    arr = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
    return pygame.surfarray.make_surface(arr)


def _plot(screen, rect, series, y0, y1, color, font, label):
    import pygame

    pygame.draw.rect(screen, (22, 24, 30), rect)
    pygame.draw.rect(screen, (48, 52, 64), rect, 1)
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
    def __init__(self, title: str):
        import pygame

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.font = pygame.font.SysFont("dejavusans", 16)
        self.font_sm = pygame.font.SysFont("dejavusans", 15)
        self.font_big = pygame.font.SysFont("dejavusans", 36)
        self.font_ind = pygame.font.SysFont("dejavusans", 28)
        self.clock = pygame.time.Clock()
        self.estop_rect = pygame.Rect(WIN_W - 188, WIN_H - 56, 168, 40)
        self.treat_rect = pygame.Rect(WIN_W - 430, WIN_H - 56, 220, 40)
        self.lidar_reset_rect = pygame.Rect(16, 588, 200, 34)
        self.lidar_fresh_rect = pygame.Rect(224, 588, 236, 34)
        self.learn_rect = pygame.Rect(16, WIN_H - 148, 210, 46)
        self.auto_rect = pygame.Rect(236, WIN_H - 148, 230, 46)
        self.take_rect = pygame.Rect(476, WIN_H - 148, 230, 46)
        self.beep_on = False
        self.flash_open = False
        self.audio_ok: bool | None = None
        self._beep_sound = None
        self._prev_rec = False

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
        screen.fill((16, 18, 24))
        font = self.font
        sm = self.font_sm

        screen.blit(font.render(view.title, True, (236, 236, 240)), (16, 10))
        tag = "обучение на паузе" if view.paused else "обучение"
        screen.blit(sm.render(f"t={view.t:6.1f} с    {tag}    {view.mode_label}", True, (180, 186, 198)), (16, 32))

        cam_rect = pygame.Rect(16, 56, 460, 210)
        lid_rect = pygame.Rect(16, 278, 460, 300)
        self._frame(screen, cam_rect, view.camera, "камера", view.sensor_error)
        lid_title = view.lidar_mode or "карта лидара"
        lid_msg = "" if view.lidar is not None else view.lidar_hold
        inner = self._frame(screen, lid_rect, view.lidar, lid_title, lid_msg)
        if inner is not None and view.marks:
            self._draw_marks(screen, inner, view.marks)
        self._lidar_reset_button(screen)
        self._lidar_fresh_button(screen, view)
        if view.lidar_warning:
            screen.blit(sm.render(view.lidar_warning[:78], True, (230, 176, 96)), (16, 626))

        rx = 492
        log_n = 6
        if view.learner == "mb" and self.flash_open:
            self._draw_mb_head(screen, view, rx)
            from .learn_flash import draw_learn_panel

            draw_learn_panel(
                screen,
                font,
                sm,
                pygame.Rect(rx, 88, WIN_W - rx - 16, 512),
                view.mb_layout,
                view.learn_flash,
                view.t,
            )
            log_y = 608
            log_n = 3
        elif view.learner == "mb":
            self._draw_mb(screen, view, rx)
            log_y = 548
        else:
            self._draw_hebb(screen, view, rx)
            log_y = 560
        screen.blit(sm.render("журнал", True, (200, 204, 214)), (rx, log_y))
        y = log_y + 18
        for line in view.log_lines[-log_n:]:
            screen.blit(sm.render(line[:78], True, (186, 190, 200)), (rx, y))
            y += 16

        self._mode_buttons(screen, view)
        pygame.draw.rect(screen, (12, 12, 16), pygame.Rect(0, WIN_H - 72, WIN_W, 72))
        focus = "окно в фокусе" if view.focused else "нажмите на окно — клавиши не читаются"
        screen.blit(sm.render(view.udp_status + "    " + focus, True, (220, 220, 220)), (16, WIN_H - 64))
        cmd = view.last_command or "—"
        screen.blit(sm.render("команда  " + cmd[:90], True, (170, 176, 188)), (16, WIN_H - 44))
        screen.blit(sm.render(view.keys_hint, True, (140, 146, 158)), (16, WIN_H - 24))

        mouse = pygame.mouse.get_pos()
        hot = self.estop_rect.collidepoint(mouse)
        color = (210, 48, 48) if hot else (160, 32, 32)
        pygame.draw.rect(screen, color, self.estop_rect, border_radius=4)
        label = font.render("E-STOP", True, (255, 240, 240))
        screen.blit(label, label.get_rect(center=self.estop_rect.center))
        if view.learner == "mb":
            self._treat_button(screen, view, mouse)
            if view.recognized and not self._prev_rec and self.beep_on:
                self._play_beep()
            self._prev_rec = view.recognized
        else:
            self._prev_rec = False
        pygame.display.flip()
        self.clock.tick(30)

    def _mode_buttons(self, screen, view: MonitorView) -> None:
        import pygame

        mouse = pygame.mouse.get_pos()
        learn_on = bool(view.learning_on)
        auto_on = bool(view.autonomy_on)
        grabbed = "ПЕРЕХВАТ" in (view.pilot_mode or "")
        self._pill(
            screen,
            self.learn_rect,
            "СТОП ОБУЧЕНИЯ P" if learn_on else "СТАРТ ОБУЧЕНИЯ P",
            (32, 118, 72) if learn_on else (58, 62, 72),
            self.learn_rect.collidepoint(mouse),
        )
        self._pill(
            screen,
            self.auto_rect,
            "СТОП АВТО A" if auto_on else "СТАРТ АВТО A",
            (32, 96, 150) if auto_on else (58, 62, 72),
            self.auto_rect.collidepoint(mouse),
        )
        self._pill(
            screen,
            self.take_rect,
            "ПЕРЕХВАТ M",
            (168, 112, 36) if grabbed else (92, 58, 32),
            self.take_rect.collidepoint(mouse),
        )
        word = view.pilot_mode or "РУЧНОЕ"
        if word == "АВТОНОМИЯ":
            color = (86, 214, 128)
        elif "ПЕРЕХВАТ" in word:
            color = (232, 176, 72)
        elif word == "СТОП":
            color = (232, 84, 76)
        else:
            color = (214, 218, 224)
        title = self.font_ind.render(word, True, color)
        screen.blit(title, (720, WIN_H - 146))
        who = self.font.render(f"ведёт: {view.pilot_who}", True, (186, 192, 204))
        screen.blit(who, (720, WIN_H - 114))
        if view.pilot_hint:
            hint = self.font_sm.render(view.pilot_hint, True, (232, 176, 72))
            screen.blit(hint, (720, WIN_H - 94))

    def _pill(self, screen, rect, text: str, color: tuple[int, int, int], hot: bool) -> None:
        import pygame

        fill = tuple(min(255, c + 28) for c in color) if hot else color
        pygame.draw.rect(screen, fill, rect, border_radius=6)
        label = self.font.render(text, True, (248, 248, 246))
        screen.blit(label, label.get_rect(center=rect.center))

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
        import pygame

        hot = self.treat_rect.collidepoint(mouse)
        if view.treat_flash:
            color = (46, 196, 96)
        elif hot:
            color = (28, 150, 72)
        else:
            color = (16, 92, 48)
        pygame.draw.rect(screen, color, self.treat_rect, border_radius=4)
        pygame.draw.rect(screen, (180, 255, 200), self.treat_rect, 2, border_radius=4)
        label = self.font.render("ЛАКОМСТВО  T", True, (245, 255, 248))
        screen.blit(label, label.get_rect(center=self.treat_rect.center))

    def _lidar_reset_button(self, screen) -> None:
        import pygame

        hot = self.lidar_reset_rect.collidepoint(pygame.mouse.get_pos())
        color = (52, 78, 112) if hot else (32, 48, 72)
        pygame.draw.rect(screen, color, self.lidar_reset_rect, border_radius=4)
        pygame.draw.rect(screen, (170, 200, 230), self.lidar_reset_rect, 1, border_radius=4)
        label = self.font_sm.render("СБРОС ЛИДАРА  C", True, (230, 236, 244))
        screen.blit(label, label.get_rect(center=self.lidar_reset_rect.center))

    def _lidar_fresh_button(self, screen, view: MonitorView) -> None:
        import pygame

        hot = self.lidar_fresh_rect.collidepoint(pygame.mouse.get_pos())
        if view.lidar_fresh_on:
            color = (24, 92, 64) if not hot else (36, 130, 88)
            ink = (220, 255, 230)
            caption = view.lidar_mode or "свежий лидар"
        else:
            color = (72, 56, 32) if not hot else (110, 82, 40)
            ink = (255, 228, 190)
            caption = "лидар копится"
        pygame.draw.rect(screen, color, self.lidar_fresh_rect, border_radius=4)
        pygame.draw.rect(screen, (190, 210, 190), self.lidar_fresh_rect, 1, border_radius=4)
        label = self.font_sm.render(f"{caption}  V", True, ink)
        screen.blit(label, label.get_rect(center=self.lidar_fresh_rect.center))

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

    def _draw_mb_head(self, screen, view: MonitorView, rx: int) -> None:
        sm = self.font_sm
        col_w = WIN_W - rx - 16
        if view.recognized:
            phrase, tone = "УЗНАЮ СОРОДИЧА", (90, 230, 130)
        else:
            phrase, tone = "НЕ УЗНАЮ", (150, 154, 162)
        screen.blit(self.font_ind.render(phrase, True, tone), (rx, 48))
        if view.confidence_ready:
            pct = f"{view.confidence:.0f}%"
        else:
            pct = "…"
        pct_s = self.font_big.render(pct, True, tone)
        screen.blit(pct_s, (rx + col_w - pct_s.get_width(), 46))

    def _draw_mb(self, screen, view: MonitorView, rx: int) -> None:
        import pygame

        sm = self.font_sm
        col_w = WIN_W - rx - 16
        self._draw_mb_head(screen, view, rx)
        raw = f"сырой MBON {view.likeness:+.0f}    {view.readout_caption}"
        screen.blit(sm.render(raw[:88], True, (150, 156, 168)), (rx, 84))
        self._progress_box(screen, view, pygame.Rect(rx, 106, col_w, 78))

        plot_w = (col_w - 8) // 2
        _plot(screen, pygame.Rect(rx, 192, plot_w, 96), view.peer_curve, 0.0, 1.0, (80, 200, 120), sm, "собака в кадре")
        _plot(
            screen,
            pygame.Rect(rx + plot_w + 8, 192, col_w - plot_w - 8, 96),
            view.other_curve,
            0.0, 1.0,
            (214, 164, 72),
            sm,
            "нет собаки",
        )
        drift_hi = max((v for _, v in view.drift_curve), default=1.0)
        if drift_hi <= 0:
            drift_hi = 1.0
        _plot(
            screen,
            pygame.Rect(rx, 294, col_w, 58),
            view.drift_curve,
            0.0,
            drift_hi,
            (120, 170, 220),
            sm,
            f"дрейф KC→MBON    {view.drift:.1f}",
        )
        self._dan_timeline(screen, view, pygame.Rect(rx, 358, col_w, 78))
        self._kc_row(screen, view, rx, 442, col_w)

    def _progress_box(self, screen, view: MonitorView, rect) -> None:
        import pygame

        pygame.draw.rect(screen, (20, 26, 24), rect)
        pygame.draw.rect(screen, (60, 90, 70), rect, 1)
        sm = self.font_sm
        screen.blit(sm.render("насколько обучен", True, (190, 220, 196)), (rect.x + 8, rect.y + 4))
        if view.dan_mode == "familiarity":
            teach = f"знакомство {view.session_novelty} / всего {view.total_novelty}"
        else:
            teach = f"лакомств {view.n_pam} / всего {view.total_pam}    наказаний {view.n_ppl1} / {view.total_ppl1}"
        line1 = f"сессия  {fmt_duration(view.session_time)}    {teach}"
        line2 = f"всего    {fmt_duration(view.total_time)}    лакомств {view.total_pam}"
        if view.session_labeled or view.total_labeled:
            ses = _metric(view.session_sep, view.session_acc, view.session_labeled)
            tot = _metric(view.total_sep, view.total_acc, view.total_labeled)
            line3 = f"D−N  сессия {ses}    всего {tot}"
        else:
            line3 = "метки D/N нет — разделение и точность ждут клавиши D и N"
        if self.beep_on and self.audio_ok is False:
            sound = "звук недоступен"
        elif self.beep_on:
            sound = "звук вкл"
        else:
            sound = "звук выкл"
        screen.blit(sm.render(line1[:96], True, (220, 224, 214)), (rect.x + 8, rect.y + 24))
        screen.blit(sm.render((line2 + "    " + sound)[:96], True, (200, 206, 198)), (rect.x + 8, rect.y + 42))
        screen.blit(sm.render(line3[:100], True, (176, 186, 178)), (rect.x + 8, rect.y + 58))

    def _dan_timeline(self, screen, view: MonitorView, rect) -> None:
        import pygame

        pygame.draw.rect(screen, (18, 28, 22), rect)
        pygame.draw.rect(screen, (70, 140, 90), rect, 2)
        sm = self.font_sm
        title = "шкала DAN    T = лакомство PAM    X = наказание PPL1"
        screen.blit(sm.render(title, True, (190, 235, 200)), (rect.x + 8, rect.y + 4))
        if view.t <= 30.0:
            t0 = 0.0
            t1 = max(view.t, 1.0)
            span_label = f"{t1:.0f} s"
        else:
            t1 = view.t
            t0 = t1 - 30.0
            span_label = "30 s"
        inner = pygame.Rect(rect.x + 8, rect.y + 24, rect.w - 16, rect.h - 32)
        pygame.draw.line(screen, (60, 80, 68), (inner.x, inner.centery), (inner.right, inner.centery), 1)
        shown = 0
        for t, kind in view.dan_events:
            if t < t0 or t > t1:
                continue
            u = (float(t) - t0) / max(t1 - t0, 1e-6)
            x = inner.x + int(u * max(inner.w - 1, 1))
            if kind == "PAM":
                pygame.draw.line(screen, (70, 230, 120), (x, inner.y + 2), (x, inner.centery - 1), 3)
            elif kind == "PPL1":
                pygame.draw.line(screen, (230, 80, 70), (x, inner.centery + 1), (x, inner.bottom - 2), 3)
            else:
                pygame.draw.line(screen, (90, 150, 220), (x, inner.y + 8), (x, inner.bottom - 8), 1)
            shown += 1
        if shown == 0:
            hint = "каждое лакомство появится здесь — собака в кадре, клавиша T"
            screen.blit(sm.render(hint, True, (140, 170, 150)), (inner.x, inner.centery - 8))
        screen.blit(sm.render(span_label, True, (120, 140, 128)), (rect.right - 48, rect.bottom - 16))

    def _kc_row(self, screen, view: MonitorView, x: int, y: int, width: int) -> None:
        import pygame

        sm = self.font_sm
        screen.blit(sm.render(f"активность KC    {view.kc_on} из {view.kc_n}", True, (200, 204, 214)), (x, y))
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
            pygame.draw.rect(screen, (28, 32, 40), pygame.Rect(rx, top, cell, 58))
            pygame.draw.rect(screen, (150, 190, 120), pygame.Rect(rx, base - h, cell, max(h, 1)))

    def _draw_hebb(self, screen, view: MonitorView, rx: int) -> None:
        import pygame

        sm = self.font_sm
        col_w = WIN_W - rx - 16
        screen.blit(self.font_big.render(f"{view.likeness:.2f}", True, (230, 210, 120)), (rx, 50))
        screen.blit(sm.render("похожесть    слой сравнения, не KC→MBON", True, (200, 204, 214)), (rx + 180, 58))
        screen.blit(
            sm.render(f"к прототипу {view.learned_match:.2f}    свой {view.n_self}    прочее {view.n_other}", True, (180, 186, 198)),
            (rx + 180, 80),
        )
        screen.blit(sm.render(view.operator_label[:86], True, (150, 156, 168)), (rx + 180, 100))
        plot_w = (col_w - 8) // 2
        _plot(screen, pygame.Rect(rx, 128, plot_w, 140), view.peer_curve, 0.0, 1.0, (80, 200, 120), sm, "собака в кадре")
        _plot(
            screen,
            pygame.Rect(rx + plot_w + 8, 128, col_w - plot_w - 8, 140),
            view.other_curve,
            0.0, 1.0,
            (214, 164, 72),
            sm,
            "нет собаки",
        )
        inv = [(float(row.get("t", 0.0)), float(row.get("invariance", 0.0))) for row in view.metric_curve]
        _plot(screen, pygame.Rect(rx, 276, col_w, 88), inv, -1.0, 1.0, (180, 160, 230), sm, "инвариантность  собака стоит − дистрактор быстрый")
        half = (col_w - 12) // 2
        _bars(screen, sm, (rx, 376), view.proto_self, half, "прототип сородича", (90, 180, 120))
        _bars(screen, sm, (rx + half + 12, 376), view.proto_other, half, "прототип прочего", (190, 140, 70))
        purity = view.metric_curve[-1].get("purity") if view.metric_curve else None
        if purity is not None:
            screen.blit(sm.render(f"чистота {float(purity):.0f} / 3", True, (180, 186, 198)), (rx, 500))

    def _frame(self, screen, rect, image, title, error: str):
        import pygame

        pygame.draw.rect(screen, (10, 12, 16), rect)
        pygame.draw.rect(screen, (48, 52, 64), rect, 1)
        screen.blit(self.font_sm.render(title, True, (180, 186, 198)), (rect.x + 8, rect.y + 4))
        inner = None
        if image is not None and getattr(image, "size", 0):
            surf = _surf_from_rgb(image)
            inner = rect.inflate(-8, -24)
            inner.y += 16
            scaled = pygame.transform.smoothscale(surf, (inner.w, inner.h))
            screen.blit(scaled, inner.topleft)
        if error:
            y = rect.y + 28
            for chunk in _wrap(error, 52):
                screen.blit(self.font_sm.render(chunk, True, (230, 120, 110)), (rect.x + 10, y))
                y += 16
        return inner


def _metric(sep: float | None, acc: float | None, n: int) -> str:
    if sep is None or acc is None or n < 1:
        return "—"
    return f"{sep:+.0f}, точность {acc * 100:.0f}% ({n})"


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
