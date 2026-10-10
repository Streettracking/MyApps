"""Fixed-width readouts for the trainer.

Numbers sit in monospace cells, right-aligned, with the sign always shown.
A label is a run of cells at a fixed column, so a new digit cannot push it.
Status phrases occupy a field of their own and do not shove the next label.
"""

from __future__ import annotations

import os

NUM_W = 7
EYES_W = 22
REASON_W = 36
MODE_W = 18
WHO_W = 10
PHASE_W = 20
SEEN_W = 10

_FONTS: dict[int, object] = {}


def signed(value, width: int = NUM_W, decimals: int = 0) -> str:
    """Right-aligned field. ``None`` is a dash. Overflow keeps the sign."""
    if value is None:
        return "-".rjust(width)
    text = "{:+.{d}f}".format(float(value), d=int(decimals))
    if len(text) <= width:
        return text.rjust(width)
    sign = text[0] if text[0] in "+-" else "+"
    return (sign + text[-(width - 1) :])[:width]


def phrase(text, width: int) -> str:
    """Left-aligned status field. Longer text is cut, shorter text is padded."""
    raw = "" if text is None else str(text)
    if len(raw) > width:
        raw = raw[:width]
    return raw.ljust(width)


def fit(text, width: int, align: str = "left") -> str:
    raw = "" if text is None else str(text)
    if len(raw) > width:
        raw = raw[:width]
    if align == "right":
        return raw.rjust(width)
    return raw.ljust(width)


def label_index(text: str, token: str) -> int:
    return text.index(token)


def skips_are_quiet(line: str) -> bool:
    """True when the unpunished-false count is zero and no reason is shown."""
    tail = line.split(":", 1)[-1]
    return signed(0) in tail and tail.replace(signed(0), "").strip() == ""


def mono_path() -> str | None:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = (
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/cascadia-code/CascadiaMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        os.path.join(windir, "Fonts", "consola.ttf"),
        os.path.join(windir, "Fonts", "cour.ttf"),
    )
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def reset_fonts() -> None:
    """Drop fonts. They die at ``pygame.quit()`` and must be opened again."""
    _FONTS.clear()


def load_font(size: int = 14):
    import pygame

    if not pygame.font.get_init():
        pygame.font.init()
    cached = _FONTS.get(int(size))
    if cached is not None:
        return cached
    path = mono_path()
    if path is None:
        path = (
            pygame.font.match_font("jetbrainsmono")
            or pygame.font.match_font("cascadiamono")
            or pygame.font.match_font("consolas")
            or pygame.font.match_font("liberationmono")
            or pygame.font.match_font("monospace")
        )
    font = pygame.font.Font(path, int(size)) if path else pygame.font.SysFont("monospace", int(size))
    _FONTS[int(size)] = font
    return font


def cell_px(size: int = 14) -> int:
    return int(load_font(size).size("0")[0])


def blit_cells(screen, text: str, x: int, y: int, color, size: int = 14, rows=None) -> int:
    """Draw ``text`` so each character occupies one cell. Spaces stay empty."""
    import pygame

    font = load_font(size)
    cell = int(font.size("0")[0])
    height = int(font.get_height())
    origin = int(x)
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        glyph = font.render(ch, True, color)
        box = pygame.Surface((cell, height), pygame.SRCALPHA)
        box.blit(glyph, ((cell - glyph.get_width()) // 2, 0))
        screen.blit(box, (origin + i * cell, int(y)))
    if rows is not None:
        rows.append((origin, int(y), text, int(size), cell, height))
    return origin + cell * len(text)


def format_eyes_line(r_l, r_r, text: str) -> str:
    return "R_L %s  R_R %s  %s" % (signed(r_l), signed(r_r), phrase(text, EYES_W))


def format_fly_line(r_l, r_r, diff, z, name: str) -> str:
    return "муха L %s  R %s  Δ %s  z %s  %s" % (
        signed(r_l),
        signed(r_r),
        signed(diff),
        signed(z, NUM_W, 2),
        phrase(name, 14),
    )


def format_range_line(dist, forward, sector, sector_smooth, gate: str) -> str:
    return "дальн %s м  вперёд %s м  сектор %s→%s  %s" % (
        signed(dist, NUM_W, 2),
        signed(forward, NUM_W, 2),
        signed(sector, 4),
        signed(sector_smooth, 4),
        phrase(gate, 12),
    )


def format_counts(pam_l, pam_r, ppl1_l, ppl1_r) -> str:
    return "учитель: PAM_L %s  PAM_R %s  PPL1_L %s  PPL1_R %s" % (
        signed(pam_l),
        signed(pam_r),
        signed(ppl1_l),
        signed(ppl1_r),
    )


def format_skips(n: int, reason: str) -> str:
    return "ложных узнаваний без наказания: %s  %s" % (signed(n), phrase(reason, REASON_W))


def format_duration(seconds: float, width: int = 12) -> str:
    from .mb_confidence import fmt_duration

    return fit(fmt_duration(seconds), width, "right")


def format_raw(likeness, caption: str) -> str:
    return "сырой MBON %s  %s" % (signed(likeness), phrase(caption, 36))


def format_drift(value) -> str:
    return "дрейф KC→MBON  %s" % signed(value, NUM_W, 1)


def format_kc(on, n) -> str:
    return "активность KC  %s из %s" % (signed(on), signed(n))


def format_span(seconds: float, windowed: bool) -> str:
    if windowed:
        return "    30 с"
    return signed(seconds, 6) + " с"


def format_lifetime(pam, ppl1, known: bool) -> str:
    if not known:
        return "Всего за всё время: PAM %s / PPL1 %s (всего %s)" % (signed(None), signed(None), signed(None))
    return "Всего за всё время: PAM %s / PPL1 %s (всего %s)" % (
        signed(pam),
        signed(ppl1),
        signed(int(pam) + int(ppl1)),
    )


def format_session_counts(pam, ppl1, seconds: float, sound: str) -> str:
    return "За этот запуск: PAM %s / PPL1 %s  %s  %s" % (
        signed(pam),
        signed(ppl1),
        format_duration(seconds),
        phrase(sound, 15),
    )


def format_familiar(prefix: str, seconds: float, novelty) -> str:
    return "%s%s  знакомство %s" % (phrase(prefix, 8), format_duration(seconds), signed(novelty))


def format_metric(sep, acc, n: int) -> str:
    percent = None if acc is None or int(n) < 1 else float(acc) * 100.0
    count = int(n) if int(n) >= 1 else None
    return "%s %s%% %s" % (signed(sep, 6), signed(percent, 4), signed(count, 5))


def format_metrics_line(session_sep, session_acc, session_n, total_sep, total_acc, total_n, labeled: bool) -> str:
    if not labeled:
        return "метки D/N нет — разделение и точность ждут клавиши D и N"
    return "D−N сессия %s  всего %s" % (
        format_metric(session_sep, session_acc, session_n),
        format_metric(total_sep, total_acc, total_n),
    )


def format_pilot_mode(mode: str) -> str:
    return phrase(mode or "РУЧНОЕ", MODE_W)


def format_pilot_rest(who: str, phase: str, seen_side: str) -> str:
    if seen_side == "L":
        seen = "видели Л"
    elif seen_side == "R":
        seen = "видели П"
    else:
        seen = ""
    return "  ведёт: %s  %s  %s" % (phrase(who, WHO_W), phrase(phase, PHASE_W), phrase(seen, SEEN_W))


def format_lidar_caption(fresh_on: bool, interval: float) -> str:
    shown = "свежий лидар %s с" % signed(interval, 5, 1)
    if fresh_on:
        return shown
    return phrase("лидар копится", len(shown))


def format_record(saved, nbytes: int, idle: bool) -> str:
    from .frame_record import format_disk

    mid = phrase("жми T/X", 10) if idle else fit(format_disk(nbytes), 10, "right")
    return "ЗАПИСЬ %s %s U" % (signed(saved, 5), mid)


def format_overlap(frac: float) -> str:
    return "перекрытие %s%%" % signed(round(float(frac) * 100.0), 4)


def format_half(name: str, value) -> str:
    return "%s %s" % (phrase(name, 3), signed(value))


def format_box_tag(zone: str, conf) -> str:
    return "%s %s" % (phrase(zone, 3), signed(conf, 6, 2))


def format_command(cmd: str) -> str:
    return "команда  %s" % phrase(cmd or "—", 28)


def format_hebb_proto(match, n_self, n_other) -> str:
    return "к прототипу %s  свой %s  прочее %s" % (signed(match, NUM_W, 2), signed(n_self), signed(n_other))


def format_purity(purity) -> str:
    return "чистота %s / %s" % (signed(purity, 4), signed(3, 2))


def format_flash_syn(n_syn, sum_abs, kc_n) -> str:
    return "синапсов %s  Σ|Δw| %s  KC активны %s" % (signed(n_syn), signed(sum_abs, NUM_W, 2), signed(kc_n))


def format_flash_out(before, after) -> str:
    shift = None if before is None or after is None else float(after) - float(before)
    return "выход MBON %s → %s  сдвиг %s" % (signed(before), signed(after), signed(shift))


def format_flash_row(delta: float, label: str) -> str:
    decimals = 0 if abs(float(delta)) >= 10.0 else 1
    return "%s  %s" % (signed(delta, NUM_W, decimals), phrase(label, 18))


def format_flash_pair(name: str, n_syn, sum_abs, before, after) -> str:
    if n_syn is None:
        return "%s %s" % (phrase(name, 4), phrase("—", 42))
    return "%s синапсов %s  Σ|Δw| %s  %s→%s" % (
        phrase(name, 4),
        signed(n_syn),
        signed(sum_abs, NUM_W, 2),
        signed(before),
        signed(after),
    )


def format_plaque(name: str, word: str) -> str:
    return "%s  %s" % (phrase(name, 1), phrase(word, 6))
