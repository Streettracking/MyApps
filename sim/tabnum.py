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
        os.path.join(windir, "Fonts", "consola.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Light.ttf",
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/cascadia-code/CascadiaMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        os.path.join(windir, "Fonts", "cour.ttf"),
    )
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def reset_fonts() -> None:
    """Drop fonts. They die at ``pygame.quit()`` and must be opened again."""
    _FONTS.clear()


_MIN_PX = 0


def set_min_px(px: int) -> None:
    """Smallest monospace size the trainer will open. ``0`` leaves the request alone."""
    global _MIN_PX
    _MIN_PX = max(0, int(px))


def load_font(size: int = 14):
    import pygame

    from .sdl_thread import require_main_thread

    require_main_thread()
    if not pygame.font.get_init():
        pygame.font.init()
    size = max(int(size), _MIN_PX)
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


def _ink_span(text: str):
    """``(start, count)`` of the non-space run, or ``None`` when the field is blank."""
    if not text or text.strip() == "":
        return None
    start = len(text) - len(text.lstrip(" "))
    count = len(text.rstrip(" ")) - start
    if count <= 0:
        return None
    return start, count


def blit_cells(screen, text: str, x: int, y: int, color, size: int = 14, rows=None, glyphs=None) -> int:
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
    if glyphs is not None:
        span = _ink_span(text)
        if span is not None:
            start, count = span
            glyphs.append(pygame.Rect(origin + start * cell, int(y), count * cell, height))
    return origin + cell * len(text)


# Gap between a label and the next slot. Slot x never follows a glyph's advance
# of a number: a number always occupies width × the width of "0".
SLOT_GAP = 8
EYES_STATUS = ("ОБА ВИДЯТ → ИДУ", "ОДИН ГЛАЗ → ДОВОРОТ", "НЕТ → ПОИСК")
FLY_STATUS = ("билатерально", "секторы")
SOUND_STATUS = ("звук выкл", "звук вкл", "звук недоступен")
RAW_STATUS = (
    "сырой выход: подход − избегание",
    "сырой выход: подход - избегание",
    "сырой выход: минус новизна",
    "сырой выход с борта",
)
SKIP_STATUS = ("", "лимит", "бокс в поле", "нет ответа YOLO", "—")


def _status_field(ui_font, text: str, samples) -> int:
    widths = [int(ui_font.size(sample)[0]) for sample in samples]
    if text:
        widths.append(int(ui_font.size(text)[0]))
    return max(widths) if widths else 0


def paint_slots(screen, x, y, slots, ui_font, value_color, label_color, rows=None, anchors=None, limit=None, glyphs=None) -> int:
    """Draw a readout as separate slots. Each slot's x is fixed for fixed slot text.

    ``("ui", text)`` is a label in the interface font. Its width is the width of
    that exact string, so it stays put.
    ``("num", text)`` is a monospace field. ``text`` is already right-aligned
    by ``signed``; the slot is ``len(text)`` cells wide no matter which digits
    are in it.
    ``("mono", text)`` is the same cell grid for a fixed-width token that is
    not a bare number (a duration, a gate).
    ``("status", text, samples)`` is left-aligned in a field sized by ``samples``,
    so a shorter phrase does not pull the next slot left.
    """
    import pygame

    cell = cell_px(14)
    cursor = int(x)
    yy = int(y)
    edge = None if limit is None else int(limit)
    for slot in slots:
        kind = slot[0]
        if kind == "ui":
            glyph = ui_font.render(slot[1], True, label_color)
            width = int(glyph.get_width())
            if edge is not None and cursor + width > edge:
                break
            if anchors is not None:
                anchors.append((kind, cursor, yy))
            screen.blit(glyph, (cursor, yy))
            if glyphs is not None and width > 0:
                glyphs.append(pygame.Rect(cursor, yy, width, int(glyph.get_height())))
            cursor += width + SLOT_GAP
            continue
        if kind in ("num", "mono"):
            text = slot[1]
            width = cell * max(len(text), 1)
            if edge is not None and cursor + width > edge:
                break
            if anchors is not None:
                anchors.append((kind, cursor, yy))
            blit_cells(screen, text, cursor, yy, value_color, size=14, rows=rows, glyphs=glyphs)
            cursor += width + SLOT_GAP
            continue
        if kind == "status":
            text = slot[1]
            samples = slot[2] if len(slot) > 2 else ()
            field = _status_field(ui_font, "", samples)
            if field <= 0:
                field = _status_field(ui_font, text, ())
            if edge is not None and cursor + int(field) > edge:
                break
            if anchors is not None:
                anchors.append((kind, cursor, yy))
            glyph = ui_font.render(text, True, value_color)
            clip = pygame.Surface((max(field, 1), max(glyph.get_height(), 1)), pygame.SRCALPHA)
            clip.blit(glyph, (0, 0))
            screen.blit(clip, (cursor, yy))
            if glyphs is not None and text:
                ink = min(int(glyph.get_width()), int(field))
                if ink > 0:
                    glyphs.append(pygame.Rect(cursor, yy, ink, int(glyph.get_height())))
            cursor += field + SLOT_GAP
            continue
        raise ValueError(kind)
    return cursor


def _pieces(line: str, pattern: str):
    """Split ``line`` by ``pattern``. ``{N}`` and ``[N]`` consume N characters."""
    i = 0
    j = 0
    while j < len(pattern):
        if pattern[j] in "{[":
            close = "}" if pattern[j] == "{" else "]"
            end = pattern.index(close, j)
            n = int(pattern[j + 1 : end])
            kind = "num" if pattern[j] == "{" else "text"
            yield kind, line[i : i + n]
            i += n
            j = end + 1
            continue
        k = j
        while k < len(pattern) and pattern[k] not in "{[":
            k += 1
        literal = pattern[j:k]
        if line[i : i + len(literal)] != literal:
            raise ValueError(line)
        i += len(literal)
        j = k
        text = literal.strip()
        if text:
            yield "ui", text
    if i != len(line):
        raise ValueError(line)


def paint_mixed(
    screen,
    x,
    y,
    line: str,
    pattern: str,
    ui_font,
    value_color,
    label_color,
    rows=None,
    anchors=None,
    status_samples=None,
    limit=None,
    glyphs=None,
) -> int:
    """Paint one formatted readout. A mismatch falls back to a single mono run."""
    try:
        slots = slots_of(line, pattern, status_samples, split_words=False)
    except (ValueError, IndexError):
        if anchors is not None:
            anchors.append(("fallback", int(x), int(y)))
        return blit_cells(screen, line, x, y, value_color, size=14, rows=rows, glyphs=glyphs)
    return paint_slots(
        screen,
        x,
        y,
        slots,
        ui_font,
        value_color,
        label_color,
        rows=rows,
        anchors=anchors,
        limit=limit,
        glyphs=glyphs,
    )


def slots_of(line: str, pattern: str, status_samples=None, split_words: bool = False):
    """Slots for one formatted line. Words of a label become their own columns."""
    pieces = list(_pieces(line, pattern))
    slots = []
    last = len(pieces) - 1
    for index, (kind, text) in enumerate(pieces):
        if kind == "ui":
            words = text.split() if split_words else [text]
            for word in words:
                if word:
                    slots.append(("ui", word))
        elif kind == "num":
            slots.append(("num", text))
        elif status_samples is not None and index == last:
            shown = text.rstrip()
            if shown:
                slots.append(("status", shown, status_samples))
        else:
            slots.append(("mono", text))
    return slots


def slot_width(slot, ui_font) -> int:
    """Width of one column. A number's width is the whole signed field, not its ink."""
    kind = slot[0]
    if kind == "ui":
        return int(ui_font.size(slot[1])[0])
    if kind in ("num", "mono"):
        return cell_px(14) * max(len(slot[1]), 1)
    if kind == "status":
        samples = slot[2] if len(slot) > 2 else ()
        field = _status_field(ui_font, "", samples)
        if field <= 0:
            field = _status_field(ui_font, slot[1], ())
        return int(field)
    raise ValueError(kind)


def wrap_slots(slots, width: int, ui_font):
    """Pack columns into rows that fit ``width``.

    ``None`` means some column is wider than the row, so the line must be
    hidden rather than cut through a number.
    """
    width = int(width)
    if width < 1:
        return None
    rows = []
    row = []
    used = 0
    for slot in slots:
        w = slot_width(slot, ui_font)
        extra = 0 if not row else SLOT_GAP
        if row and used + extra + w > width:
            rows.append(row)
            row = []
            used = 0
            extra = 0
        if w > width:
            return None
        row.append(slot)
        used += extra + w
    if row:
        rows.append(row)
    return rows


def paint_wrapped(
    screen,
    x,
    y,
    bottom,
    row_h,
    line: str,
    pattern: str,
    ui_font,
    value_color,
    label_color,
    rows=None,
    anchors=None,
    glyphs=None,
    status_samples=None,
    limit=None,
):
    """Draw a readout in fixed columns, wrapping whole columns onto the next row.

    Returns the y under the last row, or ``None`` when the line is hidden.
    A hidden line is not drawn at all: no column is sliced by the edge.
    """
    try:
        slots = slots_of(line, pattern, status_samples, split_words=True)
    except (ValueError, IndexError):
        return None
    edge = None if limit is None else int(limit)
    width = 10**9 if edge is None else edge - int(x)
    wrapped = wrap_slots(slots, width, ui_font)
    if not wrapped:
        return None
    step = int(row_h)
    if int(y) + step * len(wrapped) > int(bottom):
        return None
    cursor_y = int(y)
    for row_slots in wrapped:
        paint_slots(
            screen,
            x,
            cursor_y,
            row_slots,
            ui_font,
            value_color,
            label_color,
            rows=rows,
            anchors=anchors,
            limit=edge,
            glyphs=glyphs,
        )
        cursor_y += step
    return cursor_y


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


def lidar_metric_rows(fly_line: str, range_line: str) -> list:
    """Rows for the lidar column: ``(label, number, unit)``.

    Numbers are the stripped fixed fields. An empty fly or range line drops
    its rows. ``вектор`` is the smoothed sector's bearing in degrees.
    """
    import re

    rows = []
    fly = re.match(
        r"^муха L\s+(\S+)\s+R\s+(\S+)\s+Δ\s+(\S+)\s+z\s+(\S+)\s+(\S+)\s*$",
        str(fly_line or "").strip(),
    )
    rng = re.match(
        r"^дальн\s+(\S+)\s+м\s+вперёд\s+(\S+)\s+м\s+сектор\s+(\S+)→\s*(\S+)(?:\s+(.*))?$",
        str(range_line or "").strip(),
    )
    if fly:
        rows.append(("муха L", fly.group(1), ""))
        rows.append(("муха R", fly.group(2), ""))
        rows.append(("Δ", fly.group(3), ""))
        rows.append(("z", fly.group(4), ""))
        raw = rng.group(3) if rng else "-"
        smooth = rng.group(4) if rng else "-"
        rows.append(("вектор", _sector_bearing(raw, smooth), "°"))
        rows.append((fly.group(5), "", ""))
    if rng:
        gate = (rng.group(5) or "").strip()
        rows.append(("дальн", rng.group(1), "м"))
        rows.append(("вперёд", rng.group(2), "м"))
        rows.append(("сектор", "%s→%s" % (rng.group(3), rng.group(4)), gate))
    return rows


def _sector_bearing(raw: str, smooth: str) -> str:
    """Signed degrees of a sector center. Positive is left. A bad index is a dash."""
    import math

    from .map_marks import bin_angle
    from .raw_sense import N_AZ

    for token in (smooth, raw):
        text = str(token or "").strip()
        if text in ("", "-"):
            continue
        try:
            index = int(text)
        except ValueError:
            continue
        if 0 <= index < int(N_AZ):
            deg = float(bin_angle(index)) * 180.0 / math.pi
            return signed(deg, NUM_W, 1).strip()
    return signed(None).strip()


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
    return "ЗАПИСЬ %s %s" % (signed(saved, 5), mid)


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
