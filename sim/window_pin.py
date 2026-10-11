"""Pin a separate pygame window above the others.

Windows: one ``SetWindowPos`` per click, on the HWND from
``pygame.display.get_wm_info()['window']``. Linux and a headless driver
return False and do not raise. The call does not recreate the display and
does not borrow an SDL window wrapper.
"""

from __future__ import annotations

import math

# Same peach as the trainer accent. The idle clip is a quiet gray.
PIN_ON = (201, 120, 91)
PIN_OFF = (132, 132, 134)
PIN_TIP = "поверх окон"

# Insert-after values. Passed as pointer-sized handles, not a truncated int.
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0001
SWP_NOSIZE = 0x0002
SWP_NOACTIVATE = 0x0010


def set_always_on_top(pinned: bool) -> bool:
    """Pin or unpin the current display window. One call, main thread only.

    Returns True when Windows accepted the change. Anywhere else, and when
    the handle is missing, returns False.
    """
    from .sdl_thread import require_main_thread

    require_main_thread()
    import os

    if os.name != "nt":
        return False
    try:
        import pygame

        info = pygame.display.get_wm_info()
        hwnd = info.get("window") if isinstance(info, dict) else None
        if not hwnd:
            return False
        import ctypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        set_pos = user32.SetWindowPos
        set_pos.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        )
        set_pos.restype = ctypes.c_int
        after = HWND_TOPMOST if pinned else HWND_NOTOPMOST
        flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
        ok = set_pos(ctypes.c_void_p(int(hwnd)), ctypes.c_void_p(after), 0, 0, 0, 0, flags)
        return bool(ok)
    except Exception:
        return False


def pin_button_rect(size) -> tuple:
    """Top-right corner of the pygame surface, not the OS caption."""
    width = int(size[0])
    return (max(0, width - 34), 8, 26, 26)


def hit_pin(size, pos) -> bool:
    x, y, w, h = pin_button_rect(size)
    px, py = int(pos[0]), int(pos[1])
    return x <= px < x + w and y <= py < y + h


def consume_pin(pinned: bool, *, key=None, repeat: bool = False, button: int = 0, pos=None, size=None):
    """One T press or one left click on the clip.

    Returns ``(handled, pinned)``. A key repeat is ignored, so the window
    is moved once per press. A handled click must not start a drag.
    """
    if key is not None:
        import pygame

        if int(key) == pygame.K_t and not repeat:
            return True, not bool(pinned)
        return False, bool(pinned)
    if int(button) == 1 and pos is not None and size is not None and hit_pin(size, pos):
        return True, not bool(pinned)
    return False, bool(pinned)


def _arc_points(rect, a0: float, a1: float, steps: int = 18) -> list:
    """Ellipse samples in the same direction as ``pygame.draw.arc`` (y up)."""
    cx = rect.x + rect.w / 2.0
    cy = rect.y + rect.h / 2.0
    rx = rect.w / 2.0
    ry = rect.h / 2.0
    pts = []
    steps = max(2, int(steps))
    for i in range(steps + 1):
        t = a0 + (a1 - a0) * i / steps
        pts.append((cx + rx * math.cos(t), cy - ry * math.sin(t)))
    return pts


def _smooth_line(surface, start, end, color, width: int = 1) -> None:
    import pygame

    pygame.draw.line(surface, color, start, end, width)
    pygame.draw.aaline(surface, color, start, end)


def _smooth_arc(surface, rect, a0: float, a1: float, color, width: int = 1) -> None:
    import pygame

    pygame.draw.arc(surface, color, rect, a0, a1, width)
    pygame.draw.aalines(surface, color, False, _arc_points(rect, a0, a1, 24))


def _wire(surface, rect, color, width: int) -> None:
    """Outer arch, inner loop, and a short curl. Coordinates are the rect's own pixels."""
    x, y, w = rect.x, rect.y, rect.w
    s = w / 26.0

    def box(u, v, rw, rh):
        return pygame_rect(x + u * s, y + v * s, max(width + 1, rw * s), max(width + 1, rh * s))

    def ends(arc, a0, a1):
        pts = _arc_points(arc, a0, a1, 1)
        return pts[0], pts[-1]

    crown = box(4, 2, 17, 10)
    _smooth_arc(surface, crown, 0.0, math.pi, color, width)
    crown_right, crown_left = ends(crown, 0.0, math.pi)
    bend = box(4, 15, 5, 7)
    _smooth_arc(surface, bend, math.pi, math.tau, color, width)
    bend_left, bend_right = ends(bend, math.pi, math.tau)
    _smooth_line(surface, crown_left, bend_left, color, width)
    inner = box(9, 8, 7, 6)
    _smooth_arc(surface, inner, 0.0, math.pi, color, width)
    inner_right, inner_left = ends(inner, 0.0, math.pi)
    _smooth_line(surface, bend_right, inner_left, color, width)
    _smooth_line(surface, inner_right, (inner_right[0], inner_right[1] + 6 * s), color, width)
    curl = box(15, 14, 6, 6)
    _smooth_arc(surface, curl, 1.5 * math.pi, math.tau, color, width)
    _curl_bottom, curl_right = ends(curl, 1.5 * math.pi, math.tau)
    _smooth_line(surface, crown_right, curl_right, color, width)


def _draw_clip(surface, rect, color, plate) -> None:
    """Plate and wire, drawn large and smoothed down. No caption on the button."""
    import pygame

    scale = 4
    big = pygame.Surface((rect.w * scale, rect.h * scale), pygame.SRCALPHA)
    pygame.draw.rect(big, plate, big.get_rect(), border_radius=5 * scale)
    _wire(big, big.get_rect(), color, scale * 2)
    surface.blit(pygame.transform.smoothscale(big, (rect.w, rect.h)), rect.topleft)


def pygame_rect(x, y, w, h):
    import pygame

    return pygame.Rect(int(x), int(y), int(w), int(h))


def _draw_pin_tip(surface, rect) -> None:
    import pygame

    font = pygame.font.Font(None, 16)
    text = font.render(PIN_TIP, True, (232, 232, 236))
    tw, th = text.get_size()
    pad_x, pad_y = 6, 4
    box = pygame.Rect(0, 0, tw + pad_x * 2, th + pad_y * 2)
    box.right = rect.x - 6
    box.centery = rect.centery
    if box.left < 4:
        box.left = 4
    pygame.draw.rect(surface, (28, 29, 34), box, border_radius=4)
    pygame.draw.rect(surface, (96, 98, 104), box, 1, border_radius=4)
    surface.blit(text, (box.x + pad_x, box.y + pad_y))


def draw_clip(surface, rect, pinned: bool, hover: bool = False):
    """Paperclip on ``rect``. Peach when pinned, gray otherwise.

    The icon is lines and arcs only. ``hover`` adds the «поверх окон» tip
    beside the button, not on it. The 3D window and the trainer bar share this.
    """
    from .sdl_thread import require_main_thread

    require_main_thread()

    color = PIN_ON if pinned else PIN_OFF
    plate = (48, 34, 30) if pinned else (32, 33, 36)
    _draw_clip(surface, rect, color, plate)
    if hover:
        _draw_pin_tip(surface, rect)
    return rect


def draw_pin_button(surface, pinned: bool, hover: bool = False):
    """Paperclip in the top-right of a separate pygame window."""
    rect = pygame_rect(*pin_button_rect(surface.get_size()))
    return draw_clip(surface, rect, pinned, hover)
