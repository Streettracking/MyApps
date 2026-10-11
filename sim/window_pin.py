"""Pin a separate pygame window above the others.

Windows: one ``SetWindowPos`` per click, on the HWND from
``pygame.display.get_wm_info()['window']``. Linux and a headless driver
return False and do not raise. The call does not recreate the display and
does not borrow an SDL window wrapper.
"""

from __future__ import annotations

# Same peach as the trainer accent. The idle clip is a quiet gray.
PIN_ON = (201, 120, 91)
PIN_OFF = (132, 132, 134)

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


def draw_pin_button(surface, pinned: bool):
    """Paperclip in the top-right. Peach when pinned, gray otherwise."""
    from .sdl_thread import require_main_thread

    require_main_thread()
    import pygame

    rect = pygame.Rect(pin_button_rect(surface.get_size()))
    color = PIN_ON if pinned else PIN_OFF
    plate = (48, 34, 30) if pinned else (32, 33, 36)
    pygame.draw.rect(surface, plate, rect, border_radius=5)
    pygame.draw.rect(surface, color, rect, width=1, border_radius=5)
    x, y = rect.x, rect.y
    pygame.draw.lines(
        surface,
        color,
        False,
        (
            (x + 9, y + 21),
            (x + 9, y + 10),
            (x + 11, y + 6),
            (x + 15, y + 6),
            (x + 17, y + 10),
            (x + 17, y + 17),
            (x + 15, y + 19),
            (x + 13, y + 17),
            (x + 13, y + 11),
            (x + 12, y + 9),
            (x + 11, y + 11),
            (x + 11, y + 18),
        ),
        2,
    )
    surface.set_at(rect.center, color)
    return rect
