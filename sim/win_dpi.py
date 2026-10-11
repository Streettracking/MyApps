"""Windows DPI and the monitor work area, before a pygame window exists.

Linux and a headless run do nothing and do not raise. This module does not
create a display and does not borrow an SDL window.
"""

from __future__ import annotations

# SystemParametersInfo: the rectangle programs should stay inside.
SPI_GETWORKAREA = 0x0030


def enable_dpi_awareness() -> None:
    """Per-monitor DPI, before the first window. A miss falls back to system DPI."""
    import os

    if os.name != "nt":
        return
    try:
        import ctypes

        try:
            # 2 = PROCESS_PER_MONITOR_DPI_AWARE
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        return


def work_area():
    """``(x, y, w, h)`` of the primary work area, or None off Windows."""
    import os

    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        ok = ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
        if not ok:
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width < 64 or height < 64:
            return None
        return (int(rect.left), int(rect.top), width, height)
    except Exception:
        return None


def clip_window_box(x, y, w, h, area) -> tuple:
    """Keep a saved window inside the work area. No area leaves the box alone."""
    w = max(320, int(w or 0))
    h = max(240, int(h or 0))
    if not area:
        return x, y, w, h
    ax, ay, aw, ah = (int(area[0]), int(area[1]), int(area[2]), int(area[3]))
    w = min(w, max(320, aw))
    h = min(h, max(240, ah))
    if x is None or y is None:
        return x, y, w, h
    x = min(max(int(x), ax), ax + max(0, aw - w))
    y = min(max(int(y), ay), ay + max(0, ah - h))
    return x, y, w, h
