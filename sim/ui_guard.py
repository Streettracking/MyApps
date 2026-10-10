"""Switches for the Windows crash that started with the dock (7518c48).

Set a variable to ``1``. The trainer never calls ``pygame.mouse.set_cursor``,
``display.set_icon``, ``event.set_grab``, ``event.set_blocked``, ``pygame.scrap``,
or ``pygame.time.set_timer``.

- ``RECOGNIZER_NO_CURSOR`` — there is no cursor change to turn off. This is the
  default (``RECOGNIZER_CURSOR`` is unset). Do not add ``set_cursor``.
- ``RECOGNIZER_NO_CAPTION`` — skip ``display.set_caption``.
- ``RECOGNIZER_NO_SNAP`` — do not keep the journal or brain-panel Surface caches.
- ``RECOGNIZER_NO_CARD_CACHE`` — build each card and drop it.
- ``RECOGNIZER_NO_PRESENT`` — blit the logical canvas at the top-left. No smoothscale.
- ``RECOGNIZER_NO_SDL_WINDOW`` — do not call ``pygame._sdl2.video.Window``.
- ``RECOGNIZER_SDL_WINDOW_ON_RESIZE`` — the 7518c48 path: a new Window on every
  resize, then drop it. That stores a Python object in SDL window data and
  lets it be collected.
- ``RECOGNIZER_FORCE_SETMODE`` — call ``display.set_mode`` even when the live
  surface is already that size.
- ``RECOGNIZER_TUPLE_COLORS`` — pass plain RGB tuples into pygame instead of ThemeColor.
- ``RECOGNIZER_NO_KEY_REPEAT`` — do not call ``key.set_repeat`` in the 3D window.
"""

from __future__ import annotations

import os

_HELD_WINDOW = None


def enabled(name: str) -> bool:
    return os.environ.get(name, "") == "1"


def no_cursor() -> bool:
    return enabled("RECOGNIZER_NO_CURSOR") or not enabled("RECOGNIZER_CURSOR")


def no_caption() -> bool:
    return enabled("RECOGNIZER_NO_CAPTION")


def no_snap() -> bool:
    return enabled("RECOGNIZER_NO_SNAP")


def no_card_cache() -> bool:
    return enabled("RECOGNIZER_NO_CARD_CACHE")


def no_present() -> bool:
    return enabled("RECOGNIZER_NO_PRESENT")


def no_sdl_window() -> bool:
    return enabled("RECOGNIZER_NO_SDL_WINDOW")


def sdl_window_on_resize() -> bool:
    return enabled("RECOGNIZER_SDL_WINDOW_ON_RESIZE")


def force_setmode() -> bool:
    return enabled("RECOGNIZER_FORCE_SETMODE")


def tuple_colors() -> bool:
    return enabled("RECOGNIZER_TUPLE_COLORS")


def no_key_repeat() -> bool:
    return enabled("RECOGNIZER_NO_KEY_REPEAT")


def borrowed_window():
    """One Window for this process. from_display_module does not keep a ref.

    pygame writes the Python object into the SDL window (``pg_window``) and
    ``__dealloc__`` does not clear it. A temporary Window becomes a dangling
    pointer. Hold the single object here and do not call from_display_module
    again.
    """
    global _HELD_WINDOW

    if no_sdl_window():
        return None
    if _HELD_WINDOW is not None:
        return _HELD_WINDOW
    from pygame._sdl2.video import Window

    _HELD_WINDOW = Window.from_display_module()
    return _HELD_WINDOW


def throwaway_window():
    """7518c48: a Window nobody keeps. Only the resize bisect flag uses this."""
    if no_sdl_window():
        return None
    from pygame._sdl2.video import Window

    return Window.from_display_module()


def flatten_theme_colors() -> None:
    """Replace ThemeColor with the light-theme tuple pygame already knows."""
    import sys

    import sim.ui_theme as theme

    names = (
        "BG_TOP",
        "BG_BOT",
        "CARD",
        "LINE",
        "LABEL",
        "VALUE",
        "INK_DIM",
        "SCROLL_TRACK",
        "SCROLL_THUMB",
        "MENU",
        "INSET",
    )
    flats = {}
    for name in names:
        color = getattr(theme, name)
        light = getattr(color, "light", color)
        flats[name] = tuple(int(channel) for channel in light)
        setattr(theme, name, flats[name])
    aliases = {"INK": "VALUE", "_LABEL": "LABEL", "_VALUE": "VALUE"}
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for name, flat in flats.items():
            current = getattr(mod, name, None)
            if type(current).__name__ == "ThemeColor":
                setattr(mod, name, flat)
        for alias, source in aliases.items():
            current = getattr(mod, alias, None)
            if type(current).__name__ == "ThemeColor":
                setattr(mod, alias, flats[source])
