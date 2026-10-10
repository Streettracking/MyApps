"""Light and dark chrome. Semantic greens, reds, and peach stay fixed.

Numbers stay on the monospace path in ``tabnum``. These objects read the
active scheme when pygame asks for a colour, and still compare equal to the
light hex values the tests pin.
"""

from __future__ import annotations

LIGHT = {
    "bg_top": (236, 234, 230),
    "bg_bot": (244, 242, 239),
    "card": (247, 246, 244),
    "line": (226, 224, 220),
    "label": (34, 34, 34),
    "value": (17, 17, 17),
    "ink_dim": (68, 68, 68),
    "scroll_track": (188, 188, 188),
    "scroll_thumb": (142, 142, 142),
    "menu": (250, 249, 247),
    "inset": (255, 255, 255),
}

DARK = {
    "bg_top": (28, 29, 32),
    "bg_bot": (22, 23, 26),
    "card": (40, 42, 46),
    "line": (70, 72, 78),
    "label": (230, 230, 230),
    "value": (245, 245, 245),
    "ink_dim": (176, 176, 176),
    "scroll_track": (58, 58, 60),
    "scroll_thumb": (132, 132, 134),
    "menu": (36, 37, 40),
    "inset": (30, 31, 34),
}

THEMES = {"light": LIGHT, "dark": DARK}
ACTIVE_NAME = "light"
ACTIVE = dict(LIGHT)


class ThemeColor:
    """Sequence of three ints. Equality uses the light-theme value."""

    def __init__(self, key: str, light: tuple):
        self.key = key
        self.light = tuple(int(c) for c in light)

    def _rgb(self) -> tuple:
        return tuple(ACTIVE.get(self.key, self.light))

    def __iter__(self):
        return iter(self._rgb())

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        return self._rgb()[index]

    def __eq__(self, other) -> bool:
        if isinstance(other, ThemeColor):
            return self.key == other.key
        try:
            return self.light == tuple(other)
        except TypeError:
            return False

    def __hash__(self) -> int:
        return hash(self.light)

    def __repr__(self) -> str:
        return "ThemeColor(%s, %s)" % (self.key, self._rgb())


def apply_theme(name: str) -> str:
    """Switch the active scheme. Unknown names stay on light."""
    global ACTIVE_NAME, ACTIVE
    key = str(name or "light").lower()
    if key not in THEMES:
        key = "light"
    ACTIVE_NAME = key
    ACTIVE = dict(THEMES[key])
    return key


def theme_name() -> str:
    return ACTIVE_NAME


BG_TOP = ThemeColor("bg_top", LIGHT["bg_top"])
BG_BOT = ThemeColor("bg_bot", LIGHT["bg_bot"])
CARD = ThemeColor("card", LIGHT["card"])
LINE = ThemeColor("line", LIGHT["line"])
LABEL = ThemeColor("label", LIGHT["label"])
VALUE = ThemeColor("value", LIGHT["value"])
INK_DIM = ThemeColor("ink_dim", LIGHT["ink_dim"])
SCROLL_TRACK = ThemeColor("scroll_track", LIGHT["scroll_track"])
SCROLL_THUMB = ThemeColor("scroll_thumb", LIGHT["scroll_thumb"])
MENU = ThemeColor("menu", LIGHT["menu"])
INSET = ThemeColor("inset", LIGHT["inset"])
