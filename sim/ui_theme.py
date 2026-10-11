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
    "shadow": (140, 120, 100),
    "peach_soft": (227, 164, 135),
    "amber_soft": (240, 206, 140),
    "teacher_soft": (156, 208, 230),
    "lamp_green_hi": (196, 220, 198),
    "lamp_red_hi": (236, 196, 190),
    "lamp_estop_hi": (214, 92, 84),
    "lamp_wash": (255, 255, 255),
    "ring": (176, 174, 170),
    "estop_label": (255, 246, 244),
    "estop_edge": (255, 228, 224),
    "key_badge": (132, 132, 134),
    "key_badge_ink": (255, 255, 255),
}

DARK = {
    "bg_top": (28, 29, 32),
    "bg_bot": (22, 23, 26),
    "card": (40, 42, 46),
    "line": (70, 72, 78),
    # Main ink is near white on the card (≥ 7:1). Secondary is light gray.
    "label": (248, 248, 248),
    "value": (255, 255, 255),
    "ink_dim": (198, 200, 204),
    "scroll_track": (58, 58, 60),
    "scroll_thumb": (132, 132, 134),
    "menu": (36, 37, 40),
    "inset": (30, 31, 34),
    # Near-black lift. The light theme's warm brown reads as mud on a dark card.
    "shadow": (8, 7, 6),
    "peach_soft": (232, 176, 150),
    "amber_soft": (236, 196, 120),
    "teacher_soft": (120, 186, 214),
    "lamp_green_hi": (168, 204, 176),
    "lamp_red_hi": (214, 160, 154),
    "lamp_estop_hi": (196, 78, 72),
    "lamp_wash": (210, 212, 216),
    "ring": (150, 152, 158),
    "estop_label": (255, 246, 244),
    "estop_edge": (255, 210, 204),
    "key_badge": (88, 90, 96),
    "key_badge_ink": (255, 255, 255),
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
SHADOW = ThemeColor("shadow", LIGHT["shadow"])
PEACH_SOFT = ThemeColor("peach_soft", LIGHT["peach_soft"])
AMBER_SOFT = ThemeColor("amber_soft", LIGHT["amber_soft"])
TEACHER_SOFT = ThemeColor("teacher_soft", LIGHT["teacher_soft"])
LAMP_GREEN_HI = ThemeColor("lamp_green_hi", LIGHT["lamp_green_hi"])
LAMP_RED_HI = ThemeColor("lamp_red_hi", LIGHT["lamp_red_hi"])
LAMP_ESTOP_HI = ThemeColor("lamp_estop_hi", LIGHT["lamp_estop_hi"])
LAMP_WASH = ThemeColor("lamp_wash", LIGHT["lamp_wash"])
RING = ThemeColor("ring", LIGHT["ring"])
ESTOP_LABEL = ThemeColor("estop_label", LIGHT["estop_label"])
ESTOP_EDGE = ThemeColor("estop_edge", LIGHT["estop_edge"])
KEY_BADGE = ThemeColor("key_badge", LIGHT["key_badge"])
KEY_BADGE_INK = ThemeColor("key_badge_ink", LIGHT["key_badge_ink"])
