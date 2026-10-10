"""Splitter dock for the trainer blocks.

A camera keeps a 16:9 preference until the user drags its edge. Closing,
collapsing, and swapping leaves do not stretch the picture inside a block:
the monitor letterboxes camera frames and the lidar circle.
"""

from __future__ import annotations

HEADER = 26
SPLIT = 6
CONTROLS_H = 74
BRAIN3D_H = 280
MIN_BLOCK = 48

BLOCKS = ("camera", "lidar", "brain", "journal", "controls", "brain3d")
BLOCK_TITLES = {
    "camera": "камера",
    "lidar": "лидар",
    "brain": "панель мозга",
    "journal": "журнал",
    "controls": "кнопки",
    "brain3d": "мозг 3D",
}


class Leaf:
    def __init__(self, block: str, collapsed: bool = False, visible: bool = True):
        self.block = str(block)
        self.collapsed = bool(collapsed)
        self.visible = bool(visible)

    def to_dict(self) -> dict:
        return {"t": "leaf", "id": self.block, "collapsed": self.collapsed, "visible": self.visible}


class Split:
    def __init__(self, orient: str, ratio: float, a, b, locked: bool = False):
        self.orient = "h" if orient == "h" else "v"
        self.ratio = float(ratio)
        self.a = a
        self.b = b
        self.locked = bool(locked)

    def to_dict(self) -> dict:
        return {
            "t": "split",
            "o": self.orient,
            "r": round(self.ratio, 4),
            "locked": self.locked,
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
        }


def default_tree() -> Split:
    """Camera on the 16:9 preference, lidar and the brain column take the rest."""
    return Split(
        "v",
        0.90,
        Split(
            "h",
            0.62,
            Split("v", 0.72, Leaf("camera"), Leaf("lidar")),
            Split(
                "v",
                0.52,
                Leaf("brain"),
                Split("v", 0.55, Leaf("journal"), Leaf("brain3d")),
            ),
        ),
        Leaf("controls"),
    )


def tree_from_dict(data) -> Split:
    if not isinstance(data, dict):
        return default_tree()

    def parse(node):
        if not isinstance(node, dict):
            return None
        if node.get("t") == "leaf" and node.get("id") in BLOCKS:
            return Leaf(node["id"], bool(node.get("collapsed")), bool(node.get("visible", True)))
        if node.get("t") == "split":
            a = parse(node.get("a"))
            b = parse(node.get("b"))
            if a is None or b is None:
                return None
            return Split(node.get("o") or "v", float(node.get("r") or 0.5), a, b, bool(node.get("locked")))
        return None

    parsed = parse(data)
    names = [leaf.block for leaf in _leaves(parsed)] if isinstance(parsed, Split) else []
    if not isinstance(parsed, Split) or set(names) != set(BLOCKS) or len(names) != len(BLOCKS):
        return default_tree()
    return parsed


def _leaves(node) -> list:
    if isinstance(node, Leaf):
        return [node]
    return _leaves(node.a) + _leaves(node.b)


def find_leaf(node, block: str) -> Leaf | None:
    for leaf in _leaves(node):
        if leaf.block == block:
            return leaf
    return None


def swap_blocks(node, left: str, right: str) -> bool:
    a = find_leaf(node, left)
    b = find_leaf(node, right)
    if a is None or b is None or a is b:
        return False
    a.block, b.block = b.block, a.block
    a.collapsed, b.collapsed = b.collapsed, a.collapsed
    return True


def _preferred_height(node, width: int, height: int) -> int | None:
    """Pixels this node wants. ``None`` means it shares leftover space."""
    if isinstance(node, Leaf):
        if not node.visible:
            return 0
        if node.collapsed:
            return HEADER
        if node.block == "camera":
            return min(height, HEADER + max(80, int(round(width * 9 / 16))))
        if node.block == "controls":
            return min(height, CONTROLS_H)
        if node.block == "brain3d":
            return min(height, BRAIN3D_H)
        return None
    if node.orient == "h":
        return None
    pref_a = _preferred_height(node.a, width, height)
    pref_b = _preferred_height(node.b, width, height)
    if pref_a is None or pref_b is None:
        return None
    return min(height, pref_a + SPLIT + pref_b)


def layout_dock(node, x: int, y: int, w: int, h: int) -> dict:
    """Map each visible block to ``(x, y, w, h)`` and list splitter grips.

    The camera preference is 16:9 of its column until that split is locked
    by a drag. Hidden leaves take no space.
    """
    blocks: dict = {}
    splitters: list = []

    def place(current, rx: int, ry: int, rw: int, rh: int) -> None:
        if rw < 4 or rh < 4:
            return
        if isinstance(current, Leaf):
            if current.visible and rh >= HEADER // 2:
                blocks[current.block] = (int(rx), int(ry), int(rw), int(rh))
            return
        if not _takes_space(current.a) and not _takes_space(current.b):
            return
        if not _takes_space(current.a):
            place(current.b, rx, ry, rw, rh)
            return
        if not _takes_space(current.b):
            place(current.a, rx, ry, rw, rh)
            return
        if current.orient == "h":
            ratio = min(0.85, max(0.15, current.ratio))
            left = int(round((rw - SPLIT) * ratio))
            left = max(MIN_BLOCK, min(left, rw - SPLIT - MIN_BLOCK))
            place(current.a, rx, ry, left, rh)
            grip = (rx + left, ry, SPLIT, rh)
            splitters.append({"rect": grip, "node": current, "axis": "x", "bounds": (rx, ry, rw, rh)})
            place(current.b, rx + left + SPLIT, ry, rw - left - SPLIT, rh)
            return
        ratio = min(0.9, max(0.1, current.ratio))
        if current.locked:
            top = int(round((rh - SPLIT) * ratio))
        else:
            pref = _preferred_height(current.a, rw, rh)
            other = _preferred_height(current.b, rw, rh)
            if pref is None and other is None:
                top = int(round((rh - SPLIT) * ratio))
            elif pref is None:
                top = rh - SPLIT - int(other or 0)
            else:
                top = int(pref)
        top = max(0, min(int(top), rh - SPLIT))
        if top < HEADER and _takes_space(current.a):
            top = min(rh - SPLIT, HEADER)
        place(current.a, rx, ry, rw, top)
        grip = (rx, ry + top, rw, SPLIT)
        splitters.append({"rect": grip, "node": current, "axis": "y", "bounds": (rx, ry, rw, rh)})
        place(current.b, rx, ry + top + SPLIT, rw, rh - top - SPLIT)

    place(node, int(x), int(y), int(w), int(h))
    return {"blocks": blocks, "splitters": splitters}


def _takes_space(node) -> bool:
    if isinstance(node, Leaf):
        return bool(node.visible)
    return _takes_space(node.a) or _takes_space(node.b)


def content_rect(block_rect: tuple, collapsed: bool) -> tuple | None:
    x, y, w, h = block_rect
    if collapsed or h <= HEADER + 4:
        return None
    return (x + 8, y + HEADER, max(1, w - 16), max(1, h - HEADER - 8))


def header_controls(block_rect: tuple) -> dict:
    x, y, w, h = block_rect
    hy = y + 2
    return {
        "bar": (x, y, w, min(HEADER, h)),
        "collapse": (x + w - 46, hy, 20, 20),
        "close": (x + w - 24, hy, 20, 20),
    }


def hit_header(blocks: dict, leaves: dict, px: int, py: int) -> str | None:
    for name, rect in blocks.items():
        bar = header_controls(rect)["bar"]
        if bar[0] <= px < bar[0] + bar[2] and bar[1] <= py < bar[1] + bar[3]:
            leaf = leaves.get(name)
            if leaf is not None and leaf.visible:
                return name
    return None
