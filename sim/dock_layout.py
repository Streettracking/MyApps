"""Splitter dock for the trainer blocks.

A camera keeps a 16:9 preference until the user drags its edge. Closing,
collapsing, and swapping leaves do not stretch the picture inside a block:
the monitor letterboxes camera frames and the lidar circle.
"""

from __future__ import annotations

HEADER = 26
SPLIT = 6
CONTROLS_H = 96
ROBOT_H = 96
MIN_BLOCK = 48
# Tall enough for the fresh-lidar button and the fly table (муха L/R, Δ, z,
# вектор) at the 13px font floor. Measured against the 1080 canvas: a taller
# column still gives the extra pixels to the lidar, and this floor is what a
# shorter canvas is not allowed to undercut.
LIDAR_MIN_H = 184

BLOCKS = ("camera", "lidar", "brain", "journal", "controls", "robot")
BLOCK_TITLES = {
    "camera": "камера",
    "lidar": "лидар",
    "brain": "панель мозга",
    "journal": "журнал",
    "controls": "кнопки",
    "robot": "Робот",
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
            Split("v", 0.52, Leaf("brain"), Leaf("journal")),
        ),
        Split("h", 0.62, Leaf("controls"), Leaf("robot")),
    )


def tree_from_dict(data) -> Split:
    """Load a saved dock. An unknown leaf, including an old ``brain3d``, is dropped.

    A split with one remaining child collapses, so the sibling takes that space.
    Anything that no longer names each block exactly once falls back to the default.
    """
    if not isinstance(data, dict):
        return default_tree()

    def parse(node):
        if not isinstance(node, dict):
            return None
        if node.get("t") == "leaf":
            name = node.get("id")
            if name not in BLOCKS:
                return None
            return Leaf(name, bool(node.get("collapsed")), bool(node.get("visible", True)))
        if node.get("t") == "split":
            a = parse(node.get("a"))
            b = parse(node.get("b"))
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            try:
                ratio = float(node.get("r") or 0.5)
            except (TypeError, ValueError):
                ratio = 0.5
            return Split(node.get("o") or "v", ratio, a, b, bool(node.get("locked")))
        return None

    parsed = parse(data)
    names = [leaf.block for leaf in _leaves(parsed)] if parsed is not None else []
    if not isinstance(parsed, Split) or len(names) != len(set(names)):
        return default_tree()
    present = set(names)
    if not present <= set(BLOCKS):
        return default_tree()
    missing = [name for name in BLOCKS if name not in present]
    # A saved dock from before the robot card keeps its splits. The new card
    # sits beside the controls leaf. Any other hole falls back to the default.
    if missing:
        if missing != ["robot"] or "controls" not in present:
            return default_tree()
        _graft_beside_controls(parsed, "robot")
        names = [leaf.block for leaf in _leaves(parsed)]
        if set(names) != set(BLOCKS) or len(names) != len(BLOCKS):
            return default_tree()
    elif set(names) != set(BLOCKS):
        return default_tree()
    return parsed


def _graft_beside_controls(node, block: str) -> bool:
    """Replace the controls leaf with a row: controls, then ``block``."""
    if isinstance(node, Leaf):
        return False
    for side in ("a", "b"):
        child = getattr(node, side)
        if isinstance(child, Leaf) and child.block == "controls":
            setattr(node, side, Split("h", 0.62, child, Leaf(block)))
            return True
        if _graft_beside_controls(child, block):
            return True
    return False


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
        if node.block in ("controls", "robot"):
            return min(height, CONTROLS_H if node.block == "controls" else ROBOT_H)
        return None
    if node.orient == "h":
        # A row is as tall as its tallest child that names a height.
        pref_a = _preferred_height(node.a, max(1, width // 2), height)
        pref_b = _preferred_height(node.b, max(1, width // 2), height)
        if pref_a is None or pref_b is None:
            return None
        return min(height, max(pref_a, pref_b))
    pref_a = _preferred_height(node.a, width, height)
    pref_b = _preferred_height(node.b, width, height)
    if pref_a is None or pref_b is None:
        return None
    return min(height, pref_a + SPLIT + pref_b)


def _min_height(node, width: int) -> int:
    """Smallest block height that still draws this node readably."""
    if isinstance(node, Leaf):
        if not node.visible:
            return 0
        if node.collapsed:
            return HEADER
        if node.block == "lidar":
            return LIDAR_MIN_H
        if node.block == "camera":
            return HEADER + 80
        if node.block == "controls":
            return CONTROLS_H
        if node.block == "robot":
            return ROBOT_H
        return HEADER + 36
    if not _takes_space(node.a):
        return _min_height(node.b, width)
    if not _takes_space(node.b):
        return _min_height(node.a, width)
    if node.orient == "h":
        half = max(1, int(width) // 2)
        return max(_min_height(node.a, half), _min_height(node.b, half))
    return _min_height(node.a, width) + SPLIT + _min_height(node.b, width)


def _clamp_share(top: int, avail: int, min_a: int, min_b: int) -> int:
    """Keep both sides above their minimum. Scale the minimums when both cannot fit."""
    avail = max(0, int(avail))
    min_a = max(0, int(min_a))
    min_b = max(0, int(min_b))
    top = int(top)
    if min_a + min_b <= avail:
        return max(min_a, min(top, avail - min_b))
    if min_a + min_b <= 0:
        return max(0, min(top, avail))
    return int(round(avail * min_a / float(min_a + min_b)))


def _collapsed_height(node, width: int) -> int | None:
    """Header-stack height when every visible leaf is collapsed.

    ``None`` means the node still has an expanded block and may grow.
    A hidden leaf contributes nothing. The saved splitter ratio is not read.
    """
    if isinstance(node, Leaf):
        if not node.visible:
            return 0
        if node.collapsed:
            return HEADER
        return None
    if not _takes_space(node.a) and not _takes_space(node.b):
        return 0
    if not _takes_space(node.a):
        return _collapsed_height(node.b, width)
    if not _takes_space(node.b):
        return _collapsed_height(node.a, width)
    if node.orient == "h":
        half = max(1, int(width) // 2)
        ha = _collapsed_height(node.a, half)
        hb = _collapsed_height(node.b, half)
        if ha is None or hb is None:
            return None
        return max(ha, hb)
    ha = _collapsed_height(node.a, width)
    hb = _collapsed_height(node.b, width)
    if ha is None or hb is None:
        return None
    return ha + SPLIT + hb


def _subtract_rect(rect: tuple, cut: tuple) -> list:
    """Rectangle ``rect`` minus ``cut``. Pieces stay inside ``rect``."""
    x, y, w, h = rect
    cx, cy, cw, ch = cut
    ix0 = max(x, cx)
    iy0 = max(y, cy)
    ix1 = min(x + w, cx + cw)
    iy1 = min(y + h, cy + ch)
    if ix1 <= ix0 or iy1 <= iy0:
        return [rect]
    pieces = []
    if iy0 > y:
        pieces.append((x, y, w, iy0 - y))
    if iy1 < y + h:
        pieces.append((x, iy1, w, y + h - iy1))
    if ix0 > x:
        pieces.append((x, iy0, ix0 - x, iy1 - iy0))
    if ix1 < x + w:
        pieces.append((ix1, iy0, x + w - ix1, iy1 - iy0))
    return [piece for piece in pieces if piece[2] > 0 and piece[3] > 0]


def _clip_splitters(splitters: list, blocks: dict) -> list:
    """Drop the part of a grip that a block has taken over."""
    kept = []
    for splitter in splitters:
        pieces = [tuple(splitter["rect"])]
        for rect in blocks.values():
            nxt = []
            for piece in pieces:
                nxt.extend(_subtract_rect(piece, rect))
            pieces = nxt
        for piece in pieces:
            if piece[2] >= 2 and piece[3] >= 2:
                kept.append({
                    "rect": piece,
                    "node": splitter["node"],
                    "axis": splitter["axis"],
                    "bounds": splitter["bounds"],
                })
    return kept


def _splitter_above(splitters: list, x: int, y: int, w: int):
    """The horizontal grip that ends where this gap begins, if there is one."""
    found = None
    for splitter in splitters:
        if splitter["axis"] != "y":
            continue
        sx, sy, sw, sh = splitter["rect"]
        if abs((sy + sh) - y) > SPLIT + 1:
            continue
        if min(sx + sw, x + w) - max(sx, x) <= 0:
            continue
        found = splitter
    return found


def _absorb_gaps(blocks: dict, gaps: list, splitters: list, root) -> None:
    """Give the band above a collapsed card to the blocks sitting on it.

    A band is absorbed only when each piece matches a whole neighbor, so a
    block stays a rectangle. A band that would cut a neighbor in half is
    given back to the collapsed card instead of leaving a hole.
    """
    for gx, gy, gw, gh in gaps:
        if gw < 4 or gh < 4:
            continue
        if _give_gap_to_neighbors(blocks, splitters, root, gx, gy, gw, gh):
            continue
        _give_gap_back(blocks, gx, gy, gw, gh)


def _give_gap_to_neighbors(blocks, splitters, root, gx, gy, gw, gh) -> bool:
    owners = []
    for name, rect in blocks.items():
        leaf = find_leaf(root, name)
        if leaf is None or leaf.collapsed or not leaf.visible:
            continue
        if leaf.block in ("controls", "robot"):
            continue
        x, y, w, h = rect
        if abs((y + h + SPLIT) - gy) > 1 and abs((y + h) - gy) > 1:
            continue
        if x < gx or x + w > gx + gw:
            continue
        if w <= 0:
            continue
        owners.append((x, name, rect))
    owners.sort()
    if not owners:
        return False
    cursor = gx
    chosen = []
    for x, name, rect in owners:
        if x != cursor:
            return False
        chosen.append((name, rect))
        cursor = x + rect[2]
    if cursor != gx + gw:
        return False
    planned = []
    for name, rect in chosen:
        x, y, w, h = rect
        new_bottom = gy + gh - SPLIT
        if new_bottom <= y + h:
            return False
        planned.append((name, x, y, w, new_bottom - y, new_bottom))
    for name, x, y, w, h, new_bottom in planned:
        blocks[name] = (x, y, w, h)
        donor = _splitter_above(splitters, x, gy, w)
        if donor is None:
            continue
        splitters.append({
            "rect": (x, new_bottom, w, SPLIT),
            "node": donor["node"],
            "axis": "y",
            "bounds": donor["bounds"],
        })
    return True


def _give_gap_back(blocks, gx, gy, gw, gh) -> None:
    """Stretch the collapsed card across a band no neighbor can take whole."""
    floor = gy + gh
    for name, rect in list(blocks.items()):
        x, y, w, h = rect
        if abs(y - floor) > 1:
            continue
        if x <= gx and x + w >= gx + gw:
            blocks[name] = (x, gy, w, h + (y - gy))
            return


def layout_dock(node, x: int, y: int, w: int, h: int) -> dict:
    """Map each visible block to ``(x, y, w, h)`` and list splitter grips.

    An unlocked camera keeps a 16:9 preference while its sibling stays above
    a readable minimum. When that preference no longer fits, the saved ratio
    scales both sides and the minimum still wins. A locked split always
    scales by its ratio, then the same minimums. Hidden leaves take no space:
    the sibling subtree keeps its own ratio, so the blocks that remain grow
    in proportion to the share they already had. A collapsed leaf is only
    the header tall. The band it gives up goes to the neighbors above it.
    Splitter ratios are not rewritten, so opening the block puts it back.
    """
    blocks: dict = {}
    splitters: list = []
    gaps: list = []

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

            def place_side(child, sx, sw) -> None:
                tight = _collapsed_height(child, sw)
                if tight is not None and HEADER // 2 <= tight < rh:
                    gaps.append((sx, ry, sw, rh - tight))
                    place(child, sx, ry + (rh - tight), sw, tight)
                    return
                place(child, sx, ry, sw, rh)

            place_side(current.a, rx, left)
            grip = (rx + left, ry, SPLIT, rh)
            splitters.append({"rect": grip, "node": current, "axis": "x", "bounds": (rx, ry, rw, rh)})
            place_side(current.b, rx + left + SPLIT, rw - left - SPLIT)
            return
        avail = max(0, rh - SPLIT)
        ratio = min(0.9, max(0.1, current.ratio))
        collapsed_a = _collapsed_height(current.a, rw)
        collapsed_b = _collapsed_height(current.b, rw)
        if (
            collapsed_a is not None
            and collapsed_b is not None
            and collapsed_a + SPLIT + collapsed_b < rh
        ):
            # Both sides are header stacks and the parent gave this column
            # extra height. Park the headers at the bottom and let the band
            # above them travel to a neighbor outside this split.
            slack = rh - (collapsed_a + SPLIT + collapsed_b)
            place(current.a, rx, ry + slack, rw, collapsed_a)
            grip = (rx, ry + slack + collapsed_a, rw, SPLIT)
            splitters.append({"rect": grip, "node": current, "axis": "y", "bounds": (rx, ry, rw, rh)})
            place(current.b, rx, ry + slack + collapsed_a + SPLIT, rw, collapsed_b)
            gaps.append((rx, ry, rw, slack))
            return
        if collapsed_a is not None and collapsed_b is not None:
            top = int(round(avail * collapsed_a / float(max(1, collapsed_a + collapsed_b))))
        elif collapsed_a is not None:
            top = min(collapsed_a, avail)
        elif collapsed_b is not None:
            top = max(0, avail - min(collapsed_b, avail))
        elif current.locked:
            top = int(round(avail * ratio))
        else:
            pref = _preferred_height(current.a, rw, rh)
            other = _preferred_height(current.b, rw, rh)
            if pref is None and other is None:
                top = int(round(avail * ratio))
            elif pref is None:
                top = avail - int(other or 0)
            elif other is None:
                top = int(pref)
                leftover = avail - top
                min_b = _min_height(current.b, rw)
                if leftover < min_b:
                    ratio_bottom = avail - int(round(avail * ratio))
                    top = avail - max(min_b, ratio_bottom)
            elif int(pref) + int(other) <= avail:
                top = int(pref)
            else:
                top = int(round(avail * int(pref) / float(int(pref) + int(other))))
        if collapsed_a is None or collapsed_b is None:
            top = _clamp_share(top, avail, _min_height(current.a, rw), _min_height(current.b, rw))
        place(current.a, rx, ry, rw, top)
        grip = (rx, ry + top, rw, SPLIT)
        splitters.append({"rect": grip, "node": current, "axis": "y", "bounds": (rx, ry, rw, rh)})
        bottom_h = rh - top - SPLIT
        place(current.b, rx, ry + top + SPLIT, rw, bottom_h)

    place(node, int(x), int(y), int(w), int(h))
    _absorb_gaps(blocks, gaps, splitters, node)
    splitters = _clip_splitters(splitters, blocks)
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
        "info": (x + w - 68, hy, 20, 20),
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
