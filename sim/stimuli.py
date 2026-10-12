"""Camera stimuli that may reinforce the mushroom body.

Lidar range is not a source. A close return still stops the dog in the
pilot and in ``apply_guards``; that reflex never enters this registry.
"""

from __future__ import annotations

# A confident box this tall, or covering this much of the frame, is too close.
CLOSE_HEIGHT = 0.70
CLOSE_AREA = 0.18
# Growth of the tracked box, as a fraction of the frame, counts as an approach.
APPROACH_GROWTH = 0.02

# Learning reads the YOLO box only. Lidar is display and a motor reflex.
LEARNING_SOURCES = ("yolo_bbox",)


def learning_sources() -> tuple[str, ...]:
    return LEARNING_SOURCES


def _metrics(box) -> tuple[float, float, float]:
    x0 = float(box.x0)
    x1 = float(box.x1)
    y0 = float(getattr(box, "y0", 0.0))
    y1 = float(getattr(box, "y1", 1.0))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    width = x1 - x0
    height = y1 - y0
    if width < 0.0:
        width = 0.0
    if height < 0.0:
        height = 0.0
    return width, height, width * height


def camera_stimuli(boxes, prev_area: float | None, conf_min: float = 0.5) -> tuple[bool, bool, float | None]:
    """``(too_close, approaching, area)`` from YOLO boxes. Lidar is not an argument.

    ``too_close`` is a box whose height reaches ``CLOSE_HEIGHT`` or whose area
    reaches ``CLOSE_AREA``. ``approaching`` is the strongest box growing by
    ``APPROACH_GROWTH``. ``area`` is that box, or None when nothing qualifies.
    """
    best_area = None
    best_conf = float(conf_min) - 1e-9
    close = False
    for box in boxes or ():
        conf = float(getattr(box, "conf", 0.0))
        if conf < float(conf_min):
            continue
        _width, height, area = _metrics(box)
        if height + 1e-12 >= CLOSE_HEIGHT or area + 1e-12 >= CLOSE_AREA:
            close = True
        if conf >= best_conf:
            best_conf = conf
            best_area = area
    grew = False
    if best_area is not None and prev_area is not None and best_area > float(prev_area) + APPROACH_GROWTH:
        grew = True
    return close, grew, best_area
