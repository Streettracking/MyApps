"""O / «ошибка YOLO»: keep the raw camera frame and the YOLO boxes of that moment.

Works whether frame recording (U) is on or off. The JPEG is the exact
``/camera.jpg`` bytes, so it carries no overlays.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

CSV_HEADER = ("time", "file", "boxes (class conf [x0 y0 x1 y1], 0..1)")
# The teacher service sends no class id today, so the class reads "?".
BOX_CLASS = "?"


def _boxes_text(boxes) -> str:
    parts = []
    for box in boxes or ():
        try:
            parts.append(
                "%s %.3f [%.3f %.3f %.3f %.3f]"
                % (
                    str(getattr(box, "cls", "") or BOX_CLASS),
                    float(box.conf),
                    float(box.x0),
                    float(box.y0),
                    float(box.x1),
                    float(box.y1),
                )
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return "; ".join(parts)


class YoloErrorLog:
    def __init__(self, root: Path) -> None:
        self.dir = Path(root)
        self.count = 0

    def mark(self, jpeg: bytes | None, boxes) -> tuple[bool, str]:
        """Save one flagged frame. Returns (saved, file name or reason)."""
        if not jpeg:
            return False, "нет кадра камеры"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            now = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + "_%03d" % int((now % 1) * 1000)
            name = "yolo_error_%s.jpg" % stamp
            (self.dir / name).write_bytes(bytes(jpeg))
            csv_path = self.dir / "yolo_errors.csv"
            fresh = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                if fresh:
                    writer.writerow(CSV_HEADER)
                writer.writerow(
                    (
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        name,
                        _boxes_text(boxes),
                    )
                )
        except OSError as exc:
            return False, "не записано: %s" % exc
        self.count += 1
        return True, name
