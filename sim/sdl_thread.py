"""SDL belongs to the thread that created the window.

pygame and SDL are not safe from a worker. The preview, YOLO, frame-record,
and lidar-reset threads may keep bytes or numpy arrays. A Surface, a font,
and ``image.load`` stay on the main thread.
"""

from __future__ import annotations

import threading


def require_main_thread() -> None:
    current = threading.current_thread()
    main = threading.main_thread()
    assert current is main, "pygame/SDL only on the main thread, not %s" % current.name
