"""Keep the mushroom body on a short lidar window.

The preview server on the robot paints ``/lidar.jpg`` until something calls
``/lidar/reset``. A frame just after that call is almost empty (the map
redraws about every 0.12 s). A frame just before the next call is one short
trace and nothing older. That is the frame the brain sees. The same picture
stays on screen through the following wipe, so the operator does not watch
the map blink empty.

``interval <= 0`` leaves the server alone and feeds every JPEG through.
That is the old accumulating map.
"""

from __future__ import annotations

import math

import numpy as np

from .raw_sense import FOV, N_AZ, SEE_M, bearing_bin, wall_range

DEFAULT_LIDAR_REFRESH = 1.5


def fill_seconds(interval: float) -> float:
    """Ignore the map this long after a reset. It is still filling in.

    0.45 s is about three redraws on the existing preview server. The rest of
    the interval is what we keep, and the brain takes the last of those frames.
    """
    if interval <= 0:
        return 0.0
    fill = min(0.45, max(0.20, 0.30 * float(interval)))
    return float(min(fill, float(interval) * 0.6))


def mismatch_warning(saved: float | None, current: float) -> str | None:
    """Old weights were trained on a different lidar window.

    ``saved is None`` means a file from before this flag: that brain saw the
    accumulating map. ``current <= 0`` means accumulation is on now.
    """
    if saved is None:
        saved_fresh = False
        saved_value = 0.0
    else:
        saved_fresh = float(saved) > 0
        saved_value = float(saved)
    current_fresh = float(current) > 0
    if not saved_fresh and not current_fresh:
        return None
    if saved_fresh != current_fresh:
        if current_fresh:
            return (
                f"веса учились на накапливаемой карте лидара, сейчас свежее окно {current:.1f} с. "
                "Клавиша R сбрасывает мозг, файл не стираю."
            )
        return (
            f"веса учились на свежем окне {saved_value:.1f} с, сейчас карта копится. "
            "Клавиша R сбрасывает мозг, файл не стираю."
        )
    if saved_fresh and current_fresh and max(saved_value, current) >= 2.0 * min(saved_value, current):
        return (
            f"веса учились при окне лидара {saved_value:.1f} с, сейчас {current:.1f} с. "
            "Клавиша R сбрасывает мозг, файл не стираю."
        )
    return None


class FreshWindow:
    """Choose the lidar JPEG that is safe to learn from.

    The caller sends ``/lidar/reset`` when ``poll_reset`` is true, then
    ``ack`` when that GET finishes. Images that arrive while the reset is
    still in flight, and images in the fill period after it, are not stored.
    """

    def __init__(self, interval: float):
        self.interval = float(interval)
        self.committed = None
        self.committed_scan = None
        self.phase = "idle"  # idle | resetting | collecting
        self.origin: float | None = None
        self.need_reset = False
        self._candidate = None
        self._candidate_scan = None
        self._last_send: float | None = None
        self._fail_logged = -10.0

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def configure(self, interval: float) -> None:
        self.interval = float(interval)
        if self.interval <= 0:
            self.phase = "idle"
            self.need_reset = False
            self.origin = None
            self._candidate = None

    def start(self, now: float) -> None:
        """Drop the picture we were holding and ask for a reset on the next poll."""
        del now
        self.committed = None
        self.committed_scan = None
        self._candidate = None
        self.phase = "resetting"
        self.origin = None
        self.need_reset = True
        self._last_send = None

    def manual_clear(self) -> None:
        """Operator pressed C. Caller performs the GET itself."""
        self.committed = None
        self.committed_scan = None
        self._candidate = None
        self.origin = None
        self.phase = "resetting"
        self.need_reset = False

    def poll_reset(self, now: float) -> bool:
        if not self.enabled or not self.need_reset:
            return False
        if self._last_send is not None and now - self._last_send < 0.8:
            return False
        self.need_reset = False
        self._last_send = now
        self.phase = "resetting"
        return True

    def ack(self, ok: bool, now: float) -> bool:
        """Record the GET result. Return True when a failure should be logged."""
        if ok:
            if self.phase == "resetting":
                self.phase = "collecting"
                self.origin = now
                self._candidate = None
            return False
        self.phase = "resetting"
        self.need_reset = True
        self.origin = None
        if now - self._fail_logged < 3.0:
            return False
        self._fail_logged = now
        return True

    def push(self, image, now: float, scan=None) -> None:
        if not self.enabled:
            if image is not None:
                self.committed = image
                self.committed_scan = scan
            return
        if self.phase != "collecting" or self.origin is None or image is None:
            return
        age = now - self.origin
        fill = fill_seconds(self.interval)
        if age + 1e-4 >= fill:
            self._candidate = image
            self._candidate_scan = scan
        if age + 1e-4 >= self.interval:
            if self._candidate is not None:
                self.committed = self._candidate
                self.committed_scan = self._candidate_scan
            self._candidate = None
            self.phase = "resetting"
            self.origin = None
            self.need_reset = True

    @property
    def ready(self) -> bool:
        return (not self.enabled) or self.committed is not None


def _bin_angle(index: int) -> float:
    return float(-FOV / 2.0 + (index + 0.5) * FOV / N_AZ)


def project_near(agent, points: list[tuple[float, float]]) -> np.ndarray:
    """Ego near-channels for world points. Positive bearing is left."""
    near = np.zeros(N_AZ, dtype=np.float32)
    yaw = float(agent.yaw)
    ax, ay = float(agent.x), float(agent.y)
    for x, y in points:
        dx, dy = float(x) - ax, float(y) - ay
        dist = float(math.hypot(dx, dy))
        bx = bearing_bin(dx, dy, yaw)
        if bx is None or dist > SEE_M or dist < 1e-3:
            continue
        near[bx] = max(float(near[bx]), float(np.clip(1.0 - dist / SEE_M, 0.0, 1.0)))
    return near


class SimLidarBank:
    """Sim stand-in for the preview server's cloud.

    Points are remembered in the world. Refresh off keeps them for the whole
    run and reprojects them every step, which is the accumulating map.
    Refresh on commits the points of one interval, feeds that near-vector
    until the next commit, and does not feed the empty moment after a clear.
    """

    def __init__(self, interval: float):
        self.interval = float(interval) if interval > 0 else 0.0
        self.enabled = self.interval > 0
        self.points: list[tuple[float, float, float, tuple[int, int, int]]] = []
        self.held_xy: list[tuple[float, float]] | None = None
        self.held_near: np.ndarray | None = None
        self.held_close = np.zeros(N_AZ, dtype=np.float32)
        self.display: list[tuple[float, float, tuple[int, int, int]]] = []
        self.origin = 0.0
        self._prev_near = np.zeros(N_AZ, dtype=np.float32)

    def set_enabled(self, on: bool, now: float) -> None:
        self.enabled = bool(on) and self.interval > 0
        if on and self.interval <= 0:
            self.interval = DEFAULT_LIDAR_REFRESH
            self.enabled = True
        self.points.clear()
        self.held_xy = None
        self.held_near = None
        self.held_close[:] = 0
        self.display = []
        self.origin = float(now)
        self._prev_near[:] = 0

    def clear(self, now: float) -> None:
        self.points.clear()
        self.held_xy = None
        self.held_near = None
        self.held_close[:] = 0
        self.display = []
        self.origin = float(now)
        self._prev_near[:] = 0

    def update(self, agent, world, now: float) -> None:
        for x, y, color in _sample(agent, world):
            self.points.append((x, y, float(now), color))
        if len(self.points) > 4000:
            self.points = self.points[-4000:]
        if not self.enabled:
            xy = [(p[0], p[1]) for p in self.points]
            self.held_near = project_near(agent, xy)
            self.held_xy = xy
            self.display = [(p[0], p[1], p[3]) for p in self.points]
            return
        age = float(now) - self.origin
        if age + 1e-4 < self.interval:
            return
        window = [p for p in self.points if p[2] >= self.origin - 1e-6]
        if not window:
            self.origin = float(now)
            self.points = []
            return
        xy = [(p[0], p[1]) for p in window]
        near = project_near(agent, xy)
        self.held_close = np.clip((near - self._prev_near) * 2.0, 0.0, 1.0).astype(np.float32)
        self._prev_near = near
        self.held_near = near
        self.held_xy = xy
        self.display = [(p[0], p[1], p[3]) for p in window]
        self.points = []
        self.origin = float(now)

    def apply(self, agent, feat: np.ndarray) -> None:
        from .raw_sense import OFF_CLOSE, OFF_LIDAR

        if self.enabled:
            if self.held_near is None:
                feat[OFF_LIDAR : OFF_LIDAR + N_AZ] = 0
                feat[OFF_CLOSE : OFF_CLOSE + N_AZ] = 0
                return
            feat[OFF_LIDAR : OFF_LIDAR + N_AZ] = self.held_near
            feat[OFF_CLOSE : OFF_CLOSE + N_AZ] = self.held_close
            return
        if self.held_near is not None:
            feat[OFF_LIDAR : OFF_LIDAR + N_AZ] = self.held_near


def _sample(agent, world) -> list[tuple[float, float, tuple[int, int, int]]]:
    yaw = float(agent.yaw)
    out: list[tuple[float, float, tuple[int, int, int]]] = []
    for other in world.agents:
        if getattr(other, "agent_id", None) == agent.agent_id:
            continue
        if bearing_bin(other.x - agent.x, other.y - agent.y, yaw) is None:
            continue
        if math.hypot(other.x - agent.x, other.y - agent.y) > SEE_M:
            continue
        out.append((float(other.x), float(other.y), tuple(int(c) for c in other.color)))
    for obj in getattr(world, "distractors", []):
        if bearing_bin(obj.x - agent.x, obj.y - agent.y, yaw) is None:
            continue
        if math.hypot(obj.x - agent.x, obj.y - agent.y) > SEE_M:
            continue
        rgb = tuple(int(float(c) * 255) for c in obj.color)
        out.append((float(obj.x), float(obj.y), rgb))
    for index in range(N_AZ):
        dist = wall_range(agent.x, agent.y, yaw, index, world.cfg.width, world.cfg.height)
        if dist >= SEE_M:
            continue
        ang = yaw + _bin_angle(index)
        reach = dist * 0.98
        out.append((float(agent.x + math.cos(ang) * reach), float(agent.y + math.sin(ang) * reach), (64, 70, 78)))
    return out
