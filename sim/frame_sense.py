"""Turn camera and lidar pictures into the same 72-d vector the sim uses.

Sim learning calls ``render_view``. The live trainer calls ``features_from_frames``
on the JPEGs from ``http://<robot>:8088/camera.jpg`` and ``lidar.jpg``.
Both produce ``N_RAW`` floats. Neither vector contains a class label.
"""

from __future__ import annotations

import numpy as np

from .raw_sense import (
    FOV,
    N_AZ,
    N_RAW,
    OFF_BODY,
    OFF_CLOSE,
    OFF_FLOOR,
    OFF_LIDAR,
    OFF_MOTION,
    SEE_M,
    bearing_bin,
)

PREVIEW_CAM = (320, 180)  # w, h
PREVIEW_LIDAR = (240, 240)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode JPEG or PNG bytes to HxWx3 uint8 RGB. Raises ValueError on failure."""
    import io

    import pygame

    if not pygame.get_init():
        pygame.init()
    try:
        surf = pygame.image.load(io.BytesIO(data))
    except pygame.error as exc:
        raise ValueError(f"could not decode image ({len(data)} bytes): {exc}") from exc
    try:
        raw = pygame.image.tobytes(surf, "RGB")
    except pygame.error:
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1))
        raw = pygame.image.tobytes(surf.convert(24), "RGB")
    w, h = surf.get_size()
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()


def _sector_means(image: np.ndarray, y0: int, y1: int) -> np.ndarray:
    h, w = image.shape[:2]
    y0 = int(np.clip(y0, 0, h))
    y1 = int(np.clip(y1, y0 + 1, h))
    band = image[y0:y1].astype(np.float32) / 255.0
    out = np.zeros((N_AZ, 3), dtype=np.float32)
    for i in range(N_AZ):
        x0 = int(i * w / N_AZ)
        x1 = max(x0 + 1, int((i + 1) * w / N_AZ))
        out[i] = band[:, x0:x1].mean(axis=(0, 1))
    return out


def _residual(means: np.ndarray) -> np.ndarray:
    """Drop the scene-wide color so only a local object lights a sector."""
    res = means - np.median(means, axis=0, keepdims=True)
    return np.clip(res * 3.0, 0.0, 1.0).astype(np.float32)


def _lidar_near(lidar: np.ndarray) -> np.ndarray:
    """Top-down lidar picture. Robot at center, forward is up, bright = return."""
    if lidar.ndim == 3:
        gray = lidar.astype(np.float32).mean(axis=2)
    else:
        gray = lidar.astype(np.float32)
    h, w = gray.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rad = max(4.0, 0.92 * min(cx, cy))
    thr = float(max(gray.mean() + 0.35 * gray.std(), np.percentile(gray, 75)))
    near = np.zeros(N_AZ, dtype=np.float32)
    steps = np.linspace(0.08, 1.0, 36)
    for i in range(N_AZ):
        ang = -FOV / 2.0 + (i + 0.5) * FOV / N_AZ
        hit = None
        for s in steps:
            x = int(round(cx + np.sin(ang) * rad * s))
            y = int(round(cy - np.cos(ang) * rad * s))
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            if gray[y, x] >= thr:
                hit = float(s)
                break
        if hit is not None:
            dist = hit * SEE_M
            near[i] = float(np.clip(1.0 - dist / SEE_M, 0.0, 1.0))
    return near


def features_from_frames(
    camera: np.ndarray,
    lidar: np.ndarray | None,
    prev_camera: np.ndarray | None = None,
    prev_near: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Map one camera frame and one lidar map into the sim feature vector.

    Returns ``(feat, ego_speed, near)``. ``ego_speed`` is a proprioceptive
    stand-in from the whole-frame change: the preview HTTP API has no gait
    topic. It is not a label of what is in view.
    """
    if camera.ndim != 3 or camera.shape[2] < 3:
        raise ValueError("camera frame must be HxWx3")
    feat = np.zeros(N_RAW, dtype=np.float32)
    h = camera.shape[0]
    body = _residual(_sector_means(camera, 0, int(h * 0.45)))
    floor = _residual(_sector_means(camera, int(h * 0.55), h))
    for i in range(N_AZ):
        feat[OFF_BODY + i * 3 : OFF_BODY + i * 3 + 3] = body[i]
        feat[OFF_FLOOR + i * 3 : OFF_FLOOR + i * 3 + 3] = floor[i]

    ego = 0.0
    if prev_camera is not None and prev_camera.shape == camera.shape:
        diff = np.abs(camera.astype(np.float32) - prev_camera.astype(np.float32)) / 255.0
        ego = float(np.clip(diff.mean() * 6.0, 0.0, 1.2))
        h2 = diff.shape[0]
        upper = diff[: int(h2 * 0.45)]
        w = upper.shape[1]
        mot = np.zeros(N_AZ, dtype=np.float32)
        for i in range(N_AZ):
            x0 = int(i * w / N_AZ)
            x1 = max(x0 + 1, int((i + 1) * w / N_AZ))
            mot[i] = float(upper[:, x0:x1].mean())
        mot = np.clip((mot - np.median(mot)) * 4.0, 0.0, 1.0)
        feat[OFF_MOTION : OFF_MOTION + N_AZ] = mot

    near = np.zeros(N_AZ, dtype=np.float32)
    if lidar is not None and lidar.size:
        near = _lidar_near(lidar)
        feat[OFF_LIDAR : OFF_LIDAR + N_AZ] = near
        if prev_near is not None and len(prev_near) == N_AZ:
            closing = np.clip((near - prev_near) * 2.0, 0.0, 1.0)
            feat[OFF_CLOSE : OFF_CLOSE + N_AZ] = closing
    return feat, ego, near


def sim_previews(agent, world, trail=None) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic camera and lidar pictures for the sim monitor. Not the feature vector."""
    cw, ch = PREVIEW_CAM
    cam = np.zeros((ch, cw, 3), dtype=np.uint8)
    horizon = int(ch * 0.46)
    cam[:horizon] = (28, 40, 58)
    cam[horizon:] = (48, 44, 38)
    cam[horizon - 1 : horizon + 1] = (70, 78, 88)

    def paint(obj_x, obj_y, radius, color, body: bool) -> None:
        dx, dy = obj_x - agent.x, obj_y - agent.y
        dist = float(np.hypot(dx, dy))
        bx = bearing_bin(dx, dy, float(agent.yaw))
        if bx is None or dist > SEE_M or dist < 1e-3:
            return
        col_w = cw / N_AZ
        cx = int((bx + 0.5) * col_w)
        half = max(6, int(col_w * (0.55 + radius / max(dist, 0.4))))
        if body:
            top = int(horizon * 0.25)
            bot = horizon - 4
        else:
            top = horizon + 8
            bot = min(ch - 4, horizon + 28 + int(40 * (1.0 - dist / SEE_M)))
        x0 = max(0, cx - half)
        x1 = min(cw, cx + half)
        cam[top:bot, x0:x1] = color

    for other in world.agents:
        if other.agent_id == agent.agent_id:
            continue
        paint(other.x, other.y, 0.28, other.color, True)
    for obj in world.distractors:
        rgb = tuple(int(c * 255) for c in obj.color)
        paint(obj.x, obj.y, obj.radius, rgb, False)

    side = PREVIEW_LIDAR[0]
    lid = np.zeros((side, side, 3), dtype=np.uint8)
    lid[:] = (14, 16, 22)
    cx = cy = side // 2
    scale = (side * 0.42) / SEE_M
    yy, xx = np.ogrid[:side, :side]
    for ring in (1.0, 2.0, 3.0):
        r = ring * scale
        mask = np.abs(np.hypot(xx - cx, yy - cy) - r) < 1.1
        lid[mask] = (36, 42, 52)
    yaw = float(agent.yaw)
    if trail:
        for tx, ty, color in trail:
            dx, dy = tx - agent.x, ty - agent.y
            c, s = float(np.cos(yaw)), float(np.sin(yaw))
            forward = c * dx + s * dy
            left = -s * dx + c * dy
            px = int(cx - left * scale)
            py = int(cy - forward * scale)
            if 1 <= px < side - 1 and 1 <= py < side - 1:
                dim = tuple(max(28, int(c) // 3) for c in color)
                lid[py - 1 : py + 2, px - 1 : px + 2] = dim
    for other in list(world.agents) + list(world.distractors):
        if getattr(other, "agent_id", None) == agent.agent_id:
            continue
        dx, dy = other.x - agent.x, other.y - agent.y
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        forward = c * dx + s * dy
        left = -s * dx + c * dy
        px = int(cx - left * scale)
        py = int(cy - forward * scale)
        if not (2 <= px < side - 2 and 2 <= py < side - 2):
            continue
        if hasattr(other, "agent_id"):
            color = other.color
            rad = 5
        else:
            color = tuple(int(ch * 255) for ch in other.color)
            rad = 3
        lid[py - rad : py + rad + 1, px - rad : px + rad + 1] = color
    lid[cy - 2 : cy + 3, cx - 2 : cx + 3] = (230, 230, 236)
    lid[cy - 10 : cy - 2, cx - 1 : cx + 2] = (180, 200, 220)
    return cam, lid
