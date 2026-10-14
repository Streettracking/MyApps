#!/usr/bin/env python3
"""Camera and lidar preview on the robot, plus an optional scan for map marks.

Same endpoints as the v1 server already on the dog: ``/camera.jpg``,
``/lidar.jpg`` (``span`` query), ``/lidar/reset`` (204, clears the accumulated
cloud). Adds ``GET /lidar/scan.json``: robot pixel on that JPEG, yaw, meters
per pixel, and the nearest return in each of 8 forward sectors (110° FOV,
sector 0 on the right, positive bearing to the left). An older trainer that
never asks for the JSON keeps working. Deploy with ``robot/deploy_preview_server.py``
and ``GO2_SSH_PASS``; this file has no password.
"""
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

PORT = 8088
camera_jpeg = b""
lidar_jpeg = b""
lidar_span = 8.0
lidar_reset = False
scan_json = b'{"robot_px": null, "yaw_rad": null, "ranges_m": null}'
lock = threading.Lock()


def ruler_steps(px_per_cm):
    candidates = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
    minor = candidates[-1]
    for step in candidates:
        if step * px_per_cm >= 4:
            minor = step
            break
    major = minor
    for step in candidates:
        if step >= minor and step * px_per_cm >= 42:
            major = step
            break
    return minor, major


def camera_loop(client):
    global camera_jpeg
    while True:
        try:
            code, data = client.GetImageSample()
            if code == 0 and data:
                payload = bytes(data)
                if payload:
                    with lock:
                        camera_jpeg = payload
        except Exception:
            time.sleep(0.2)
        time.sleep(0.05)


def lidar_loop():
    global lidar_jpeg
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image, ImageDraw
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from unitree_sdk2py.idl.nav_msgs.msg.dds_._Odometry_ import Odometry_
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_._PointCloud2_ import PointCloud2_
    except Exception as error:
        print("lidar import failed", error, flush=True)
        return

    half = 30.0
    resolution = 0.05
    cells = int(2 * half / resolution)
    hits = np.zeros((cells, cells), dtype=np.uint16)
    levels = np.zeros((cells, cells), dtype=np.uint8)
    trail = np.zeros((cells, cells), dtype=np.uint8)
    pose = None
    latest_xy = np.zeros((0, 2), dtype=np.float32)
    drawn = 0.0
    view_w, view_h = 480, 270
    ruler_x = 46
    ruler_y = 26
    map_w = view_w - ruler_x
    map_h = view_h - ruler_y

    participant = DomainParticipant(0)
    clouds = DataReader(participant, Topic(participant, "rt/utlidar/cloud_deskewed", PointCloud2_))
    odoms = DataReader(participant, Topic(participant, "rt/utlidar/robot_odom", Odometry_))
    print("lidar map open", flush=True)

    def cell_of(x_value, y_value):
        column = int((x_value + half) / resolution)
        row = int((y_value + half) / resolution)
        return row, column

    def remember_pose(sample):
        position = sample.pose.pose.position
        row, column = cell_of(position.x, position.y)
        if 0 <= row < cells and 0 <= column < cells:
            trail[row, column] = 1

    def remember_cloud(sample, robot_xy):
        nonlocal latest_xy
        raw = np.ascontiguousarray(np.asarray(sample.data, dtype=np.uint8))
        step = int(sample.point_step) or 16
        count = len(raw) // step
        if count <= 0:
            return
        xyz = np.ndarray((count, 3), dtype=np.float32, buffer=raw, strides=(step, 4))
        points = xyz[np.isfinite(xyz).all(axis=1)]
        if len(points) == 0:
            return
        if robot_xy is not None:
            delta = points[:, :2] - robot_xy
            distance = np.linalg.norm(delta, axis=1)
            points = points[(distance > 0.35) & (distance < 12.0)]
        if len(points) == 0:
            return
        points = points[(points[:, 2] > -0.4) & (points[:, 2] < 2.2)]
        if len(points) == 0:
            return
        rows = ((points[:, 1] + half) / resolution).astype(np.int32)
        columns = ((points[:, 0] + half) / resolution).astype(np.int32)
        inside = (rows >= 0) & (rows < cells) & (columns >= 0) & (columns < cells)
        rows = rows[inside]
        columns = columns[inside]
        if len(rows) == 0:
            return
        np.add.at(hits, (rows, columns), 1)
        hits[rows, columns] = np.minimum(hits[rows, columns], 40)
        stored = np.clip(points[inside, 2] * 100.0, 0, 255).astype(np.uint8)
        np.maximum.at(levels, (rows, columns), stored)
        latest_xy = np.array(points[inside, :2], dtype=np.float32, copy=True)

    def window_center():
        if pose is not None:
            position = pose.pose.pose.position
            return float(position.x), float(position.y)
        occupied = np.nonzero(hits > 0)
        if len(occupied[0]) == 0:
            return 0.0, 0.0
        return (
            (float(occupied[1].mean()) + 0.5) * resolution - half,
            (float(occupied[0].mean()) + 0.5) * resolution - half,
        )

    def render():
        with lock:
            span = lidar_span
        span_h = span * (map_h / map_w)
        center_x, center_y = window_center()
        x0 = center_x - span / 2.0
        x1 = center_x + span / 2.0
        y0 = center_y - span_h / 2.0
        y1 = center_y + span_h / 2.0
        src_c0 = max(int(math.floor((x0 + half) / resolution)), 0)
        src_c1 = min(int(math.ceil((x1 + half) / resolution)), cells)
        src_r0 = max(int(math.floor((y0 + half) / resolution)), 0)
        src_r1 = min(int(math.ceil((y1 + half) / resolution)), cells)
        image = Image.new("RGB", (view_w, view_h), (28, 25, 23))
        if src_c1 > src_c0 and src_r1 > src_r0:
            crop_hits = hits[src_r0:src_r1, src_c0:src_c1]
            crop_levels = levels[src_r0:src_r1, src_c0:src_c1]
            crop_trail = trail[src_r0:src_r1, src_c0:src_c1]
            rgb = np.zeros(crop_hits.shape + (3,), dtype=np.uint8)
            rgb[:] = (28, 25, 23)
            floor = (crop_hits > 0) & (crop_levels < 18)
            wall = (crop_hits > 0) & (crop_levels >= 18)
            walked = (crop_trail > 0) & ~wall
            rgb[floor] = (62, 58, 54)
            rgb[walked] = (196, 164, 110)
            if np.any(wall):
                shade = np.clip((crop_levels.astype(np.float32) - 18.0) / 120.0, 0.0, 1.0)
                rgb[wall, 0] = (36 + 50 * shade[wall]).astype(np.uint8)
                rgb[wall, 1] = (110 + 120 * shade[wall]).astype(np.uint8)
                rgb[wall, 2] = (64 + 36 * shade[wall]).astype(np.uint8)
            rgb = rgb[::-1]
            world_x0 = src_c0 * resolution - half
            world_y1 = src_r1 * resolution - half
            dest_x = int(round(ruler_x + (world_x0 - x0) / span * map_w))
            dest_y = int(round((y1 - world_y1) / span_h * map_h))
            dest_w = max(int(round((src_c1 - src_c0) * resolution / span * map_w)), 1)
            dest_h = max(int(round((src_r1 - src_r0) * resolution / span_h * map_h)), 1)
            resized = Image.fromarray(rgb, "RGB").resize((dest_w, dest_h), Image.NEAREST)
            image.paste(resized, (dest_x, dest_y))
        draw = ImageDraw.Draw(image)
        if pose is not None:
            position = pose.pose.pose.position
            orientation = pose.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
            )
            px = ruler_x + (position.x - x0) / span * map_w
            py = (y1 - position.y) / span_h * map_h
            nose = 11
            side = 6
            draw.polygon(
                [
                    (px + math.cos(yaw) * nose, py - math.sin(yaw) * nose),
                    (px + math.cos(yaw + 2.4) * side, py - math.sin(yaw + 2.4) * side),
                    (px + math.cos(yaw - 2.4) * side, py - math.sin(yaw - 2.4) * side),
                ],
                fill=(250, 250, 249),
            )
        draw_ruler(draw, span, span_h)
        publish_scan(center_x, center_y, span, span_h)
        out = BytesIO()
        image.save(out, format="JPEG", quality=80)
        with lock:
            global lidar_jpeg
            lidar_jpeg = out.getvalue()

    def draw_ruler(draw, span, span_h):
        from PIL import ImageFont

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
        ink = (250, 250, 249)
        dim = (168, 162, 158)
        width_cm = int(round(span * 100.0))
        height_cm = int(round(span_h * 100.0))
        px_per_cm_x = map_w / width_cm
        px_per_cm_y = map_h / height_cm
        minor, major = ruler_steps(px_per_cm_x)
        minor_y, major_y = ruler_steps(px_per_cm_y)
        origin_x = ruler_x + map_w / 2.0
        origin_y = map_h / 2.0
        draw.rectangle((0, map_h, view_w, view_h), fill=(28, 25, 23))
        draw.rectangle((0, 0, ruler_x, view_h), fill=(28, 25, 23))
        draw.line((ruler_x, map_h, view_w - 1, map_h), fill=ink)
        draw.line((ruler_x, 0, ruler_x, map_h), fill=ink)
        for mark in range(0, width_cm // 2 + 1, minor):
            signs = (1,) if mark == 0 else (1, -1)
            for sign in signs:
                x = origin_x + sign * mark * px_per_cm_x
                if x < ruler_x - 1 or x > view_w:
                    continue
                is_major = mark % major == 0
                length = 10 if is_major else 5
                draw.line((x, map_h, x, map_h + length), fill=ink)
                if is_major and ruler_x + 10 < x < view_w - 36:
                    draw.text((x - 8, map_h + 11), str(sign * mark), fill=dim, font=font)
        for mark in range(0, height_cm // 2 + 1, minor_y):
            signs = (1,) if mark == 0 else (1, -1)
            for sign in signs:
                y = origin_y - sign * mark * px_per_cm_y
                if y < 0 or y > map_h + 1:
                    continue
                is_major = mark % major_y == 0
                length = 10 if is_major else 5
                draw.line((ruler_x - length, y, ruler_x, y), fill=ink)
                if is_major and 10 < y < map_h - 8:
                    draw.text((2, y - 6), str(sign * mark), fill=dim, font=font)
        draw.text((view_w - 24, map_h + 11), "см", fill=ink, font=font)

    def publish_scan(center_x, center_y, span, span_h):
        global scan_json
        fov = math.radians(110.0)
        yaw = None
        robot_px = None
        ranges = None
        if pose is not None:
            position = pose.pose.pose.position
            orientation = pose.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
            )
            px = ruler_x + (position.x - (center_x - span / 2.0)) / span * map_w
            py = ((center_y + span_h / 2.0) - position.y) / span_h * map_h
            robot_px = [round(px, 2), round(py, 2)]
            ranges = [None] * 8
            if len(latest_xy):
                origin = np.array([position.x, position.y], dtype=np.float32)
                delta = latest_xy - origin
                dist = np.linalg.norm(delta, axis=1)
                ang = np.arctan2(delta[:, 1], delta[:, 0]) - yaw
                ang = (ang + np.pi) % (2.0 * np.pi) - np.pi
                half_bin = fov / 16.0
                for index in range(8):
                    center = -fov / 2.0 + (index + 0.5) * fov / 8.0
                    delta_ang = (ang - center + np.pi) % (2.0 * np.pi) - np.pi
                    picked = (np.abs(delta_ang) <= half_bin) & (dist > 0.35) & (dist < 12.0)
                    if np.any(picked):
                        ranges[index] = round(float(dist[picked].min()), 3)
        payload = {
            "image_w": view_w,
            "image_h": view_h,
            "robot_px": robot_px,
            "yaw_rad": None if yaw is None else round(yaw, 5),
            "meters_per_px": span / map_w,
            "span_m": span,
            "fov_rad": fov,
            "axes": "world_x_right_y_up",
            "bearings_rad": [round(-fov / 2.0 + (i + 0.5) * fov / 8.0, 5) for i in range(8)],
            "ranges_m": ranges,
        }
        encoded = json.dumps(payload).encode("utf-8")
        with lock:
            scan_json = encoded

    while True:
        with lock:
            global lidar_reset
            should_reset = lidar_reset
            lidar_reset = False
        if should_reset:
            hits.fill(0)
            levels.fill(0)
            trail.fill(0)
            latest_xy = np.zeros((0, 2), dtype=np.float32)
            drawn = 0.0
            try:
                render()
            except Exception as error:
                print("lidar reset", type(error).__name__, error, flush=True)
        sample = None
        try:
            for item in odoms.take():
                pose = item
                remember_pose(item)
            for item in clouds.take():
                sample = item
        except Exception as error:
            print("lidar read", type(error).__name__, error, flush=True)
            time.sleep(0.2)
            continue
        if sample is None or not sample.data:
            time.sleep(0.03)
            continue
        try:
            robot_xy = None
            if pose is not None:
                robot_xy = np.array(
                    [pose.pose.pose.position.x, pose.pose.pose.position.y],
                    dtype=np.float32,
                )
            remember_cloud(sample, robot_xy)
            now = time.time()
            if now - drawn >= 0.12:
                drawn = now
                render()
        except Exception as error:
            print("lidar render", type(error).__name__, error, flush=True)
            time.sleep(0.2)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global lidar_span
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        if parsed.path.startswith("/lidar/reset"):
            global lidar_reset
            with lock:
                lidar_reset = True
            self.send_response(204)
            self.end_headers()
            return
        if parsed.path.startswith("/camera.jpg"):
            body = camera_jpeg
            content_type = "image/jpeg"
        elif parsed.path.startswith("/lidar.jpg"):
            query = parse_qs(parsed.query)
            if "span" in query:
                try:
                    requested = float(query["span"][0])
                except ValueError:
                    requested = lidar_span
                with lock:
                    lidar_span = min(40.0, max(1.0, requested))
            body = lidar_jpeg
            content_type = "image/jpeg"
        elif parsed.path.startswith("/lidar/scan.json"):
            body = scan_json
            content_type = "application/json"
        else:
            self.send_response(404)
            self.end_headers()
            return
        if not body:
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


def main():
    ChannelFactoryInitialize(0)
    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()
    threading.Thread(target=camera_loop, args=(video,), daemon=True).start()
    threading.Thread(target=lidar_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"preview on {PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
