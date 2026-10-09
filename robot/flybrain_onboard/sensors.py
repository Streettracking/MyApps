"""Camera and lidar from the preview already running on port 8088.

JPEGs come from ``http://127.0.0.1:8088``. This process does not open
``VideoClient``. A read-only DDS cloud is optional and only supplies
range for the one-metre stop. The brain still sees the JPEG.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from io import BytesIO

import numpy as np

from sim.lidar_fresh import FreshWindow
from sim.pilot import ego_sector_ranges


def jpeg_complete(data: bytes) -> bool:
    """True when the buffer is a JPEG with the end marker, not a half write."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return False
    tail = data.rstrip(b"\x00\r\n\t ")
    return tail.endswith(b"\xff\xd9")


def _imdecode_quiet(cv2, buf):
    """Decode one JPEG. libjpeg warnings stay off the process log.

    A half-written preview frame makes cv2 print ``Corrupt JPEG data``
    on the C stderr. That text was filling ``flybrain.log``. The warning
    is captured and the frame is rejected.
    """
    import os

    read_fd, write_fd = os.pipe()
    saved = os.dup(2)
    bgr = None
    try:
        try:
            os.dup2(write_fd, 2)
            os.close(write_fd)
            write_fd = -1
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            if write_fd >= 0:
                os.close(write_fd)
        chunks = []
        while True:
            piece = os.read(read_fd, 4096)
            if not piece:
                break
            chunks.append(piece)
    finally:
        os.close(read_fd)
    err = b"".join(chunks)
    bad = b"Corrupt JPEG" in err or b"premature end" in err
    return bgr, bad


def decode_jpeg(data: bytes) -> np.ndarray:
    """RGB image. OpenCV or PIL. Not pygame.

    A frame without the JPEG end marker ``FFD9`` is still being written
    by the preview server. It is skipped before a decoder can warn.
    """
    if not jpeg_complete(data):
        raise ValueError("incomplete jpeg")
    buf = np.frombuffer(data, dtype=np.uint8)
    try:
        import cv2
    except Exception:
        cv2 = None
    if cv2 is not None:
        bgr, corrupt = _imdecode_quiet(cv2, buf)
        if corrupt or bgr is None:
            raise ValueError("corrupt jpeg")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image

    image = Image.open(BytesIO(data)).convert("RGB")
    return np.asarray(image)


def fetch_bytes(url: str, timeout: float = 1.5) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flybrain-onboard"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class JpegEyes:
    """Background fetch of camera.jpg and lidar.jpg, with the fresh lidar window."""

    def __init__(self, base: str = "http://127.0.0.1:8088", interval: float = 1.5):
        self.base = base.rstrip("/")
        self.interval = float(interval)
        self.fresh = FreshWindow(self.interval if self.interval > 0 else 0.0)
        self.error = ""
        self.camera = None
        self.lidar = None
        self.brain_lidar = None
        self.hold = ""
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="jpeg-eyes", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.5)

    def _reset(self, quiet: bool) -> None:
        try:
            fetch_bytes(self.base + "/lidar/reset", timeout=1.5)
            self.fresh.ack(True, time.monotonic())
        except Exception:
            self.fresh.ack(False, time.monotonic())

    def _loop(self) -> None:
        while not self._stop:
            now = time.monotonic()
            try:
                cam_b = fetch_bytes(self.base + "/camera.jpg", timeout=1.5)
                lid_b = fetch_bytes(self.base + "/lidar.jpg", timeout=1.5)
                cam = decode_jpeg(cam_b)
                lid = decode_jpeg(lid_b)
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
                time.sleep(0.3)
                continue
            window = self.fresh
            brain = lid
            hold = ""
            if self.interval > 0:
                if window.phase == "idle":
                    window.configure(self.interval)
                    window.start(now)
                window.push(lid, now, None)
                if window.poll_reset(now):
                    self._reset(True)
                if window.committed is None:
                    brain = None
                    hold = "набор свежего лидара"
                else:
                    brain = window.committed
            with self._lock:
                self.camera = cam
                self.lidar = lid
                self.brain_lidar = brain
                self.hold = hold
                self.error = ""
            time.sleep(0.05)

    def latest(self):
        with self._lock:
            return self.camera, self.brain_lidar, self.lidar, self.error, self.hold


class CloudRanges:
    """Optional nearest range per camera sector from the deskewed cloud.

    If Cyclone or the Unitree types are missing, ``start`` does nothing and
    the JPEG ranges stay in charge.
    """

    def __init__(self):
        self.ranges = None
        self.stamp = 0.0
        self.ok = False
        self._stop = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        try:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.sub import DataReader
            from cyclonedds.topic import Topic
            from unitree_sdk2py.idl.nav_msgs.msg.dds_._Odometry_ import Odometry_
            from unitree_sdk2py.idl.sensor_msgs.msg.dds_._PointCloud2_ import PointCloud2_
        except Exception:
            return False
        self.ok = True
        threading.Thread(
            target=self._loop,
            args=(DomainParticipant, DataReader, Topic, PointCloud2_, Odometry_),
            name="cloud-ranges",
            daemon=True,
        ).start()
        return True

    def snapshot(self):
        with self._lock:
            return self.ranges, self.stamp

    def _loop(self, DomainParticipant, DataReader, Topic, PointCloud2_, Odometry_) -> None:
        participant = DomainParticipant(0)
        clouds = DataReader(participant, Topic(participant, "rt/utlidar/cloud_deskewed", PointCloud2_))
        odoms = DataReader(participant, Topic(participant, "rt/utlidar/robot_odom", Odometry_))
        pose = None
        while not self._stop:
            for sample in odoms.take(8):
                if sample is not None:
                    pose = sample
            if pose is None:
                time.sleep(0.05)
                continue
            for sample in clouds.take(2):
                if sample is None:
                    continue
                ranges = _ranges_of(sample, pose)
                if ranges is None:
                    continue
                with self._lock:
                    self.ranges = ranges
                    self.stamp = time.monotonic()
            time.sleep(0.05)


def _yaw_of(orientation) -> float:
    w = float(orientation.w)
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _ranges_of(sample, pose):
    try:
        raw = np.ascontiguousarray(np.asarray(sample.data, dtype=np.uint8))
        step = int(sample.point_step) or 16
        count = len(raw) // step
        if count <= 0:
            return None
        xyz = np.ndarray((count, 3), dtype=np.float32, buffer=raw, strides=(step, 4))
        points = xyz[np.isfinite(xyz).all(axis=1)]
        points = points[(points[:, 2] > -0.4) & (points[:, 2] < 2.2)]
        if len(points) == 0:
            return None
        origin = np.array([float(pose.pose.pose.position.x), float(pose.pose.pose.position.y)], dtype=np.float32)
        yaw = _yaw_of(pose.pose.pose.orientation)
        return ego_sector_ranges(points[:, :2], origin, yaw)
    except Exception:
        return None


def cloud_forward(ranges) -> float | None:
    if not ranges:
        return None
    found = [ranges[i] for i in (3, 4) if i < len(ranges) and ranges[i] is not None]
    if not found:
        return None
    return float(min(found))
