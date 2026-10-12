#!/usr/bin/env python3
"""YOLO box service for the trainer. Run this from the laptop venv, not from the exe.

Listens on 127.0.0.1 only. POST /detect with a JPEG body, GET /health.
The trainer does not import this module.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _decode(raw: bytes):
    import numpy as np

    try:
        import cv2

        arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("jpeg")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    except Exception:
        import io

        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def _boxes(model, raw: bytes) -> list[dict]:
    image = _decode(raw)
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return []
    found = []
    results = model.predict(image, verbose=False, conf=0.05)
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            found.append(
                {
                    "x0": float(xyxy[0]) / float(width),
                    "y0": float(xyxy[1]) / float(height),
                    "x1": float(xyxy[2]) / float(width),
                    "y1": float(xyxy[3]) / float(height),
                    "conf": float(box.conf[0]),
                }
            )
    return found


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _json(self, payload: dict, code: int = 200) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/health":
                self._json({"ok": False}, 404)
                return
            self._json({"ok": True})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/detect":
                self._json({"ok": False}, 404)
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                boxes = _boxes(model, raw)
            except Exception as exc:
                self._json({"ok": False, "error": str(exc), "boxes": []}, 400)
                return
            self._json({"ok": True, "boxes": boxes})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local YOLO service for the Go2 trainer")
    parser.add_argument("--weights", required=True, help="best.pt from the laptop training run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost"):
        print("Refusing to listen outside localhost.", flush=True)
        return 2
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Нет ultralytics. Запускай этот скрипт из venv YOLO, не из Python тренажёра.", flush=True)
        return 2
    model = YOLO(args.weights)
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(model))
    print("yolo teacher on http://%s:%s" % (args.host, int(args.port)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
