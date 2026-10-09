#!/usr/bin/env python3
"""Onboard fly brain. Run on the Go2 as ``python3 /root/flybrain/main.py``.

Does not touch the preview server. Does not call Damp or SwitchJoystick.
After start the dog is not walking and the weights are not changing until
the laptop asks.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent.parent, HERE.parent):
    if (candidate / "sim" / "pilot.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from sim.hemifield import DEFAULT_OVERLAP  # noqa: E402


def _pid_path(explicit: str) -> Path | None:
    if explicit:
        return Path(explicit)
    if Path("/root/flybrain").is_dir():
        return Path("/root/flybrain/flybrain.pid")
    return None


def _write_pid(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("%s\n" % os.getpid())
    except OSError:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Onboard fly mushroom body")
    parser.add_argument("--preview", default="http://127.0.0.1:8088")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--state", default="")
    parser.add_argument("--npz", default="")
    parser.add_argument("--pidfile", default="")
    parser.add_argument("--dry", action="store_true", help="No SportClient. For a laptop smoke test.")
    parser.add_argument("--lidar-refresh", type=float, default=1.5)
    parser.add_argument("--self-radius", type=float, default=0.6, help="Ignore lidar returns closer than this, metres.")
    parser.add_argument("--steer", choices=("bilateral", "sectors"), default="bilateral")
    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help="Shared fraction of the field, 0..0.5. 0 is the old hard midline. Default 0.4.",
    )
    args = parser.parse_args(argv)
    stop = {"stop": False}

    def _on_signal(_signum, _frame) -> None:
        stop["stop"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    pid_path = _pid_path(args.pidfile)
    if pid_path is not None:
        _write_pid(pid_path)
    sys.path.insert(0, str(HERE))
    from app import make_brain, serve
    from drive import SportDrive
    from sensors import CloudRanges, JpegEyes

    state = args.state or str(HERE / "state" / "mb_train_state.npz")
    Path(state).parent.mkdir(parents=True, exist_ok=True)
    client = None
    if not args.dry:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        ChannelFactoryInitialize(0)
        client = SportClient()
        client.SetTimeout(3.0)
        client.Init()
    drive = SportDrive(client)
    brain = make_brain(drive, state, args.npz or None, self_radius=args.self_radius, overlap=args.overlap)
    brain.pilot.set_steer(args.steer)
    eyes = JpegEyes(args.preview, interval=args.lidar_refresh)
    eyes.start()
    cloud = CloudRanges(self_radius=args.self_radius)
    cloud.start()
    server = serve(brain, args.port)
    print(f"flybrain on :{args.port}  preview {args.preview}  state {state}", flush=True)
    last_save = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            ranges, stamp = cloud.snapshot()
            if ranges is not None:
                brain.set_cloud(ranges, stamp)
            camera, brain_lidar, _lidar, _error, _hold = eyes.latest()
            if camera is not None and brain_lidar is not None:
                brain.ingest(camera, brain_lidar, now)
            brain.tick(now)
            if stop["stop"]:
                break
            if now - last_save >= 30.0:
                brain.mb.lidar_refresh = float(args.lidar_refresh)
                brain.mb.save(state)
                last_save = now
            time.sleep(0.1)
    except KeyboardInterrupt:
        brain.halt()
    finally:
        brain.halt()
        eyes.stop()
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
