#!/usr/bin/env python3
"""Onboard fly brain. Run on the Go2 as ``python3 /root/flybrain/main.py``.

Does not touch the preview server. Does not call Damp or SwitchJoystick.
After start the dog is not walking and the weights are not changing until
the laptop asks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent.parent, HERE.parent):
    if (candidate / "sim" / "pilot.py").is_file():
        sys.path.insert(0, str(candidate))
        break


def main(argv=None) -> int:
    sys.path.insert(0, str(HERE))
    from app import make_brain, serve
    from drive import SportDrive
    from sensors import CloudRanges, JpegEyes

    parser = argparse.ArgumentParser(description="Onboard fly mushroom body")
    parser.add_argument("--preview", default="http://127.0.0.1:8088")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--state", default="")
    parser.add_argument("--npz", default="")
    parser.add_argument("--dry", action="store_true", help="No SportClient. For a laptop smoke test.")
    parser.add_argument("--lidar-refresh", type=float, default=1.5)
    args = parser.parse_args(argv)
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
    brain = make_brain(drive, state, args.npz or None)
    eyes = JpegEyes(args.preview, interval=args.lidar_refresh)
    eyes.start()
    cloud = CloudRanges()
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
