#!/usr/bin/env python3
"""Upload robot_preview_server_v2.py and start it on the Go2.

The password is only ``GO2_SSH_PASS``. Nothing in this file is a secret.
The process listens on port 8088 and replaces a previous copy started from
``/tmp/robot_preview_server.py`` or ``/tmp/robot_preview_server_v2.py``.

    GO2_SSH_PASS=... python robot/deploy_preview_server.py

Optional: ``GO2_HOST`` (default 192.168.35.213) and ``GO2_SSH_USER`` (default root).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import urllib.request

HOST = os.environ.get("GO2_HOST", "192.168.35.213")
USER = os.environ.get("GO2_SSH_USER", "root")
PASSWORD = os.environ.get("GO2_SSH_PASS", "")
LOCAL = Path(__file__).with_name("robot_preview_server_v2.py")
REMOTE = "/tmp/robot_preview_server_v2.py"


def main() -> int:
    if not PASSWORD:
        print("Set GO2_SSH_PASS. This script does not contain a password.", file=sys.stderr)
        return 2
    if not LOCAL.is_file():
        print(f"Missing {LOCAL}", file=sys.stderr)
        return 2
    try:
        import paramiko
    except ImportError:
        print("pip install paramiko", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=12)
    sftp = client.open_sftp()
    sftp.put(str(LOCAL), REMOTE)
    sftp.close()
    client.exec_command(
        "pkill -f /tmp/robot_preview_server.py || true; "
        "pkill -f /tmp/robot_preview_server_v2.py || true"
    )
    time.sleep(1)
    cmd = (
        "PYTHONPATH=/unitree/module/pet_go:/root/go2_flask_api "
        f"nohup python3 {REMOTE} >/tmp/preview.log 2>&1 &"
    )
    client.exec_command(cmd)
    time.sleep(5)
    _, stdout, _ = client.exec_command("tail -20 /tmp/preview.log; ss -tlnp | grep 8088 || true")
    time.sleep(2)
    print(stdout.read().decode("utf-8", errors="replace")[:2000])
    client.close()

    for path in ("/camera.jpg", "/lidar.jpg", "/lidar/scan.json"):
        url = f"http://{HOST}:8088{path}"
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                data = response.read()
                print(path, response.status, len(data))
        except Exception as exc:
            print(path, "fail", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
