#!/usr/bin/env python3
"""Copy the onboard fly brain to one Go2 and start it by hand.

    GO2_SSH_PASS=... python robot/deploy_flybrain.py start
    GO2_SSH_PASS=... python robot/deploy_flybrain.py start --state logs/mb_train_state.npz
    GO2_SSH_PASS=... python robot/deploy_flybrain.py stop
    GO2_SSH_PASS=... python robot/deploy_flybrain.py status

``start`` with no ``--state`` leaves a clean brain. The connectome copied
to the dog is ``connectome_mb_v1_np1.npz`` (no pickled object arrays).

The password is only ``GO2_SSH_PASS``. This script does not install
packages, does not write a systemd unit, and does not touch the preview
server, its unit, or ``/tmp/robot_preview_server.py``.

Remote files stay under ``/root/flybrain/``. The process is ``nohup``
``python3 /root/flybrain/main.py``. Stop kills only that pid, and only
when ``/proc/<pid>/cmdline`` contains ``flybrain``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOST = os.environ.get("GO2_HOST", "192.168.35.213")
USER = os.environ.get("GO2_SSH_USER", "root")
PASSWORD = os.environ.get("GO2_SSH_PASS", "")
REMOTE = "/root/flybrain"
ROOT = Path(__file__).resolve().parents[1]
ONBOARD = ROOT / "robot" / "flybrain_onboard"

SIM_FILES = (
    "mb_runtime.py",
    "mb_train.py",
    "mb_confidence.py",
    "raw_sense.py",
    "learn_flash.py",
    "frame_sense.py",
    "lidar_fresh.py",
    "map_marks.py",
    "pilot.py",
    "npz_compat.py",
)

FORBIDDEN = (
    "preview_server",
    "robot_preview",
    "/tmp/robot_preview",
    "robot_preview_server.service",
)


def _reject(cmd: str) -> None:
    low = cmd.lower()
    for bad in FORBIDDEN:
        if bad in low:
            raise SystemExit(f"refusing a remote command that mentions {bad}")


def _run(client, cmd: str, wait: float = 0.5) -> str:
    import time

    _reject(cmd)
    _stdin, stdout, stderr = client.exec_command(cmd)
    time.sleep(wait)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return (out + err).strip()


def _mkdir(sftp, path: str) -> None:
    parts = path.strip("/").split("/")
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def _connectome_np1() -> Path:
    """Pickle-free connectome. Built from the numpy 2 file when missing."""
    dest = ROOT / "artifacts" / "connectome_mb_v1_np1.npz"
    src = ROOT / "artifacts" / "connectome_mb_v1.npz"
    if dest.is_file():
        return dest
    sys.path.insert(0, str(ROOT))
    from sim.npz_compat import write_pickle_free

    return write_pickle_free(src, dest)


def _upload(sftp, state_path: str = "") -> None:
    _mkdir(sftp, REMOTE + "/sim")
    _mkdir(sftp, REMOTE + "/artifacts")
    _mkdir(sftp, REMOTE + "/state")
    for name in ("main.py", "app.py", "drive.py", "sensors.py", "__init__.py"):
        local = ONBOARD / name
        if local.is_file():
            sftp.put(str(local), f"{REMOTE}/{name}")
    with sftp.file(REMOTE + "/sim/__init__.py", "w") as handle:
        handle.write("")
    for name in SIM_FILES:
        sftp.put(str(ROOT / "sim" / name), f"{REMOTE}/sim/{name}")
    sftp.put(str(_connectome_np1()), REMOTE + "/artifacts/connectome_mb_v1_np1.npz")
    if state_path:
        import tempfile

        sys.path.insert(0, str(ROOT))
        from sim.npz_compat import write_pickle_free

        plain = Path(tempfile.mkdtemp(prefix="flybrain-state-")) / "mb_train_state.npz"
        write_pickle_free(state_path, plain)
        sftp.put(str(plain), REMOTE + "/state/mb_train_state.npz")


def _start_cmd() -> str:
    return (
        "bash -lc '"
        "pid=$(cat /root/flybrain/flybrain.pid 2>/dev/null || true); "
        "if [ -n \"$pid\" ] && [ -r /proc/$pid/cmdline ] && "
        "tr \"\\0\" \" \" < /proc/$pid/cmdline | grep -q flybrain; then "
        "echo already; exit 0; fi; "
        "cd /root/flybrain && "
        "PYTHONPATH=/root/flybrain:/unitree/module/pet_go:/root/go2_flask_api "
        "nohup python3 /root/flybrain/main.py >> /root/flybrain/flybrain.log 2>&1 & "
        "echo $! > /root/flybrain/flybrain.pid; echo started'"
    )


def _stop_cmd() -> str:
    return (
        "bash -lc '"
        "pid=$(cat /root/flybrain/flybrain.pid 2>/dev/null || true); "
        "if [ -n \"$pid\" ] && [ -r /proc/$pid/cmdline ] && "
        "tr \"\\0\" \" \" < /proc/$pid/cmdline | grep -q flybrain; then "
        "kill $pid; echo stopped; else echo not-running; fi'"
    )


def _status_cmd() -> str:
    return (
        "bash -lc '"
        "pid=$(cat /root/flybrain/flybrain.pid 2>/dev/null || true); "
        "echo pid=$pid; "
        "if [ -n \"$pid\" ] && [ -r /proc/$pid/cmdline ]; then "
        "tr \"\\0\" \" \" < /proc/$pid/cmdline; echo; else echo not-running; fi; "
        "tail -n 20 /root/flybrain/flybrain.log 2>/dev/null || true'"
    )


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Copy the onboard fly brain and start it by hand")
    parser.add_argument("action", nargs="?", default="start", choices=("start", "stop", "status"))
    parser.add_argument(
        "--state",
        default="",
        help="mb_train_state.npz to copy onto the dog. Omit this for a clean brain.",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    action = args.action
    state_path = args.state
    if state_path and not Path(state_path).is_file():
        print("No state file at %s" % state_path, file=sys.stderr)
        return 2
    if not PASSWORD:
        print("Set GO2_SSH_PASS. This script does not contain a password.", file=sys.stderr)
        return 2
    try:
        import paramiko
    except ImportError:
        print("pip install paramiko", file=sys.stderr)
        return 2
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=12)
    try:
        if action == "start":
            sftp = client.open_sftp()
            try:
                _upload(sftp, state_path)
            finally:
                sftp.close()
            if state_path:
                print("state %s" % state_path)
            else:
                print(_run(client, "rm -f /root/flybrain/state/mb_train_state.npz"))
                print("clean brain: no --state file copied")
            print(_run(client, "rm -f /root/flybrain/artifacts/connectome_mb_v1.npz"))
            print(_run(client, _start_cmd(), wait=1.5))
        elif action == "stop":
            print(_run(client, _stop_cmd()))
        print(_run(client, _status_cmd()))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
