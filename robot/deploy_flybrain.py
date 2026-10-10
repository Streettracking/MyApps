#!/usr/bin/env python3
"""Copy the onboard fly brain to one Go2 and start it by hand.

    GO2_SSH_PASS=... python robot/deploy_flybrain.py start
    GO2_SSH_PASS=... python robot/deploy_flybrain.py start --state logs/mb_train_state.npz
    GO2_SSH_PASS=... python robot/deploy_flybrain.py stop
    GO2_SSH_PASS=... python robot/deploy_flybrain.py status

``start`` keeps the state file already on the dog. ``--clean`` deletes it.
``--state PATH`` replaces it. The connectome copied to the dog is
``connectome_mb_v1_np1.npz`` (no pickled object arrays).

The password is only ``GO2_SSH_PASS``. This script does not install
packages, does not write a systemd unit, and does not touch the preview
server, its unit, or ``/tmp/robot_preview_server.py``.

Remote files stay under ``/root/flybrain/``. Start detaches with
``setsid`` so the SSH channel closes, and ``main.py`` writes its own
pid to ``flybrain.pid``. Stop kills that pid and any leftover
``python3 /root/flybrain/main.py``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_SAFE_ARG = re.compile(r"^[A-Za-z0-9_./:+-]+$")

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
    "hemifield.py",
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


def _run(client, cmd: str, timeout: float = 12.0) -> str:
    """Run one remote command and return. Never block on a live child.

    A background process that still holds the SSH pipe used to keep
    ``stdout.read`` open until the brain exited. The caller passes a
    timeout, and the channel is closed when that timeout lands.
    """
    import time

    _reject(cmd)
    stdin, stdout, stderr = client.exec_command(cmd)
    try:
        stdin.close()
    except Exception:
        pass
    channel = stdout.channel
    deadline = time.monotonic() + float(timeout)
    out_chunks = []
    err_chunks = []
    while time.monotonic() < deadline:
        progressed = False
        if channel.recv_ready():
            out_chunks.append(channel.recv(65536))
            progressed = True
        if channel.recv_stderr_ready():
            err_chunks.append(channel.recv_stderr(65536))
            progressed = True
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        if not progressed:
            time.sleep(0.05)
    while channel.recv_ready():
        out_chunks.append(channel.recv(65536))
    while channel.recv_stderr_ready():
        err_chunks.append(channel.recv_stderr(65536))
    try:
        channel.close()
    except Exception:
        pass
    out = b"".join(out_chunks).decode("utf-8", errors="replace")
    err = b"".join(err_chunks).decode("utf-8", errors="replace")
    return (out + err).strip()


def _alive_fn(root: str) -> str:
    """Shell function: true only when an argv is exactly ``root/main.py``.

    ``pgrep -f`` also sees the shell that contains that path as text.
    Matching one argv skips that shell and matches only the interpreter.
    """
    main = root + "/main.py"
    return (
        "_fly_alive() { "
        "for p in $(pgrep -f %s || true); do "
        "if [ -r /proc/$p/cmdline ] && tr \"\\0\" \"\\n\" < /proc/$p/cmdline | grep -qx %s; then "
        "return 0; "
        "fi; "
        "done; "
        "return 1; "
        "}; "
    ) % (main, main)


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


def _remote_args(extra) -> str:
    """Shell-safe tokens appended to ``python3 main.py``. Rejects anything else."""
    parts = []
    for token in extra or ():
        text = str(token)
        if not text or _SAFE_ARG.match(text) is None:
            raise SystemExit("unsafe argument for main.py: %s" % text)
        parts.append(text)
    if not parts:
        return ""
    return " " + " ".join(parts)


def _start_cmd(root: str = REMOTE, extra=None) -> str:
    """Detach python. ``main.py`` writes the pid; ``$!`` is not the shell.

    ``setsid`` puts python in its own session. ``< /dev/null`` and the
    log redirect drop the SSH pipe, so this command exits at once.
    """
    main = root + "/main.py"
    log = root + "/flybrain.log"
    return (
        "bash -lc '"
        + _alive_fn(root)
        + "cd %s || exit 1; "
        "export PYTHONPATH=%s:/unitree/module/pet_go:/root/go2_flask_api; "
        "if _fly_alive; then echo already; exit 0; fi; "
        "nohup setsid python3 %s%s >> %s 2>&1 < /dev/null & "
        "disown || true; "
        "echo started; "
        "exit 0'"
    ) % (root, root, main, _remote_args(extra), log)


def _stop_cmd(root: str = REMOTE) -> str:
    """Kill the pid file entry, then every process whose argv is main.py."""
    pidfile = root + "/flybrain.pid"
    return (
        "bash -lc '"
        + _alive_fn(root)
        + "pid=$(cat %s 2>/dev/null || true); "
        "if [ -n \"$pid\" ] && [ -r /proc/$pid/cmdline ] && "
        "tr \"\\0\" \" \" < /proc/$pid/cmdline | grep -q flybrain; then "
        "kill $pid 2>/dev/null || true; "
        "fi; "
        "for p in $(pgrep -f %s || true); do "
        "if [ -r /proc/$p/cmdline ] && tr \"\\0\" \"\\n\" < /proc/$p/cmdline | grep -qx %s; then "
        "kill $p 2>/dev/null || true; "
        "fi; "
        "done; "
        "sleep 0.6; "
        "if _fly_alive; then "
        "for p in $(pgrep -f %s || true); do "
        "if [ -r /proc/$p/cmdline ] && tr \"\\0\" \"\\n\" < /proc/$p/cmdline | grep -qx %s; then "
        "kill -9 $p 2>/dev/null || true; "
        "fi; "
        "done; "
        "sleep 0.2; "
        "fi; "
        "if _fly_alive; then echo still-running; else echo stopped; fi'"
    ) % (pidfile, root + "/main.py", root + "/main.py", root + "/main.py", root + "/main.py")


def _status_cmd(root: str = REMOTE) -> str:
    pidfile = root + "/flybrain.pid"
    log = root + "/flybrain.log"
    main = root + "/main.py"
    return (
        "bash -lc '"
        "pid=$(cat %s 2>/dev/null || true); "
        "echo pid=$pid; "
        "if [ -n \"$pid\" ] && [ -r /proc/$pid/cmdline ]; then "
        "tr \"\\0\" \" \" < /proc/$pid/cmdline; echo; "
        "else echo not-running; fi; "
        "for p in $(pgrep -f %s || true); do "
        "if [ -r /proc/$p/cmdline ] && tr \"\\0\" \"\\n\" < /proc/$p/cmdline | grep -qx %s; then "
        "echo main=$p; "
        "fi; "
        "done; "
        "tail -n 20 %s 2>/dev/null || true'"
    ) % (pidfile, main, main, log)


def state_plan(state_path: str, clean: bool) -> str:
    """What ``start`` does with ``/root/flybrain/state/mb_train_state.npz``.

    ``upload`` replaces it, ``wipe`` deletes it, ``keep`` leaves it alone.
    """
    if state_path:
        return "upload"
    if clean:
        return "wipe"
    return "keep"


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Copy the onboard fly brain and start it by hand")
    parser.add_argument("action", nargs="?", default="start", choices=("start", "stop", "status"))
    parser.add_argument(
        "--state",
        default="",
        help="Replace the dog's mb_train_state.npz with this file. Omit to keep the file already there.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the dog's mb_train_state.npz. Without this, start keeps it.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Forward --overlap VALUE to main.py (0..0.5). Other unknown flags are forwarded on start too.",
    )
    args, unknown = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    extra = []
    if args.overlap is not None:
        if args.overlap < 0.0 or args.overlap > 0.5:
            print("overlap must be between 0 and 0.5", file=sys.stderr)
            return 2
        extra.extend(["--overlap", "%g" % args.overlap])
    extra.extend(unknown)
    action = args.action
    if action == "start":
        _remote_args(extra)
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
            plan = state_plan(state_path, bool(args.clean))
            if plan == "upload":
                print("state %s" % state_path)
            elif plan == "wipe":
                print(_run(client, "rm -f /root/flybrain/state/mb_train_state.npz"))
                print("clean brain: --clean removed the state on the dog")
            else:
                print("keep state: /root/flybrain/state/mb_train_state.npz")
            print(_run(client, "rm -f /root/flybrain/artifacts/connectome_mb_v1.npz"))
            print(_run(client, _start_cmd(extra=extra), timeout=8.0))
        elif action == "stop":
            print(_run(client, _stop_cmd()))
        print(_run(client, _status_cmd()))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
