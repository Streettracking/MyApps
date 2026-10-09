"""PyInstaller entry for the recognition trainer.

Defaults match the lab: robot preview at 192.168.35.213:8088, drive commands
to the local go2_wr_server at 127.0.0.1:5451. ``--sim`` opens the zone-free
simulator instead, so the exe can be tried with no robot.

Default learning is the mushroom body. T is a PAM treat, X is a PPL1 punish.
Familiarity is ``--dan familiarity``. ``A`` lets the mushroom body search
and approach. ``M`` is takeover. ``--onboard IP`` watches the brain on the dog.
The Hebbian layer is ``--learner hebb``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Static imports so PyInstaller traces the trainer and the connectome runtime.
import sim.frame_sense  # noqa: F401
import sim.learn_flash  # noqa: F401
import sim.npz_compat  # noqa: F401
import sim.onboard_link  # noqa: F401
import sim.pilot  # noqa: F401
import sim.lidar_fresh  # noqa: F401
import sim.map_marks  # noqa: F401
import sim.go2_udp  # noqa: F401
import sim.mb_confidence  # noqa: F401
import sim.mb_runtime  # noqa: F401
import sim.mb_train  # noqa: F401
import sim.raw_sense  # noqa: F401
import sim.recognize  # noqa: F401
import sim.recognize_train as recognize_train
import sim.recognize_train_live as recognize_train_live
import sim.train_monitor  # noqa: F401
import sim.world  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    # Audio is left on the system driver so the optional beep can play.
    # A missing device is reported when B is pressed; the window still opens.
    p = argparse.ArgumentParser(description="Go2 conspecific recognition trainer")
    p.add_argument("--sim", action="store_true", help="Open the simulator training mode instead of the robot")
    p.add_argument("--robot-ip", default=recognize_train_live.DEFAULT_IP)
    p.add_argument("--preview-port", type=int, default=recognize_train_live.DEFAULT_PREVIEW)
    p.add_argument("--udp-host", default=recognize_train_live.DEFAULT_UDP_HOST)
    p.add_argument("--udp-port", type=int, default=recognize_train_live.DEFAULT_UDP_PORT)
    p.add_argument("--state", type=Path, default=None)
    p.add_argument("--load", action="store_true")
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=5)
    p.add_argument("--screenshot", type=Path, default=None)
    p.add_argument("--reset-lidar", action="store_true")
    p.add_argument("--learner", choices=("mb", "hebb"), default="mb")
    p.add_argument("--dan", choices=("teacher", "familiarity"), default="teacher")
    p.add_argument("--eta", type=float, default=0.2)
    p.add_argument("--auto-teach", action="store_true")
    p.add_argument("--punish", action="store_true", help="With --auto-teach in the sim, PPL1 on distractor-only views")
    p.add_argument(
        "--lidar-refresh",
        type=float,
        default=recognize_train_live.DEFAULT_LIDAR_REFRESH,
        help="Seconds between fresh lidar windows. 0 keeps the accumulating map.",
    )
    p.add_argument("--onboard", default="", help="Robot IP. Brain stays on the dog. This window does not send UDP.")
    p.add_argument("--onboard-port", type=int, default=8090)
    p.add_argument("--return-auto", type=float, default=0.0)
    p.add_argument("--steer", choices=("bilateral", "sectors"), default="bilateral")
    args = p.parse_args(argv)
    if args.sim and args.onboard:
        print("--sim and --onboard are different windows. Pick one.", file=sys.stderr)
        return 2
    if args.state is None:
        name = "mb_train_state.npz" if args.learner == "mb" else "recognizer_state.json"
        args.state = Path("logs") / name
    if args.sim:
        forwarded = [
            "--agents",
            str(args.agents),
            "--seconds",
            str(args.seconds),
            "--seed",
            str(args.seed),
            "--state",
            str(args.state),
            "--learner",
            args.learner,
            "--dan",
            args.dan,
            "--eta",
            str(args.eta),
        ]
        if args.headless:
            forwarded.append("--headless")
        if args.load or args.state.is_file():
            forwarded.append("--load")
        if args.screenshot:
            forwarded.extend(["--screenshot", str(args.screenshot)])
        if args.auto_teach:
            forwarded.append("--auto-teach")
        if args.punish:
            forwarded.append("--punish")
        forwarded.extend(["--lidar-refresh", str(args.lidar_refresh)])
        forwarded.extend(["--return-auto", str(args.return_auto)])
        forwarded.extend(["--steer", args.steer])
        return recognize_train.main(forwarded)
    forwarded = [
        "--robot-ip",
        args.robot_ip,
        "--preview-port",
        str(args.preview_port),
        "--udp-host",
        args.udp_host,
        "--udp-port",
        str(args.udp_port),
        "--state",
        str(args.state),
        "--max-frames",
        str(args.max_frames),
        "--seconds",
        str(args.seconds),
        "--learner",
        args.learner,
        "--dan",
        args.dan,
        "--eta",
        str(args.eta),
        "--seed",
        str(args.seed),
    ]
    if args.headless:
        forwarded.append("--headless")
    if args.load or args.state.is_file():
        forwarded.append("--load")
    if args.reset_lidar:
        forwarded.append("--reset-lidar")
    forwarded.extend(["--lidar-refresh", str(args.lidar_refresh)])
    forwarded.extend(["--return-auto", str(args.return_auto)])
    forwarded.extend(["--steer", args.steer])
    if args.onboard:
        forwarded.extend(["--onboard", args.onboard, "--onboard-port", str(args.onboard_port)])
    return recognize_train_live.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
