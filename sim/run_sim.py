"""CLI for FlyWire MB × Go2 arena simulator (Windows / Linux / macOS)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "artifacts" / "connectome_mb_v1.npz"


def main() -> int:
    p = argparse.ArgumentParser(description="FlyWire MB × Go2 arena simulator (Windows OK)")
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--seconds", type=float, default=120.0, help="0 in GUI = run until Esc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--blind-peers", action="store_true")
    p.add_argument("--move-zones", action="store_true", help="Projected zones drift over time")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--gui", action="store_true", help="force GUI (default when not --headless)")
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--eta", type=float, default=0.05)
    args = p.parse_args()

    if not args.npz.exists():
        raise SystemExit(f"Missing connectome artifact: {args.npz}")

    use_gui = not args.headless
    if use_gui:
        from sim.viz_pygame import run_pygame

        seconds = None if args.seconds <= 0 else args.seconds
        summary = run_pygame(
            npz=args.npz,
            duration_s=seconds or 10**9,
            n_agents=args.agents,
            move_zones=args.move_zones,
            sense=not args.blind_peers,
        )
        # if unlimited, still return last summary
    else:
        from sim.simulator import SimConfig, Simulator

        log_dir = args.log_dir or (ROOT / "logs" / "sim_last")
        cfg = SimConfig(
            npz_path=args.npz,
            duration_s=args.seconds,
            n_agents=args.agents,
            sense_conspecifics=not args.blind_peers,
            move_zones=args.move_zones,
            eta=args.eta,
            seed=args.seed,
            log_dir=log_dir,
        )
        summary = Simulator(cfg).run()

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
