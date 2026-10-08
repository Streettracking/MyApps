"""Headless comparison: fixed detector, raw features, recognition layer, blind peers.

Usage:
    python -m sim.compare_percept --seconds 60 --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.simulator import SimConfig, Simulator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "artifacts" / "connectome_mb_v1.npz"

CONDITIONS = (
    ("fixed", "fixed", True),
    ("raw", "raw", True),
    ("recognize", "recognize", True),
    ("blind", "fixed", False),
    ("raw_blind", "raw", False),
)


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def run_one(percept: str, sense: bool, seed: int, seconds: float, npz: Path) -> dict:
    cfg = SimConfig(
        npz_path=npz,
        duration_s=seconds,
        n_agents=3,
        sense_conspecifics=sense,
        percept=percept,
        seed=seed,
        log_dir=None,
        explore_eps=0.25,
    )
    summary = Simulator(cfg).run()
    dog_n = dist_n = dog_app = dist_app = 0
    r_steps = r_dog = r_dist = 0
    plastics, drifts, diffs, dog_l2, dist_l2 = [], [], [], [], []
    for ag in summary["agents"].values():
        ctx = ag["context"]
        dog_n += ctx["dog_only"]
        dist_n += ctx["dist_only"]
        dog_app += ctx["dog_only_approach"]
        dist_app += ctx["dist_only_approach"]
        r_steps += ctx["r_steps"]
        r_dog += ctx["r_and_dog"]
        r_dist += ctx["r_and_dist"]
        plastics.append(ag["plastic_updates"])
        drifts.append(ag["weight_drift"])
        diffs.append(ag["learned_shift"]["diff_l2"])
        dog_l2.append(ag["learned_shift"]["learned_dog_l2"])
        dist_l2.append(ag["learned_shift"]["learned_dist_l2"])
    row = {
        "seed": seed,
        "mean_PI": summary["mean_PI"],
        "plastic_updates": float(np.mean(plastics)),
        "weight_drift": float(np.mean(drifts)),
        "diff_l2": float(np.mean(diffs)),
        "learned_dog_l2": float(np.mean(dog_l2)),
        "learned_dist_l2": float(np.mean(dist_l2)),
        "approach_dog": _rate(dog_app, dog_n),
        "approach_dist": _rate(dist_app, dist_n),
        "dog_only_steps": dog_n,
        "dist_only_steps": dist_n,
        "frac_reward_with_dog": _rate(r_dog, r_steps),
        "frac_reward_with_dist": _rate(r_dist, r_steps),
    }
    final = (summary.get("recognition") or {}).get("final") or {}
    curve = (summary.get("recognition") or {}).get("curve")
    for key in (
        "sep_moving",
        "sep_static",
        "invariance",
        "purity",
        "like_dog",
        "like_dist",
        "like_static",
        "like_dog_still",
        "like_dist_fast",
        "n_self",
    ):
        row[key] = final.get(key)
    row["recognition_curve"] = curve
    return row


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def aggregate(rows: list[dict]) -> dict:
    keys = (
        "mean_PI",
        "plastic_updates",
        "weight_drift",
        "diff_l2",
        "learned_dog_l2",
        "learned_dist_l2",
        "approach_dog",
        "approach_dist",
        "frac_reward_with_dog",
        "frac_reward_with_dist",
        "sep_moving",
        "sep_static",
        "invariance",
        "purity",
        "like_dog",
        "like_dist",
        "like_static",
        "like_dog_still",
        "like_dist_fast",
        "n_self",
    )
    return {k: _mean(rows, k) for k in keys}


def mean_curve(rows: list[dict]) -> list[dict] | None:
    curves = [r["recognition_curve"] for r in rows if r.get("recognition_curve")]
    if not curves:
        return None
    n = min(len(c) for c in curves)
    keys = [k for k in curves[0][0] if k != "t"]
    out = []
    for i in range(n):
        point = {"t": float(curves[0][i]["t"])}
        for key in keys:
            point[key] = float(np.mean([c[i][key] for c in curves]))
        out.append(point)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "percept_compare.json")
    args = p.parse_args()
    report = {"seconds": args.seconds, "seeds": list(range(args.seeds)), "conditions": {}}
    for name, percept, sense in CONDITIONS:
        rows = []
        for seed in range(args.seeds):
            row = run_one(percept, sense, seed, args.seconds, args.npz)
            rows.append(row)
            print(f"{name} seed={seed} PI={row['mean_PI']:+.3f} diff_l2={row['diff_l2']:.3f}")
        block = {"runs": rows, "mean": aggregate(rows)}
        curve = mean_curve(rows)
        if curve is not None:
            block["curve"] = curve
        report["conditions"][name] = block
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in report["conditions"].items()}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
