#!/usr/bin/env python3
"""Sanity-check artifacts/connectome_mb_v1.npz loads and has expected roles."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


def main() -> int:
    npz = np.load(ART / "connectome_mb_v1.npz", allow_pickle=True)
    manifest = json.loads((ART / "manifest.json").read_text())
    roles = npz["roles"]
    counts = {r: int((roles == r).sum()) for r in ["PN", "KC", "MBON", "DAN"]}
    print("roles", counts)
    print("pn->kc", len(npz["pn_kc_w"]), "kc->mbon", len(npz["kc_mbon_w"]))
    assert counts["KC"] > 1000
    assert counts["PN"] > 100
    assert counts["MBON"] > 20
    assert counts["DAN"] > 50
    assert len(npz["pn_kc_w"]) > 1000
    assert len(npz["kc_mbon_w"]) > 1000
    assert manifest["snapshot"] == "FAFB-783"
    print("OK connectome_mb_v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
