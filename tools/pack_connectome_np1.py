#!/usr/bin/env python3
"""Write artifacts/connectome_mb_v1_np1.npz without pickled object arrays.

Run this on the laptop after the connectome file changes. Deploy copies
the result. Numpy 1.24 on the robot opens it with allow_pickle left off.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.npz_compat import open_npz, write_pickle_free  # noqa: E402


def main() -> int:
    src = ROOT / "artifacts" / "connectome_mb_v1.npz"
    dest = ROOT / "artifacts" / "connectome_mb_v1_np1.npz"
    write_pickle_free(src, dest)
    with open_npz(dest) as loaded:
        objects = [key for key in loaded.files if loaded[key].dtype == object]
    if objects:
        print("still object arrays: %s" % ", ".join(objects), file=sys.stderr)
        return 1
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
