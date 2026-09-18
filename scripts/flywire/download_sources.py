#!/usr/bin/env python3
"""Download FlyWire FAFB-783 sources needed to rebuild connectome_mb_v1.

Codex API requires a token; this script uses public Zenodo + GitHub mirrors instead.
Full proofread_connections feather is ~813MB and is NOT committed to git.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "flywire" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOURCES = [
    {
        "path": RAW / "Supplemental_file1_neuron_annotations.tsv",
        "url": (
            "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
            "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
        ),
        "optional": False,
    },
    {
        "path": RAW / "proofread_root_ids_783.npy",
        "url": "https://zenodo.org/api/records/10676866/files/proofread_root_ids_783.npy/content",
        "optional": False,
    },
    {
        "path": RAW / "proofread_connections_783.feather",
        "url": "https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content",
        "optional": False,
        "expected_sha256_file": RAW / "PROOFREAD_CONNECTIONS_SHA256.txt",
    },
]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: pathlib.Path) -> None:
    print(f"Downloading {url}\n  -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"  size={dest.stat().st_size} sha256={sha256_file(dest)}")


def main() -> int:
    skip_large = "--skip-large" in sys.argv
    for src in SOURCES:
        path: pathlib.Path = src["path"]
        if path.name.endswith(".feather") and skip_large:
            print(f"Skipping large file {path.name}")
            continue
        if path.exists() and path.stat().st_size > 0:
            print(f"Exists: {path} ({path.stat().st_size} bytes)")
            continue
        download(src["url"], path)
        sha_file = src.get("expected_sha256_file")
        if sha_file and sha_file.exists():
            expected = sha_file.read_text().split()[0]
            got = sha256_file(path)
            if expected != got:
                print(f"WARNING: sha256 mismatch for {path.name}: {got} != {expected}")
            else:
                print("sha256 OK")
    print("Done. Next: python scripts/flywire/build_mb_subgraph.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
