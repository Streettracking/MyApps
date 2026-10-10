#!/usr/bin/env python3
"""Download the FlyWire brain shell and MB skeletons for the 3D window.

No CAVE token. Two public sources:

* neuropil surface: ``FLYWIRE.ply`` from navis-flybrains (GitHub)
* skeletons: precomputed 783 release at
  ``https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/<root_id>``

Our ``connectome_mb_v1`` root ids are that 783 snapshot, so the files match.
L2 skeletons for a newer materialization need a FlyWire CAVE token; this
script does not use that path. Neurons missing from the 783 cache are skipped.
White fibres inside the optic lobes are not part of this mushroom-body root
set, so they are not downloaded. The lobes of the shell are the eyes.

Writes ``artifacts/fly_brain_view.npz``. The 3D window uses it when present
and otherwise keeps the soma point cloud.

    python tools/fetch_flywire_view.py
    python tools/fetch_flywire_view.py --limit 40
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.mb_flywire import (  # noqa: E402
    KIND_KC_AB,
    KIND_KC_APB,
    KIND_KC_G,
    KIND_KC_OTHER,
    KIND_MBON,
    KIND_PAM,
    KIND_PPL1,
    VIEW_NPZ,
)
from sim.npz_compat import open_npz  # noqa: E402

MESH_URL = (
    "https://raw.githubusercontent.com/navis-org/navis-flybrains/"
    "main/flybrains/meshes/FLYWIRE.ply"
)
SKEL_URL = "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/{}"
CONNECTOME = ROOT / "artifacts" / "connectome_mb_v1_np1.npz"


def _kind(role: str, cell_type: str, appetitive: bool) -> int:
    if role == "MBON":
        return KIND_MBON
    if role == "DAN":
        return KIND_PAM if appetitive else KIND_PPL1
    if cell_type.startswith("KCg"):
        return KIND_KC_G
    if cell_type.startswith("KCab"):
        return KIND_KC_AB
    if cell_type.startswith("KCa"):
        return KIND_KC_APB
    return KIND_KC_OTHER


def _load_ply(path: Path):
    raw = path.read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    text = head.decode("ascii", "replace")
    nv = nf = None
    for line in text.splitlines():
        if line.startswith("element vertex"):
            nv = int(line.split()[-1])
        elif line.startswith("element face"):
            nf = int(line.split()[-1])
    verts = np.frombuffer(body, dtype="<f4", count=nv * 3, offset=0).reshape(nv, 3).copy()
    rec = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    faces = np.ndarray(nf, dtype=rec, buffer=body[nv * 12: nv * 12 + nf * 13])
    if int(faces["n"].min()) != 3 or int(faces["n"].max()) != 3:
        raise RuntimeError("ожидались треугольники в FLYWIRE.ply")
    tri = np.stack([faces["a"], faces["b"], faces["c"]], axis=1).astype(np.int32)
    return verts, tri


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "fly-mb-view/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        dest.write_bytes(response.read())


def _parse_skeleton(raw: bytes):
    if len(raw) < 16:
        return None
    nv = int(np.frombuffer(raw, dtype="<u4", count=1, offset=0)[0])
    ne = int(np.frombuffer(raw, dtype="<u4", count=1, offset=4)[0])
    if nv < 2 or ne < 1 or nv > 500000 or ne > 1000000:
        return None
    need = 8 + nv * 12 + ne * 8
    if need > len(raw):
        return None
    xyz = np.frombuffer(raw, dtype="<f4", count=nv * 3, offset=8).reshape(nv, 3).copy()
    edges = np.frombuffer(raw, dtype="<u4", count=ne * 2, offset=8 + nv * 12).reshape(ne, 2).copy()
    if not np.isfinite(xyz).all():
        return None
    return xyz, edges


def _simplify(xyz: np.ndarray, edges: np.ndarray, max_points: int = 36):
    """Keep branch points and shorten the chains between them."""
    n = int(len(xyz))
    if n <= max_points:
        keep = (edges[:, 0] < n) & (edges[:, 1] < n)
        return xyz.astype(np.float32), edges[keep].astype(np.int32)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        a = int(a)
        b = int(b)
        if a >= n or b >= n or a == b:
            continue
        adj[a].append(b)
        adj[b].append(a)
    deg = [len(item) for item in adj]
    keys = [i for i, d in enumerate(deg) if d != 2]
    if not keys:
        keys = [int(np.argmax(np.linalg.norm(xyz - xyz.mean(0), axis=1)))]
    used = set()
    chains = []

    def ek(a, b):
        return (a, b) if a < b else (b, a)

    for start in keys:
        for nb in adj[start]:
            mark = ek(start, nb)
            if mark in used:
                continue
            chain = [start, nb]
            used.add(mark)
            prev, cur = start, nb
            guard = 0
            while deg[cur] == 2 and guard < n:
                guard += 1
                nxts = [item for item in adj[cur] if item != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                mark = ek(cur, nxt)
                if mark in used:
                    break
                used.add(mark)
                chain.append(nxt)
                prev, cur = cur, nxt
            chains.append(chain)
    pts = []
    eds = []
    # Spend the budget on longer chains first.
    lengths = []
    for chain in chains:
        coords = xyz[np.asarray(chain, dtype=np.int64)]
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        lengths.append(float(seg.sum()) if len(seg) else 0.0)
    order = np.argsort(lengths)[::-1]
    remaining = max_points
    for index in order:
        chain = chains[int(index)]
        if remaining < 2:
            break
        coords = xyz[np.asarray(chain, dtype=np.int64)]
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total = float(seg.sum()) if len(seg) else 0.0
        budget = 2 if total < 1.0 else min(remaining, max(2, int(round(total / 18000.0)) + 1))
        cum = np.concatenate([[0.0], np.cumsum(seg)]) if len(seg) else np.zeros(1)
        samples = np.linspace(0.0, max(total, 0.0), budget)
        out = []
        base = len(pts)
        for sample in samples:
            if len(seg) == 0:
                pts.append(coords[0])
                out.append(base)
                continue
            i = int(np.searchsorted(cum, sample, side="right") - 1)
            i = min(max(i, 0), len(seg) - 1)
            span = float(seg[i]) if float(seg[i]) > 1e-6 else 1.0
            t = min(max((sample - float(cum[i])) / span, 0.0), 1.0)
            pts.append(coords[i] * (1.0 - t) + coords[i + 1] * t)
            out.append(base + len(out))
        for a, b in zip(out, out[1:]):
            eds.append((a, b))
        remaining -= len(out)
    if len(pts) < 2 or not eds:
        return xyz[:2].astype(np.float32), np.zeros((0, 2), np.int32)
    return np.asarray(pts, dtype=np.float32), np.asarray(eds, dtype=np.int32)


def _decimate(verts, faces, optic, cell: float):
    quant = np.floor(verts / cell).astype(np.int64)
    _uniq, inv = np.unique(quant, axis=0, return_inverse=True)
    bins = int(inv.max()) + 1
    acc = np.zeros((bins, 3), dtype=np.float64)
    count = np.bincount(inv).astype(np.float64)
    np.add.at(acc, inv, verts)
    new_v = (acc / count[:, None]).astype(np.float32)
    votes = np.zeros((bins, 3), dtype=np.int32)
    for label in (0, 1, 2):
        np.add.at(votes[:, label], inv, (optic == label).astype(np.int32))
    new_opt = np.argmax(votes, axis=1).astype(np.uint8)
    mapped = inv[faces]
    ok = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) & (mapped[:, 0] != mapped[:, 2])
    mapped = np.sort(mapped[ok], axis=1)
    mapped = np.unique(mapped, axis=0).astype(np.int32)
    return new_v, mapped, new_opt


def _neurons(limit: int):
    path = CONNECTOME if CONNECTOME.is_file() else ROOT / "artifacts" / "connectome_mb_v1.npz"
    z = open_npz(path)
    try:
        roots = z["root_ids"].astype(np.int64)
        roles = np.array([str(item) for item in z["roles"]])
        types = np.array([str(item) for item in z["cell_types"]])
        sides = np.array([str(item) for item in z["sides"]])
        kc_idx = z["kc_idx"].astype(np.int32)
        mbon_idx = z["mbon_idx"].astype(np.int32)
        app = z["dan_appetitive_idx"].astype(np.int32) if "dan_appetitive_idx" in z.files else np.zeros(0, np.int32)
        av = z["dan_aversive_idx"].astype(np.int32) if "dan_aversive_idx" in z.files else np.zeros(0, np.int32)
    finally:
        z.close()
    local = {int(g): i for i, g in enumerate(kc_idx)}
    appetitive = set(int(i) for i in app)
    rows = []
    for index in list(kc_idx) + list(mbon_idx) + list(app) + list(av):
        index = int(index)
        role = roles[index]
        if role not in ("KC", "MBON", "DAN"):
            continue
        rows.append((
            int(roots[index]),
            _kind(role, types[index], index in appetitive),
            1 if sides[index] == "left" else 0,
            int(local.get(index, -1)),
        ))
    if limit > 0:
        rows = rows[:limit]
    return rows


def _fetch_one(row):
    root_id, kind, side, kc_local = row
    url = SKEL_URL.format(root_id)
    raw = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fly-mb-view/1.0"})
            with urllib.request.urlopen(req, timeout=40) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == 2:
                return None
            time.sleep(0.4 * (attempt + 1))
        except Exception:
            if attempt == 2:
                return None
            time.sleep(0.4 * (attempt + 1))
    parsed = _parse_skeleton(raw)
    if parsed is None:
        return None
    xyz, edges = _simplify(*parsed)
    if len(edges) == 0:
        return None
    return root_id, kind, side, kc_local, xyz, edges


def build(limit: int = 0, workers: int = 8) -> Path:
    ply = ROOT / "artifacts" / "_flywire_neuropil.ply"
    if not ply.is_file() or ply.stat().st_size < 10000:
        print("mesh", MESH_URL)
        _download(MESH_URL, ply)
    raw_verts, raw_faces = _load_ply(ply)
    span = float(raw_verts[:, 0].max() - raw_verts[:, 0].min())
    raw_optic = np.zeros(len(raw_verts), dtype=np.uint8)
    raw_optic[raw_verts[:, 0] < raw_verts[:, 0].min() + 0.30 * span] = 1
    raw_optic[raw_verts[:, 0] > raw_verts[:, 0].max() - 0.30 * span] = 2
    # Same centre for the shell and the skeletons. Smaller FlyWire x is
    # anatomical left, smaller y is dorsal, larger z is anterior.
    center = raw_verts.mean(axis=0)
    scale = np.float32(2.7 / max(span, 1.0))

    def _show(block):
        out = np.empty(block.shape, dtype=np.float32)
        out[:, 0] = (block[:, 0] - center[0]) * scale
        out[:, 1] = (center[1] - block[:, 1]) * scale
        out[:, 2] = (block[:, 2] - center[2]) * scale
        return out

    fine_v, fine_f, fine_o = _decimate(raw_verts, raw_faces, raw_optic, span / 52.0)
    coarse_v, coarse_f, coarse_o = _decimate(raw_verts, raw_faces, raw_optic, span / 20.0)
    display = _show(fine_v)
    display_lod = _show(coarse_v)
    print(
        "shell", display.shape, fine_f.shape,
        "lod", display_lod.shape, coarse_f.shape,
        "optic", {int(k): int((fine_o == k).sum()) for k in (0, 1, 2)},
    )

    rows = _neurons(limit)
    print("skeletons", len(rows))
    points = []
    edges = []
    kinds = []
    sides = []
    kc_locals = []
    root_ids = []
    ptr = [0]
    done = 0
    missed_ids = []
    fetched = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_fetch_one, row): index for index, row in enumerate(rows)}
        for future in as_completed(futures):
            done += 1
            index = futures[future]
            item = future.result()
            fetched[index] = item
            if item is None:
                missed_ids.append(rows[index][0])
            if done % 250 == 0 or done == len(rows):
                print("  %s/%s missed %s" % (done, len(rows), len(missed_ids)), flush=True)
    base = 0
    for item in fetched:
        if item is None:
            continue
        root_id, kind, side, kc_local, xyz, ed = item
        show = _show(xyz)
        points.append(show.astype(np.float32))
        edges.append(ed.astype(np.int32) + base)
        base += int(len(show))
        kinds.append(kind)
        sides.append(side)
        kc_locals.append(kc_local)
        root_ids.append(root_id)
        ptr.append(ptr[-1] + len(ed))
    if len(points) < 50:
        raise RuntimeError("слишком мало скелетов (%s). Сеть или идентификаторы 783." % len(points))
    out = VIEW_NPZ
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vertices=display.astype(np.float32),
        faces=fine_f.astype(np.int32),
        optic=fine_o.astype(np.uint8),
        vertices_lod=display_lod.astype(np.float32),
        faces_lod=coarse_f.astype(np.int32),
        optic_lod=coarse_o.astype(np.uint8),
        sk_points=np.concatenate(points, axis=0),
        sk_edges=np.concatenate(edges, axis=0),
        neuron_ptr=np.asarray(ptr, dtype=np.int32),
        neuron_kind=np.asarray(kinds, dtype=np.uint8),
        neuron_side=np.asarray(sides, dtype=np.uint8),
        neuron_kc=np.asarray(kc_locals, dtype=np.int32),
        root_id=np.asarray(root_ids, dtype=np.int64),
        fit_dist=np.float32(4.6),
    )
    if missed_ids:
        sample = ", ".join(str(item) for item in missed_ids[:8])
        print("пропущены (нет в кэше 783), например:", sample)
    print("wrote", out, "bytes", out.stat().st_size, "neurons", len(kinds), "missed", len(missed_ids))
    return out


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download FlyWire shell and MB skeletons")
    parser.add_argument("--limit", type=int, default=0, help="only the first N neurons, for a trial")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    build(limit=args.limit, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
