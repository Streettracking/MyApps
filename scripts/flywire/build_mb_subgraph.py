#!/usr/bin/env python3
"""Build artifacts/connectome_mb_v1.* from local FlyWire FAFB-783 sources."""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.feather as feather

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "flywire" / "raw"
ART = ROOT / "artifacts"
SYN_THRESH = 5


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    ann_tsv = RAW / "Supplemental_file1_neuron_annotations.tsv"
    ann_gz = RAW / "Supplemental_file1_neuron_annotations.tsv.gz"
    if not ann_tsv.exists() and ann_gz.exists():
        with gzip.open(ann_gz, "rb") as f_in, ann_tsv.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    conn_path = RAW / "proofread_connections_783.feather"
    if not ann_tsv.exists() or not conn_path.exists():
        raise SystemExit(
            "Missing sources. Run: python scripts/flywire/download_sources.py"
        )

    ann = pd.read_csv(ann_tsv, sep="\t", low_memory=False)
    ann["root_id"] = ann["root_id"].astype(np.int64)
    cc = ann["cell_class"].astype(str)
    csc = ann["cell_sub_class"].astype(str)
    role = np.array([""] * len(ann), dtype=object)
    role[cc.str.lower().eq("kenyon_cell")] = "KC"
    role[cc.str.upper().eq("MBON")] = "MBON"
    role[cc.str.upper().eq("DAN")] = "DAN"
    role[cc.str.lower().eq("alpn")] = "PN"
    role[
        (role == "")
        & cc.str.lower().eq("olfactory")
        & csc.str.contains("glomerular|uniglomerular|multiglomerular", case=False, na=False)
    ] = "PN"
    ann = ann.assign(role=role)
    mb = ann[ann["role"] != ""].copy()
    keep_ids = set(mb["root_id"].tolist())

    conn = feather.read_table(conn_path).to_pandas()
    conn = conn[
        conn["pre_pt_root_id"].isin(keep_ids) & conn["post_pt_root_id"].isin(keep_ids)
    ].copy()
    conn_f = conn[conn["syn_count"] >= SYN_THRESH].copy()
    deg_ids = set(conn_f["pre_pt_root_id"]).union(set(conn_f["post_pt_root_id"]))
    core = set(mb.loc[mb.role.isin(["MBON", "DAN"]), "root_id"])
    mb_conn = mb[mb["root_id"].isin(deg_ids.union(core))].copy()
    mb_conn = mb_conn.sort_values(["role", "cell_type", "root_id"]).reset_index(drop=True)
    id_to_idx = {int(r): i for i, r in enumerate(mb_conn["root_id"])}

    conn_f = conn_f[
        conn_f["pre_pt_root_id"].isin(id_to_idx) & conn_f["post_pt_root_id"].isin(id_to_idx)
    ].copy()
    pre = conn_f["pre_pt_root_id"].map(id_to_idx).to_numpy(np.int32)
    post = conn_f["post_pt_root_id"].map(id_to_idx).to_numpy(np.int32)
    w = conn_f["syn_count"].to_numpy(np.float32)

    roles_arr = mb_conn["role"].to_numpy()
    pn_idx = np.where(roles_arr == "PN")[0].astype(np.int32)
    kc_idx = np.where(roles_arr == "KC")[0].astype(np.int32)
    mbon_idx = np.where(roles_arr == "MBON")[0].astype(np.int32)
    dan_idx = np.where(roles_arr == "DAN")[0].astype(np.int32)

    def filter_edges(src_set, dst_set):
        m = np.array([p in src_set and q in dst_set for p, q in zip(pre, post)], dtype=bool)
        return pre[m], post[m], w[m]

    e_pn_kc = filter_edges(set(pn_idx), set(kc_idx))
    e_kc_mbon = filter_edges(set(kc_idx), set(mbon_idx))
    e_dan_kc = filter_edges(set(dan_idx), set(kc_idx))
    e_dan_mbon = filter_edges(set(dan_idx), set(mbon_idx))

    ctypes = mb_conn["cell_type"].astype(str).to_numpy()
    dan_aversive = np.array(
        [i for i in dan_idx if str(ctypes[i]).upper().startswith("PPL1")], dtype=np.int32
    )
    dan_appetitive = np.array(
        [i for i in dan_idx if str(ctypes[i]).upper().startswith("PAM")], dtype=np.int32
    )

    kc_recv_av = set(e_dan_kc[1][np.isin(e_dan_kc[0], dan_aversive)]) if len(e_dan_kc[0]) else set()
    kc_recv_ap = set(e_dan_kc[1][np.isin(e_dan_kc[0], dan_appetitive)]) if len(e_dan_kc[0]) else set()
    mbon_av = set(e_dan_mbon[1][np.isin(e_dan_mbon[0], dan_aversive)]) if len(e_dan_mbon[0]) else set()
    mbon_ap = set(e_dan_mbon[1][np.isin(e_dan_mbon[0], dan_appetitive)]) if len(e_dan_mbon[0]) else set()
    mask_av = np.array(
        [(int(p) in kc_recv_av) or (int(q) in mbon_av) for p, q in zip(e_kc_mbon[0], e_kc_mbon[1])],
        dtype=np.bool_,
    ) if len(e_kc_mbon[0]) else np.array([], dtype=np.bool_)
    mask_ap = np.array(
        [(int(p) in kc_recv_ap) or (int(q) in mbon_ap) for p, q in zip(e_kc_mbon[0], e_kc_mbon[1])],
        dtype=np.bool_,
    ) if len(e_kc_mbon[0]) else np.array([], dtype=np.bool_)

    npz_path = ART / "connectome_mb_v1.npz"
    np.savez_compressed(
        npz_path,
        root_ids=mb_conn["root_id"].to_numpy(np.int64),
        roles=mb_conn["role"].to_numpy(),
        cell_types=mb_conn["cell_type"].astype(str).fillna("").to_numpy(),
        cell_classes=mb_conn["cell_class"].astype(str).fillna("").to_numpy(),
        sides=mb_conn["side"].astype(str).fillna("").to_numpy(),
        top_nt=mb_conn["top_nt"].astype(str).fillna("").to_numpy(),
        pn_idx=pn_idx,
        kc_idx=kc_idx,
        mbon_idx=mbon_idx,
        dan_idx=dan_idx,
        dan_aversive_idx=dan_aversive,
        dan_appetitive_idx=dan_appetitive,
        pn_kc_pre=e_pn_kc[0].astype(np.int32),
        pn_kc_post=e_pn_kc[1].astype(np.int32),
        pn_kc_w=e_pn_kc[2].astype(np.float32),
        kc_mbon_pre=e_kc_mbon[0].astype(np.int32),
        kc_mbon_post=e_kc_mbon[1].astype(np.int32),
        kc_mbon_w=e_kc_mbon[2].astype(np.float32),
        kc_mbon_mask_aversive=mask_av,
        kc_mbon_mask_appetitive=mask_ap,
        dan_kc_pre=e_dan_kc[0].astype(np.int32),
        dan_kc_post=e_dan_kc[1].astype(np.int32),
        dan_kc_w=e_dan_kc[2].astype(np.float32),
        dan_mbon_pre=e_dan_mbon[0].astype(np.int32),
        dan_mbon_post=e_dan_mbon[1].astype(np.int32),
        dan_mbon_w=e_dan_mbon[2].astype(np.float32),
        syn_threshold=np.array([SYN_THRESH], dtype=np.int32),
    )

    edges_out = conn_f.copy()
    role_map = dict(zip(mb_conn.root_id, mb_conn.role))
    edges_out["pre_role"] = edges_out["pre_pt_root_id"].map(role_map)
    edges_out["post_role"] = edges_out["post_pt_root_id"].map(role_map)
    edges_out.to_csv(ART / "connectome_mb_v1_edges.csv.gz", index=False, compression="gzip")
    mb_conn.to_csv(ART / "connectome_mb_v1_neurons.csv.gz", index=False, compression="gzip")

    if not ann_gz.exists():
        with ann_tsv.open("rb") as f_in, gzip.open(ann_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    conn_sha = hashlib.sha256(conn_path.read_bytes()).hexdigest()
    (RAW / "PROOFREAD_CONNECTIONS_SHA256.txt").write_text(
        f"{conn_sha}  proofread_connections_783.feather\n"
        f"source: https://zenodo.org/records/10676866\n"
        f"size_bytes: {conn_path.stat().st_size}\n"
    )

    manifest = {
        "artifact_id": "connectome_mb_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": "FAFB-783",
        "synapse_threshold": SYN_THRESH,
        "counts": {
            "neurons": int(len(mb_conn)),
            "PN": int(len(pn_idx)),
            "KC": int(len(kc_idx)),
            "MBON": int(len(mbon_idx)),
            "DAN": int(len(dan_idx)),
            "DAN_PPL1_aversive": int(len(dan_aversive)),
            "DAN_PAM_appetitive": int(len(dan_appetitive)),
            "edges_retained": int(len(conn_f)),
            "edges_pn_kc": int(len(e_pn_kc[0])),
            "edges_kc_mbon": int(len(e_kc_mbon[0])),
            "edges_dan_kc": int(len(e_dan_kc[0])),
            "edges_dan_mbon": int(len(e_dan_mbon[0])),
            "plastic_kc_mbon_aversive": int(mask_av.sum()) if len(mask_av) else 0,
            "plastic_kc_mbon_appetitive": int(mask_ap.sum()) if len(mask_ap) else 0,
        },
        "files": {
            "npz": "artifacts/connectome_mb_v1.npz",
            "neurons_csv_gz": "artifacts/connectome_mb_v1_neurons.csv.gz",
            "edges_csv_gz": "artifacts/connectome_mb_v1_edges.csv.gz",
        },
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "citations": [
            "Dorkenwald et al., Nature (2024)",
            "Schlegel et al., Nature (2024)",
            "Zenodo 10.5281/zenodo.10676866",
            "flyconnectome/flywire_annotations",
        ],
    }
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))
    print("wrote", npz_path)


if __name__ == "__main__":
    main()
