# src/swctools/radius_qc.py
import os, glob, csv
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .config import load_section
from .io import read_swc_file, write_swc_file, fmt_radius
from .graph import build_parent_index, build_children_lists, build_adj, precompute_neighbors
from .stats import per_type_stats

def process_one_file(in_path: str, cfg: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    io=cfg["io"]; pol=cfg["policy"]; nb=cfg["neighbors"]; thr=cfg["thresholds"]
    base=os.path.basename(in_path); name_noext=os.path.splitext(base)[0]
    out_path=os.path.join(io["output_dir"], f"{name_noext}.swc")

    arr=read_swc_file(in_path)
    if len(arr)==0:
        write_swc_file(arr,out_path); return base, []

    parent_idx = build_parent_index(arr)
    children   = build_children_lists(parent_idx)
    adj        = build_adj(parent_idx, children)
    neighbors  = precompute_neighbors(adj, depth=nb["depth"], max_n=nb["neighbor_max"])
    types      = arr["type"].astype(int)

    mean_by_type, pL_by_type, pH_by_type = per_type_stats(arr, thr["p_low"], thr["p_high"], thr["min_per_type_count"])

    outlier = np.zeros(len(arr), dtype=bool)
    for i in range(len(arr)):
        r=float(arr[i]["radius"])
        if r<=0: outlier[i]=True; continue
        t=int(types[i]); pL=pL_by_type.get(t,0.0); pH=pH_by_type.get(t,0.0)
        if r<pL or r>pH: outlier[i]=True

    changes = []
    for i in range(len(arr)):
        if pol["skip_soma_changes"] and types[i] == 1: continue
        if not outlier[i]: continue

        t = int(types[i])
        cand_all = neighbors[i]
        cand_same = [j for j in cand_all if types[j] == t and arr[j]["radius"] > 0 and not outlier[j]]

        method = "";
        n_vals = [];
        n_ids = []
        if cand_same:
            n_vals = [float(arr[j]["radius"]) for j in cand_same]
            n_ids = [int(arr[j]["id"]) for j in cand_same]
            rep = float(np.mean(n_vals));
            method = "same_type_neighbor_mean"
        else:
            cand_same_all = [j for j in cand_all if types[j] == t and arr[j]["radius"] > 0]
            if cand_same_all:
                n_vals = [float(arr[j]["radius"]) for j in cand_same_all]
                n_ids = [int(arr[j]["id"]) for j in cand_same_all]
                rep = float(np.mean(n_vals));
                method = "same_type_neighbor_mean_all"
            else:
                rep = float(mean_by_type.get(t, 0.0))
                if rep <= 0: rep = float(pol["eps_min_radius"])
                method = "per_type_mean_fallback"

        old=float(arr[i]["radius"])
        if rep!=old:
            arr[i]["radius"]=rep
            arr[i]["radius_str"]=fmt_radius(rep)
            changes.append({
                "file": base, "node_id": int(arr[i]["id"]), "type": t,
                "radius_old": f"{old:.6g}", "radius_new": f"{rep:.6g}",
                "method": method, "neighbor_count": len(n_vals),
                "neighbor_ids": ";".join(map(str,n_ids)) if n_ids else "",
                "neighbor_radii_used": ";".join(f"{v:.6g}" for v in n_vals) if n_vals else "",
                "type_mean": f"{mean_by_type.get(t,0.0):.6g}",
                "type_pLow": f"{pL_by_type.get(t,0.0):.6g}",
                "type_pHigh": f"{pH_by_type.get(t,0.0):.6g}",
            })

    write_swc_file(arr, out_path)
    return base, changes

def process_dir(cfg: Dict[str, Any]):
    io=cfg["io"]
    pattern = "**/*.swc" if io["recursive"] else io["pattern"]
    files = sorted(glob.glob(os.path.join(io["input_dir"], pattern), recursive=io["recursive"]))
    if not files: raise FileNotFoundError("No SWC files found with current config")

    os.makedirs(io["output_dir"], exist_ok=True)

    all_changes=[]
    workers = int(cfg["compute"]["workers"])
    if workers<=1:
        for f in files:
            _, ch = process_one_file(f, cfg); all_changes.extend(ch)
    else:
        ctx=mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs={ex.submit(process_one_file, f, cfg): f for f in files}
            for fut in as_completed(futs):
                _, ch = fut.result(); all_changes.extend(ch)

    # logs
    with open(io["log_csv"], "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=[
            "file","node_id","type","radius_old","radius_new","method",
            "neighbor_count","neighbor_ids","neighbor_radii_used","type_mean","type_pLow","type_pHigh"
        ])
        w.writeheader(); [w.writerow(r) for r in all_changes]

    changed_counts = Counter(r["file"] for r in all_changes)
    with open(io["log_summary"], "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["file","nodes_changed"])
        w.writeheader()
        for f in sorted(changed_counts):
            w.writerow({"file": f, "nodes_changed": changed_counts[f]})

    return all_changes, changed_counts

def cli_main():
    import argparse
    from .config import load_section
    ap=argparse.ArgumentParser(description="Type-aware percentile radius QC")
    ap.add_argument("--config", required=True, help="Path to config.json")
    ap.add_argument("--section", default="radius_qc")
    args=ap.parse_args()
    cfg=load_section(args.config, args.section)
    process_dir(cfg)
