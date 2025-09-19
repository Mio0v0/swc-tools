#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import csv
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import Counter

# ===========================
#  HARD-CODED CONFIG / PATHS
# ===========================
INPUT_DIR   = r"D:\Desktop\Projectome SWC Files_cleaned"
OUTPUT_DIR  = r"D:\Desktop\Projectome SWC Files_radius_pct_modified"
LOG_CSV     = "radius_changes_percentile.csv"      # node-level change log (inside OUTPUT_DIR)
LOG_SUMMARY = "files_changed_summary.csv"          # NEW: file-level summary (inside OUTPUT_DIR)
PATTERN     = "*.swc"
RECURSIVE   = False
WORKERS     = 4

# Percentile-based outlier rule (per type, per file)
P_LOW       = 2.5        # lower percentile cutoff
P_HIGH      = 97.5       # upper percentile cutoff
MIN_PER_TYPE_COUNT = 20  # if fewer than this many positive radii for a type, fall back to global percentiles

# Neighborhood search
DEPTH        = 1         # 1 = parent+children; 2 = plus neighbors-of-neighbors
NEIGHBOR_MAX = 16        # cap neighbors considered

# Policy
SKIP_SOMA_CHANGES = True # do not change type==1 radii
EPS_MIN_RADIUS    = 1e-9 # tiny floor to avoid zeros/negatives in fallback

def fmt_radius(x: float) -> str:
    # match scientific tokens like 7.8125000000000e-03
    return f"{float(x):.13e}"

# ===========================
#       I/O Helpers
# ===========================
def _safe_int(x: str, default: int = -1) -> int:
    try:
        xl = str(x).strip().lower()
        if xl in ("na", "nan", ""):
            return default
        return int(float(x))
    except Exception:
        return default

def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        if isinstance(x, (int, float)):
            return float(x)
        xl = str(x).strip().lower()
        if xl in ("na", "nan", ""):
            return default
        return float(x)
    except Exception:
        return default

def read_swc_file(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            rid   = _safe_int(parts[0])
            rtype = _safe_int(parts[1])
            x_str = parts[2]; y_str = parts[3]; z_str = parts[4]
            rrad_str = parts[5]
            rpar  = _safe_int(parts[6])

            rx    = _safe_float(x_str)
            ry    = _safe_float(y_str)
            rz    = _safe_float(z_str)
            rrad  = _safe_float(rrad_str)

            rows.append((rid, rtype, rx, ry, rz, rpar, x_str, y_str, z_str, rrad, rrad_str))

    dtype = np.dtype([
        ("id", "i8"),
        ("type", "i4"),
        ("x", "f8"),
        ("y", "f8"),
        ("z", "f8"),
        ("parent", "i8"),
        ("x_str", "U128"),
        ("y_str", "U128"),
        ("z_str", "U128"),
        ("radius", "f8"),
        ("radius_str", "U128"),
    ])
    if not rows:
        return np.empty((0,), dtype=dtype)
    return np.array(rows, dtype=dtype)

def write_swc_file(arr: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ID type x y z radius parent\n")
        for n in arr:
            rid = int(n["id"]); rtype = int(n["type"]); parent = int(n["parent"])
            xs = str(n["x_str"]); ys = str(n["y_str"]); zs = str(n["z_str"])
            rs = str(n["radius_str"])
            f.write(f"{rid} {rtype} {xs} {ys} {zs} {rs} {parent}\n")

# ===========================
#    Graph (neighbors)
# ===========================
def id_to_index_map(arr: np.ndarray) -> Dict[int, int]:
    return {int(n["id"]): i for i, n in enumerate(arr)}

def build_parent_index(arr: np.ndarray) -> np.ndarray:
    id2idx = id_to_index_map(arr)
    N = len(arr)
    parent_idx = np.full(N, -1, dtype=np.int64)
    for i in range(N):
        p = int(arr[i]["parent"])
        if p == -1:
            continue
        j = id2idx.get(p, -1)
        parent_idx[i] = j
    return parent_idx

def build_children_lists(parent_idx: np.ndarray) -> List[List[int]]:
    N = len(parent_idx)
    kids = [[] for _ in range(N)]
    for i in range(N):
        j = int(parent_idx[i])
        if j >= 0:
            kids[j].append(i)
    return kids

def build_adj(parent_idx: np.ndarray, children: List[List[int]]) -> List[List[int]]:
    N = len(parent_idx)
    adj = [[] for _ in range(N)]
    for i in range(N):
        p = int(parent_idx[i])
        if p >= 0:
            adj[i].append(p)
        if children[i]:
            adj[i].extend(children[i])
    return adj

def precompute_neighbors(adj: List[List[int]], depth: int, max_n: int) -> List[List[int]]:
    if depth <= 1:
        return [nbrs[:max_n] for nbrs in adj]
    # depth==2: neighbors + neighbors-of-neighbors
    N = len(adj)
    out = []
    for u in range(N):
        seen = set([u])
        first = adj[u]
        acc = []
        for v in first:
            if v not in seen:
                seen.add(v); acc.append(v)
        for v in first:
            for w in adj[v]:
                if w in seen:
                    continue
                seen.add(w); acc.append(w)
                if len(acc) >= max_n:
                    break
            if len(acc) >= max_n:
                break
        out.append(acc[:max_n])
    return out

# ===========================
#     Per-type statistics
# ===========================
def per_type_stats(arr: np.ndarray, p_low: float, p_high: float, min_count: int
                   ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Returns:
      mean_by_type, p_low_by_type, p_high_by_type
    """
    types = arr["type"].astype(int)
    mean_by_type: Dict[int, float] = {}
    pL_by_type: Dict[int, float]   = {}
    pH_by_type: Dict[int, float]   = {}

    # global fallback if per-type is too small
    all_pos = np.array([float(r) for r in arr["radius"] if r > 0], dtype=float)
    g_pL = float(np.percentile(all_pos, p_low)) if all_pos.size else 0.0
    g_pH = float(np.percentile(all_pos, p_high)) if all_pos.size else 0.0
    g_mean = float(np.mean(all_pos)) if all_pos.size else 0.0

    for t in np.unique(types):
        vals = np.array([float(arr[i]["radius"]) for i in range(len(arr))
                         if types[i] == t and arr[i]["radius"] > 0], dtype=float)
        if vals.size >= max(3, min_count):
            mean_by_type[t] = float(np.mean(vals))
            pL_by_type[t]   = float(np.percentile(vals, p_low))
            pH_by_type[t]   = float(np.percentile(vals, p_high))
        else:
            mean_by_type[t] = g_mean
            pL_by_type[t]   = g_pL
            pH_by_type[t]   = g_pH
    return mean_by_type, pL_by_type, pH_by_type

# ===========================
#      Per-file processing
# ===========================
def process_one_file(in_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    base = os.path.basename(in_path)
    name_noext = os.path.splitext(base)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name_noext}.swc")

    arr = read_swc_file(in_path)
    N = len(arr)

    if N == 0:
        write_swc_file(arr, out_path)
        print(f"     (empty) wrote replica: {out_path}")
        return (base, [])

    parent_idx = build_parent_index(arr)
    children   = build_children_lists(parent_idx)
    adj        = build_adj(parent_idx, children)
    neighbors  = precompute_neighbors(adj, depth=DEPTH, max_n=NEIGHBOR_MAX)
    types      = arr["type"].astype(int)

    mean_by_type, pL_by_type, pH_by_type = per_type_stats(
        arr, P_LOW, P_HIGH, MIN_PER_TYPE_COUNT
    )

    # Flag outliers (per type, per file)
    outlier_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        r = float(arr[i]["radius"])
        if r <= 0:
            outlier_mask[i] = True
            continue
        t = int(types[i])
        pL = pL_by_type.get(t, 0.0)
        pH = pH_by_type.get(t, 0.0)
        if (r < pL) or (r > pH):
            outlier_mask[i] = True

    # Replace flagged nodes
    changes: List[Dict[str, Any]] = []
    for i in range(N):
        if SKIP_SOMA_CHANGES and types[i] == 1:
            continue
        if not outlier_mask[i]:
            continue

        t = int(types[i])

        # same-type neighbor IDs (prefer parent/children, optionally depth-2)
        cand_all = neighbors[i]
        cand_same = [j for j in cand_all if types[j] == t and arr[j]["radius"] > 0 and not outlier_mask[j]]

        method = ""
        if len(cand_same) >= 1:
            # mean of non-outlier same-type neighbor radii
            n_vals = [float(arr[j]["radius"]) for j in cand_same]
            n_ids  = [int(arr[j]["id"]) for j in cand_same]
            rep = float(np.mean(n_vals))
            method = "same_type_neighbor_mean"
        else:
            # fallback: allow all same-type neighbors with positive radius (even if flagged)
            cand_same_all = [j for j in cand_all if types[j] == t and arr[j]["radius"] > 0]
            if len(cand_same_all) >= 1:
                n_vals = [float(arr[j]["radius"]) for j in cand_same_all]
                n_ids  = [int(arr[j]["id"]) for j in cand_same_all]
                rep = float(np.mean(n_vals))
                method = "same_type_neighbor_mean_all"
            else:
                # fallback: per-type mean, then tiny floor
                rep = float(mean_by_type.get(t, 0.0))
                n_vals, n_ids = [], []
                if rep <= 0:
                    rep = EPS_MIN_RADIUS
                method = "per_type_mean_fallback"

        old = float(arr[i]["radius"])
        if rep != old:
            arr[i]["radius"] = rep
            arr[i]["radius_str"] = fmt_radius(rep)
            changes.append({
                "file": base,
                "node_id": int(arr[i]["id"]),
                "type": t,
                "radius_old": f"{old:.6g}",
                "radius_new": f"{rep:.6g}",
                "method": method,
                "neighbor_count": len(n_vals),
                "neighbor_ids": ";".join(map(str, n_ids)) if n_ids else "",
                "neighbor_radii_used": ";".join(f"{v:.6g}" for v in n_vals) if n_vals else "",
                "type_mean": f"{mean_by_type.get(t, 0.0):.6g}",
                "type_pLow": f"{pL_by_type.get(t, 0.0):.6g}",
                "type_pHigh": f"{pH_by_type.get(t, 0.0):.6g}",
            })

    write_swc_file(arr, out_path)
    return (base, changes)

# ===========================
#            MAIN
# ===========================
def main():
    print("=== Type-aware Percentile Radius QC ===")
    print(f"Input dir  : {os.path.abspath(INPUT_DIR)}")
    print(f"Output dir : {os.path.abspath(OUTPUT_DIR)}")
    pattern = "**/*.swc" if RECURSIVE else PATTERN
    print(f"Pattern    : {pattern}  (recursive={RECURSIVE})")
    print(f"Workers    : {WORKERS}")
    print(f"Cutoffs    : P_LOW={P_LOW}  P_HIGH={P_HIGH}  (per type)")
    print(f"Policy     : SKIP_SOMA_CHANGES={SKIP_SOMA_CHANGES}, depth={DEPTH}, neighbor_max={NEIGHBOR_MAX}")
    print("========================================")

    files = sorted(glob.glob(os.path.join(INPUT_DIR, pattern), recursive=RECURSIVE))
    if not files:
        print("No SWC files found. Check INPUT_DIR/PATTERN/RECURSIVE.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_changes: List[Dict[str, Any]] = []

    if WORKERS <= 1:
        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {os.path.basename(f)}")
            _, ch = process_one_file(f)
            if ch:
                all_changes.extend(ch)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
            futs = {ex.submit(process_one_file, f): f for f in files}
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    base, ch = fut.result()
                    print(f"[{k}/{len(files)}] done {base}  (changes: {len(ch)})")
                    if ch:
                        all_changes.extend(ch)
                except Exception as e:
                    print(f"[{k}/{len(files)}] ERROR: {futs[fut]} -> {e}")

    # Write CSV log (only changed nodes)
    csv_path = os.path.join(OUTPUT_DIR, LOG_CSV)
    with open(csv_path, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=[
            "file",
            "node_id",
            "type",
            "radius_old",
            "radius_new",
            "method",
            "neighbor_count",
            "neighbor_ids",
            "neighbor_radii_used",
            "type_mean",
            "type_pLow",
            "type_pHigh",
        ])
        w.writeheader()
        for r in all_changes:
            w.writerow(r)

    # ===== NEW: file-level summary CSV (only files that changed) =====
    changed_counts = Counter(r["file"] for r in all_changes)
    summary_rows = [{"file": f, "nodes_changed": changed_counts[f]} for f in sorted(changed_counts)]
    csv_path_summary = os.path.join(OUTPUT_DIR, LOG_SUMMARY)
    with open(csv_path_summary, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["file", "nodes_changed"])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"\nNode-level CSV written to: {csv_path}")
    print(f"File-level summary CSV written to: {csv_path_summary}")
    print(f"Total files changed: {len(changed_counts)}  (nodes changed: {len(all_changes)})")
    print("=== Done ===")

if __name__ == "__main__":
    mp.freeze_support()   # Windows parallel safety (needed on Windows when using multiprocessing)
    main()
