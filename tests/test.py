#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import csv
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime

# ===========================
#  HARD-CODED CONFIG / PATHS
# ===========================
INPUT_DIR   = r"D:\Desktop\Projectome SWC Files_cleaned"
OUTPUT_DIR  = r"D:\Desktop\Projectome SWC Files_radis_modified"
LOG_CSV     = "radius_changes.csv"   # will be created inside OUTPUT_DIR
PATTERN     = "*.swc"
RECURSIVE   = False
WORKERS     = 4

# Outlier detection (log-scale robust z) — LESS STRICT
LOG_Z_THRESHOLD = 4.5
MIN_NEIGHBORS   = 3

# Neighbor depth/cap (1 = parent+children; 2 = also neighbors-of-neighbors)
DEPTH        = 1
NEIGHBOR_MAX = 16

# Small positive floor for bad/zero radii
EPS_MIN_RADIUS_FACTOR = 1e-3

def fmt_radius(x: float) -> str:
    # mimic scientific tokens like 7.8125000000000e-03
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
#    Graph (FAST neighbors)
# ===========================
def id_to_index_map(arr: np.ndarray) -> Dict[int, int]:
    return {int(n["id"]): i for i, n in enumerate(arr)}

def build_parent_index(arr: np.ndarray) -> np.ndarray:
    """Return index of parent row for each row, or -1 if none/unknown."""
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
    """Undirected adjacency: parent+children per node (depth-1 neighborhood)."""
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
    """
    Precompute neighbor list per node.
    depth=1 -> direct neighbors (parent+children)
    depth=2 -> neighbors + neighbors-of-neighbors (dedup, capped)
    """
    if depth <= 1:
        return [nbrs[:max_n] for nbrs in adj]

    N = len(adj)
    out = []
    for u in range(N):
        seen = set([u])
        first = adj[u]
        acc = []
        for v in first:
            if v not in seen:
                seen.add(v); acc.append(v)
        # neighbors-of-neighbors
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
#  Robust log-scale stats
# ===========================
def robust_log_stats(vals: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    vals = np.array([v for v in vals if v > 0], dtype=float)
    if vals.size == 0:
        return None, None
    logv = np.log(vals)
    med = float(np.median(logv))
    mad = float(np.median(np.abs(logv - med)))
    return med, mad

def is_outlier_log(value: float, med_log: Optional[float], mad_log: Optional[float], z_thresh: float) -> bool:
    if value <= 0 or med_log is None or mad_log is None or mad_log == 0.0:
        return True
    z = abs(math.log(value) - med_log) / mad_log
    return z > z_thresh

# ===========================
#  Trusted neighbor gather
# ===========================
def trusted_neighbors(
    i: int,
    depth: int,
    max_neigh: int,
    neighbors: List[List[int]],
    parent_idx: np.ndarray,
    children: List[List[int]],
    outlier_mask: np.ndarray,
    arr: np.ndarray
) -> List[int]:
    # Start with depth-1 neighbors that are NOT outliers and have positive radius
    cand = [j for j in neighbors[i] if (not outlier_mask[j]) and (arr[j]["radius"] > 0)]
    if len(cand) >= MIN_NEIGHBORS:
        # Dedup + cap
        return list(dict.fromkeys(cand))[:max_neigh]

    # Add siblings if parent exists
    p = int(parent_idx[i])
    if p >= 0:
        sibs = [k for k in children[p] if k != i and (not outlier_mask[k]) and (arr[k]["radius"] > 0)]
        cand.extend(sibs)

    # Expand to depth-2 neighbors, still trusted only
    if len(cand) < MIN_NEIGHBORS and depth >= 2:
        seen = set([i]) | set(neighbors[i])
        for v in neighbors[i]:
            for w in neighbors[v]:
                if w in seen:
                    continue
                seen.add(w)
                if (not outlier_mask[w]) and (arr[w]["radius"] > 0):
                    cand.append(w)
                    if len(cand) >= max_neigh:
                        break
            if len(cand) >= max_neigh:
                break

    # Dedup + cap
    cand = list(dict.fromkeys(cand))[:max_neigh]
    return cand

# ===========================
#        Per-file pass
# ===========================
def process_one_file(
    in_path: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    base = os.path.basename(in_path)
    name_noext = os.path.splitext(base)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name_noext}.swc")

    arr = read_swc_file(in_path)
    N = len(arr)

    if N == 0:
        write_swc_file(arr, out_path)
        print(f"     (empty) wrote replica: {out_path}")
        return (base, [])

    # Build adjacency ONCE (parent + children)
    parent_idx = build_parent_index(arr)
    children   = build_children_lists(parent_idx)
    adj        = build_adj(parent_idx, children)
    neighbors  = precompute_neighbors(adj, depth=DEPTH, max_n=NEIGHBOR_MAX)

    types = arr["type"].astype(int)

    # -------- Per-type GLOBAL stats (for detection fallback) --------
    # Using all positive radii (preliminary, may include outliers)
    per_type_medlog: Dict[int, Optional[float]] = {}
    per_type_madlog: Dict[int, Optional[float]] = {}
    per_type_median_radius: Dict[int, float] = {}
    for t in np.unique(types):
        vals = np.array([float(arr[i]["radius"]) for i in range(N) if types[i] == t and arr[i]["radius"] > 0], dtype=float)
        if vals.size > 0:
            ml, md = robust_log_stats(vals)
            per_type_medlog[t] = ml
            per_type_madlog[t] = md
            per_type_median_radius[t] = float(np.median(vals))
        else:
            per_type_medlog[t] = None
            per_type_madlog[t] = None
            per_type_median_radius[t] = 0.0

    # Overall global stats
    all_r = np.array([float(r) for r in arr["radius"] if r > 0], dtype=float)
    g_med_log, g_mad_log = robust_log_stats(all_r)
    g_med_radius = float(np.median(all_r)) if all_r.size > 0 else 0.0
    min_radius_floor = max(g_med_radius * EPS_MIN_RADIUS_FACTOR, 1e-9)

    # -----------------------
    # Pass 1: DETECTION ONLY (type-aware)
    # -----------------------
    outlier_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        ri = float(arr[i]["radius"])
        ti = types[i]

        # First: same-type local neighbors
        neigh = neighbors[i]
        same_type_vals = [float(arr[j]["radius"]) for j in neigh if (types[j] == ti and arr[j]["radius"] > 0)]
        if len(same_type_vals) >= MIN_NEIGHBORS:
            med_use, mad_use = robust_log_stats(np.array(same_type_vals, dtype=float))
        else:
            # fallback: all-type local neighbors
            all_vals = [float(arr[j]["radius"]) for j in neigh if arr[j]["radius"] > 0]
            if len(all_vals) >= MIN_NEIGHBORS:
                med_use, mad_use = robust_log_stats(np.array(all_vals, dtype=float))
            else:
                # fallback: per-type global
                med_use, mad_use = per_type_medlog.get(ti), per_type_madlog.get(ti)
                if med_use is None or mad_use is None:
                    # final fallback: overall global
                    med_use, mad_use = g_med_log, g_mad_log

        outlier_mask[i] = is_outlier_log(ri, med_use, mad_use, LOG_Z_THRESHOLD)

    # Never modify soma; also don't treat soma as "bad" for replacement exclusion ambiguity:
    # We still keep their outlier flags (affects using soma as neighbor), but we SKIP changes later.
    # (If you want soma always usable as neighbors, uncomment the next line)
    # outlier_mask[(types == 1)] = False

    # -------- Per-type NON-OUTLIER medians for replacement fallback --------
    per_type_nonout_median: Dict[int, float] = {}
    for t in np.unique(types):
        good = np.array([float(arr[i]["radius"]) for i in range(N)
                         if types[i] == t and (not outlier_mask[i]) and arr[i]["radius"] > 0], dtype=float)
        per_type_nonout_median[t] = float(np.median(good)) if good.size > 0 else per_type_median_radius.get(t, 0.0)

    # Overall non-outlier median
    good_all = np.array([float(arr[i]["radius"]) for i in range(N)
                         if (not outlier_mask[i]) and arr[i]["radius"] > 0], dtype=float)
    g_good_med_radius = float(np.median(good_all)) if good_all.size > 0 else g_med_radius

    # -----------------------
    # Pass 2: REPLACEMENT (trusted neighbors, type-aware)
    # -----------------------
    changes: List[Dict[str, Any]] = []
    changed_count = 0

    for i in range(N):
        # Do NOT change soma
        if types[i] == 1:
            continue

        if not outlier_mask[i]:
            continue  # only modify flagged nodes

        cand_all = trusted_neighbors(
            i, DEPTH, NEIGHBOR_MAX, neighbors, parent_idx, children, outlier_mask, arr
        )
        # Prefer same-type candidates
        cand_same = [j for j in cand_all if types[j] == types[i]]

        # Build value/id lists with preference
        if len(cand_same) >= 1:
            n_vals = [float(arr[j]["radius"]) for j in cand_same]
            n_ids  = [int(arr[j]["id"]) for j in cand_same]
            method = "same_type_neighbor_median"
        elif len(cand_all) >= 1:
            n_vals = [float(arr[j]["radius"]) for j in cand_all]
            n_ids  = [int(arr[j]["id"]) for j in cand_all]
            method = "all_neighbor_median"
        else:
            n_vals = []
            n_ids = []
            method = "fallback"

        if n_vals:
            rep = float(np.median(n_vals))   # robust replacement
        else:
            # per-type non-outlier median → overall non-outlier → overall median → floor
            rep = per_type_nonout_median.get(types[i], 0.0)
            if rep <= 0:
                rep = g_good_med_radius if g_good_med_radius > 0 else g_med_radius
            if rep <= 0:
                rep = min_radius_floor
            method = "per_type_or_global_median_fallback"

        rep = max(rep, min_radius_floor)
        old = float(arr[i]["radius"])
        if rep != old:
            arr[i]["radius"] = rep
            arr[i]["radius_str"] = fmt_radius(rep)
            changed_count += 1
            changes.append({
                "file": base,
                "node_id": int(arr[i]["id"]),
                "type": int(arr[i]["type"]),
                "radius_old": old,
                "radius_new": rep,
                "method": method,
                "neighbor_count": len(n_vals),
                "neighbor_ids": ";".join(map(str, n_ids)) if n_ids else "",
                "neighbor_radii_used": ";".join(f"{v:.6g}" for v in n_vals) if n_vals else "",
                "stats_source": "type_aware_trusted",
                "global_med_log": g_med_log,
                "global_mad_log": g_mad_log,
            })

    write_swc_file(arr, out_path)
    return (base, changes)

# ===========================
#            MAIN
# ===========================
def main():
    print("=== FAST Type-Aware Trusted-Neighbor Radius QC ===")
    print(f"Input dir  : {os.path.abspath(INPUT_DIR)}")
    print(f"Output dir : {os.path.abspath(OUTPUT_DIR)}")
    pattern = "**/*.swc" if RECURSIVE else PATTERN
    print(f"Pattern    : {pattern}  (recursive={RECURSIVE})")
    print(f"Workers    : {WORKERS}")
    print(f"Params     : z={LOG_Z_THRESHOLD}, min_neigh={MIN_NEIGHBORS}, depth={DEPTH}, max_neigh={NEIGHBOR_MAX}")
    print("========================================")

    files = sorted(glob.glob(os.path.join(INPUT_DIR, pattern), recursive=RECURSIVE))
    if not files:
        print("No SWC files found. Check INPUT_DIR/PATTERN/RECURSIVE.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_changes: List[Dict[str, Any]] = []

    if WORKERS <= 1:
        for i, f in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {os.path.basename(f)}")
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
            "stats_source",
            "global_med_log",
            "global_mad_log",
        ])
        w.writeheader()
        for r in all_changes:
            # tidy number formatting for readability while keeping same columns
            r = dict(r)
            r["radius_old"] = f"{float(r['radius_old']):.6g}"
            r["radius_new"] = f"{float(r['radius_new']):.6g}"
            if r["global_med_log"] is not None:
                r["global_med_log"] = f"{float(r['global_med_log']):.6f}"
            else:
                r["global_med_log"] = ""
            if r["global_mad_log"] is not None:
                r["global_mad_log"] = f"{float(r['global_mad_log']):.6f}"
            else:
                r["global_mad_log"] = ""
            w.writerow(r)

    print(f"\nCSV log written to: {csv_path}")
    print(f"Total files changed: {len(set(r['file'] for r in all_changes))}  (nodes changed: {len(all_changes)})")
    print("=== Done ===")

if __name__ == "__main__":
    mp.freeze_support()   # Windows parallel safety
    main()
