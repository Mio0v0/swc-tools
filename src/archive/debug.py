# debug_shift_split_trace.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import json

# -------------------------------
# SWC I/O (preserve xyz/radius strings)
# -------------------------------

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
            x_str = parts[2]
            y_str = parts[3]
            z_str = parts[4]
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

# -------------------------------
# Graph utilities
# -------------------------------

def id_to_index_map(arr: np.ndarray) -> Dict[int, int]:
    return {int(n["id"]): i for i, n in enumerate(arr)}

def build_children_map(arr: np.ndarray) -> Dict[int, List[int]]:
    id2idx = id_to_index_map(arr)
    kids = {i: [] for i in range(len(arr))}
    for i in range(len(arr)):
        p = int(arr[i]["parent"])
        if p == -1:
            continue
        j = id2idx.get(p)
        if j is not None:
            kids[j].append(i)
    return kids

def descendants_of(arr: np.ndarray, root_idx: int, kids_map: Dict[int, List[int]]) -> set:
    out = set()
    stack = list(kids_map.get(root_idx, []))
    while stack:
        u = stack.pop()
        if u in out:
            continue
        out.add(u)
        stack.extend(kids_map.get(u, []))
    return out

def build_undirected_adjacency(arr: np.ndarray) -> List[List[int]]:
    N = len(arr)
    adj = [[] for _ in range(N)]
    id2idx = id_to_index_map(arr)
    for i in range(N):
        p = int(arr[i]["parent"])
        if p == -1:
            continue
        j = id2idx.get(p, None)
        if j is None:
            continue
        adj[i].append(j)
        adj[j].append(i)
    return adj

def connected_components(arr: np.ndarray) -> List[List[int]]:
    N = len(arr)
    adj = build_undirected_adjacency(arr)
    seen = [False] * N
    comps = []
    for i in range(N):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps

def find_soma_roots(arr: np.ndarray) -> List[int]:
    # "main soma root" = type==1 and parent==-1
    return [i for i, n in enumerate(arr) if int(n["type"]) == 1 and int(n["parent"]) == -1]

def soma_component_indices(arr: np.ndarray) -> Optional[List[int]]:
    soma_roots = find_soma_roots(arr)
    if not soma_roots:
        return None
    sroot = soma_roots[0]
    for comp in connected_components(arr):
        if sroot in comp:
            return comp
    return None

# -------------------------------
# Distances, thresholds, nearest search
# -------------------------------

def euclid3(ax, ay, az, bx, by, bz) -> float:
    dx = ax - bx; dy = ay - by; dz = az - bz
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def edge_lengths_in_component(arr: np.ndarray, comp_indices: List[int]) -> np.ndarray:
    id2idx = id_to_index_map(arr)
    comp_set = set(comp_indices)
    lengths = []
    for i in comp_indices:
        p = int(arr[i]["parent"])
        if p == -1:
            continue
        j = id2idx.get(p)
        if j is None or j not in comp_set:
            continue
        lengths.append(euclid3(arr[i]["x"], arr[i]["y"], arr[i]["z"],
                               arr[j]["x"], arr[j]["y"], arr[j]["z"]))
    return np.array(lengths, dtype=float)

def robust_threshold_T(edge_lengths: np.ndarray) -> Tuple[float, float, float, float]:
    if edge_lengths.size == 0:
        return 0.0, float("nan"), float("nan"), float("nan")
    d_med = float(np.median(edge_lengths))
    mad = float(np.median(np.abs(edge_lengths - d_med)))
    d_p95 = float(np.percentile(edge_lengths, 95))
    T = float(max(1.25 * d_p95, d_med + 3 * mad))
    return T, d_med, mad, d_p95

def build_exclusion_set(arr: np.ndarray, root_idx: int) -> set:
    kids_map = build_children_map(arr)
    excl = descendants_of(arr, root_idx, kids_map)
    excl.add(root_idx)
    return excl

def nearest_non_descendant_in_set(arr: np.ndarray, q_idx: int, candidate_indices: List[int], exclude: set
                                 ) -> Tuple[Optional[int], float]:
    xi, yi, zi = arr[q_idx]["x"], arr[q_idx]["y"], arr[q_idx]["z"]
    best, best_d = None, float("inf")
    for j in candidate_indices:
        if j in exclude:
            continue
        d = euclid3(xi, yi, zi, arr[j]["x"], arr[j]["y"], arr[j]["z"])
        if d < best_d:
            best, best_d = j, d
    return best, best_d

def topk_non_descendants(arr: np.ndarray, q_idx: int, candidate_indices: List[int], exclude: set, k: int = 3
                        ) -> List[Tuple[int, float]]:
    xi, yi, zi = arr[q_idx]["x"], arr[q_idx]["y"], arr[q_idx]["z"]
    pairs = []
    for j in candidate_indices:
        if j in exclude:
            continue
        d = euclid3(xi, yi, zi, arr[j]["x"], arr[j]["y"], arr[j]["z"])
        pairs.append((j, d))
    pairs.sort(key=lambda t: t[1])
    return [(int(arr[j]["id"]), float(d)) for j, d in pairs[:k]]

# -------------------------------
# Rigid shift estimation
# -------------------------------

def consistent_offset_for_subtree(arr: np.ndarray, root_idx: int, soma_comp: List[int]) -> Tuple[np.ndarray, float]:
    comp_set = set(soma_comp)
    kids_map = build_children_map(arr)
    sub = [root_idx] + sorted(list(descendants_of(arr, root_idx, kids_map)))
    deltas = []
    for i in sub:
        xi, yi, zi = arr[i]["x"], arr[i]["y"], arr[i]["z"]
        # nearest in soma comp
        best_j, best_d = None, float("inf")
        for j in comp_set:
            d = euclid3(xi, yi, zi, arr[j]["x"], arr[j]["y"], arr[j]["z"])
            if d < best_d:
                best_j, best_d = j, d
        dx = arr[best_j]["x"] - xi
        dy = arr[best_j]["y"] - yi
        dz = arr[best_j]["z"] - zi
        deltas.append((dx, dy, dz))
    deltas = np.array(deltas, float)
    med = np.median(deltas, axis=0)          # proposed rigid shift vector
    offs = np.linalg.norm(deltas - med, axis=1)
    mad = float(np.median(offs))              # consistency of the offset
    return med, mad

# -------------------------------
# Debug tracer
# -------------------------------

def trace_shift_vs_split(
    swc_path: str,
    out_csv: str,
    shift_accept_mad: float = 5.0
) -> None:
    arr = read_swc_file(swc_path)
    if len(arr) == 0:
        raise RuntimeError("Empty or unreadable SWC.")

    # collect roots (parent == -1) in file order
    roots = [i for i in range(len(arr)) if int(arr[i]["parent"]) == -1]
    # nothing to do if <= 1 root
    extra_roots = roots[1:] if len(roots) > 1 else []

    soma_comp = soma_component_indices(arr)
    # Prepare CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file",
            "total_rows",
            "root_row_index_1based",
            "root_row_of_total",
            "root_id",
            "root_type",
            "subtree_size",
            # thresholding context
            "edge_median",
            "edge_MAD",
            "edge_p95",
            "T",
            # nearest BEFORE shift
            "distance_before_shift",
            "nearest_parent_id_before",
            "top3_before_json",
            # offset/shift context
            "offset_dx",
            "offset_dy",
            "offset_dz",
            "offset_MAD",
            # nearest AFTER shift (virtual)
            "distance_after_shift",
            "nearest_parent_id_after",
            "top3_after_json",
            # outcome
            "decision",
            "reason"
        ])
        w.writeheader()

        for r in extra_roots:
            root_id = int(arr[r]["id"])
            root_type = int(arr[r]["type"])
            kids_map = build_children_map(arr)
            desc = descendants_of(arr, r, kids_map)
            subtree_size = 1 + len(desc)

            # Candidate set = soma component if present; otherwise all nodes
            if soma_comp is None:
                candidate_indices = list(range(len(arr)))
                # No soma component → by policy we split; but still log distances for insight
                T = float("nan"); d_med = float("nan"); mad_edges = float("nan"); p95 = float("nan")
            else:
                candidate_indices = soma_comp
                L = edge_lengths_in_component(arr, soma_comp)
                T, d_med, mad_edges, p95 = robust_threshold_T(L)

            exclude = set([r]) | desc

            # BEFORE shift
            parent_idx_before, d_before = nearest_non_descendant_in_set(arr, r, candidate_indices, exclude)
            top3_before = topk_non_descendants(arr, r, candidate_indices, exclude, k=3)
            nearest_id_before = int(arr[parent_idx_before]["id"]) if parent_idx_before is not None else None

            decision = None
            reason = None
            nearest_id_after = None
            d_after = None
            delta = np.array([0.0, 0.0, 0.0])
            offset_mad = None
            top3_after = []

            if soma_comp is None:
                decision = "split"
                reason = "no_soma_component"
            else:
                # If already within T, this would be a reconnect (no split) — but we’re here to explain split/shift-fail.
                if parent_idx_before is None:
                    decision = "split"
                    reason = "no_non_descendant_candidate"
                elif d_before <= T:
                    decision = "reconnect"
                    reason = "distance_before_shift<=T"
                else:
                    # Try shift
                    delta, offset_mad = consistent_offset_for_subtree(arr, r, soma_comp)
                    # Only “accept” a shift if offset MAD is small enough
                    if offset_mad is not None and offset_mad <= shift_accept_mad:
                        # virtual coordinates of root after shift
                        xi = arr[r]["x"] + float(delta[0])
                        yi = arr[r]["y"] + float(delta[1])
                        zi = arr[r]["z"] + float(delta[2])

                        best, best_d = None, float('inf')
                        for j in candidate_indices:
                            if j in exclude:
                                continue
                            d = euclid3(xi, yi, zi, arr[j]["x"], arr[j]["y"], arr[j]["z"])
                            if d < best_d:
                                best, best_d = j, d
                        if best is not None:
                            d_after = float(best_d)
                            nearest_id_after = int(arr[best]["id"])
                            # build top3 AFTER shift
                            pairs = []
                            for j in candidate_indices:
                                if j in exclude:
                                    continue
                                d = euclid3(xi, yi, zi, arr[j]["x"], arr[j]["y"], arr[j]["z"])
                                pairs.append((j, d))
                            pairs.sort(key=lambda t: t[1])
                            top3_after = [(int(arr[j]["id"]), float(d)) for j, d in pairs[:3]]
                        else:
                            d_after = None
                            nearest_id_after = None
                            top3_after = []

                        if d_after is not None and d_after <= T:
                            decision = "shift_reconnect"
                            reason = "distance_after_shift<=T"
                        else:
                            decision = "split"
                            reason = "after_shift>T"
                    else:
                        decision = "split"
                        reason = "offset_mad>threshold"

            # Only record rows that ended up split OR that explain the (rare) reconnect outcome
            # (If you strictly want only failures, you can filter for decision=='split'.)
            w.writerow({
                "file": os.path.basename(swc_path),
                "total_rows": len(arr),
                "root_row_index_1based": r + 1,
                "root_row_of_total": f"{r+1}/{len(arr)}",
                "root_id": root_id,
                "root_type": root_type,
                "subtree_size": subtree_size,
                "edge_median": d_med if soma_comp is not None else None,
                "edge_MAD": mad_edges if soma_comp is not None else None,
                "edge_p95": p95 if soma_comp is not None else None,
                "T": T if soma_comp is not None else None,
                "distance_before_shift": d_before if math.isfinite(d_before) else None,
                "nearest_parent_id_before": nearest_id_before,
                "top3_before_json": json.dumps(top3_before),
                "offset_dx": float(delta[0]) if delta is not None else None,
                "offset_dy": float(delta[1]) if delta is not None else None,
                "offset_dz": float(delta[2]) if delta is not None else None,
                "offset_MAD": offset_mad,
                "distance_after_shift": d_after,
                "nearest_parent_id_after": nearest_id_after,
                "top3_after_json": json.dumps(top3_after),
                "decision": decision,
                "reason": reason
            })

    print(f"Wrote step-by-step trace to: {out_csv}")

# -------------------------------
# CLI
# -------------------------------

def main():
    # ==== HARD-CODED INPUTS ====
    swc_path = r"D:\Desktop\Projectome SWC Files\202461_072.swc"
    out_csv  = r"D:\Desktop\Projectome SWC Files_cleaned\debug_202461_072.csv"
    shift_accept_mad = 5.0  # tweak if you want stricter/looser shift acceptance

    # Make sure the output folder exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Run the trace
    trace_shift_vs_split(swc_path, out_csv, shift_accept_mad=shift_accept_mad)

if __name__ == "__main__":
    main()

