#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import csv
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional

import json
import numpy as np

# ==========================================================
#                     SWC I/O Helpers
# ==========================================================

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
    """
    Reads a SWC file into a numpy structured array with fields:
      id (i8), type (i4), x (f8), y (f8), z (f8), parent (i8),
      x_str (U128), y_str (U128), z_str (U128), radius (f8), radius_str (U128)

    - Skips comments (#) / blank lines.
    - PRESERVES the original textual tokens for x, y, z, and radius so we can write them back verbatim.
    """
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

            # numeric versions (for distance ops)
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
    """
    Writes SWC back to disk.
    - x, y, z, radius columns use the preserved original strings (x_str, y_str, z_str, radius_str) EXACTLY,
      unless coordinates were changed (e.g., after a shift), in which case x_str/y_str/z_str were updated.
    - id, type, parent written as integers.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ID type x y z radius parent\n")
        for n in arr:
            rid   = int(n["id"])
            rtype = int(n["type"])
            parent = int(n["parent"])
            xs = str(n["x_str"])
            ys = str(n["y_str"])
            zs = str(n["z_str"])
            rs = str(n["radius_str"])
            f.write(f"{rid} {rtype} {xs} {ys} {zs} {rs} {parent}\n")

# ==========================================================
#                 Graph / Topology Utilities
# ==========================================================

def id_to_index_map(arr: np.ndarray) -> Dict[int, int]:
    return {int(n["id"]): i for i, n in enumerate(arr)}

def build_children_map(arr: np.ndarray) -> Dict[int, List[int]]:
    """Return map: idx -> list of child indices (via parent links)."""
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
    """All descendants (excluding root_idx itself)."""
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
    """Adjacency by parent-child links, undirected."""
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
    """Return list of components; each is a list of indices (0-based)."""
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
    """Soma origins: type == 1 and parent == -1"""
    return [i for i, n in enumerate(arr) if int(n["type"]) == 1 and int(n["parent"]) == -1]

def find_non_soma_roots(arr: np.ndarray) -> List[int]:
    """All non-soma origins: parent == -1 and type != 1"""
    return [i for i, n in enumerate(arr) if int(n["parent"]) == -1 and int(n["type"]) != 1]

def soma_component_indices(arr: np.ndarray) -> Optional[List[int]]:
    """Return indices of the component containing the (single) soma root, else None."""
    soma_roots = find_soma_roots(arr)
    if not soma_roots:
        return None
    sroot = soma_roots[0]
    for comp in connected_components(arr):
        if sroot in comp:
            return comp
    return None

# ==========================================================
#            Edge Stats & Distance / Thresholds
# ==========================================================

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

def robust_threshold_T(edge_lengths: np.ndarray) -> float:
    if edge_lengths.size == 0:
        # Extremely conservative if no edges known (unlikely) → force split unless zero-distance merge
        return 0.0
    d_med = np.median(edge_lengths)
    mad = np.median(np.abs(edge_lengths - d_med))
    d_p95 = np.percentile(edge_lengths, 95)
    return float(max(1.25 * d_p95, d_med + 3 * mad))

# ==========================================================
#                   Zero-distance MERGE
# ==========================================================

def reindex_ids_preserve_strings(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    new_ids = np.arange(1, len(out) + 1, dtype=np.int64)
    old_ids = out["id"].astype(np.int64)
    remap = {int(o): int(n) for o, n in zip(old_ids, new_ids)}
    out["id"] = new_ids
    new_par = []
    for p in arr["parent"]:
        p = int(p)
        new_par.append(-1 if p == -1 else remap.get(p, -1))
    out["parent"] = np.array(new_par, dtype=np.int64)
    return out

def resolve_zero_distance_orphan(arr: np.ndarray, orphan_idx: int, eps: float = 0.0):
    """
    If 'orphan_idx' has a coincident node (distance <= eps), merge:
      - Redirect all children of orphan to the coincident node.
      - Delete the orphan row and reindex IDs/parents.
    Prefer a coincident node that is NOT a descendant; else pick a descendant.
    Returns (arr2, did_merge: bool, details: dict)
    """
    xi, yi, zi = arr[orphan_idx]["x"], arr[orphan_idx]["y"], arr[orphan_idx]["z"]
    coincident = []
    for j in range(len(arr)):
        if j == orphan_idx: continue
        if abs(arr[j]["x"] - xi) <= eps and abs(arr[j]["y"] - yi) <= eps and abs(arr[j]["z"] - zi) <= eps:
            coincident.append(j)
    if not coincident:
        return arr, False, {}

    kids_map = build_children_map(arr)
    desc = descendants_of(arr, orphan_idx, kids_map)

    chosen = None
    for j in coincident:
        if j not in desc:
            chosen = j
            break
    if chosen is None:
        chosen = coincident[0]

    orphan_id = int(arr[orphan_idx]["id"])
    chosen_id = int(arr[chosen]["id"])
    children_reparented = 0
    for i in range(len(arr)):
        if int(arr[i]["parent"]) == orphan_id:
            arr[i]["parent"] = chosen_id
            children_reparented += 1

    mask = np.ones(len(arr), dtype=bool); mask[orphan_idx] = False
    arr2 = arr[mask]
    arr2 = reindex_ids_preserve_strings(arr2)

    details = {
        "change": "merged_duplicate",
        "orphan_old_id": int(orphan_id),
        "coincident_id": int(chosen_id),
        "children_reparented": int(children_reparented),
        "nodes_removed": 1,
        "distance_before_shift": 0.0,     # coincident
        "threshold_T": None,
        "distance_after_shift": None
    }
    return arr2, True, details

# ==========================================================
#            Adaptive RECONNECT + optional SHIFT
# ==========================================================

def subtree_indices(arr: np.ndarray, root_idx: int) -> List[int]:
    kids_map = build_children_map(arr)
    return [root_idx] + sorted(list(descendants_of(arr, root_idx, kids_map)))

def nearest_non_descendant_in_soma(arr: np.ndarray, root_idx: int, soma_comp: List[int]) -> Tuple[Optional[int], float]:
    """Find nearest node in soma component, excluding the orphan itself and all its descendants."""
    kids_map = build_children_map(arr)
    excl = descendants_of(arr, root_idx, kids_map)
    excl.add(root_idx)
    cand = [j for j in soma_comp if j not in excl]
    if not cand:
        return None, float("inf")
    xi, yi, zi = arr[root_idx]["x"], arr[root_idx]["y"], arr[root_idx]["z"]
    best, best_d = None, float("inf")
    for j in cand:
        d = euclid3(xi, yi, zi, arr[j]["x"], arr[j]["y"], arr[j]["z"])
        if d < best_d:
            best, best_d = j, d
    return best, best_d

def consistent_offset_for_subtree(arr: np.ndarray, root_idx: int, soma_comp: List[int]) -> Tuple[np.ndarray, float]:
    """
    For each node in the orphan subtree, find nearest neighbor in soma component,
    return (median_offset_vector, MAD of offset magnitudes around the median).
    """
    comp_set = set(soma_comp)
    kids_map = build_children_map(arr)
    sub = subtree_indices(arr, root_idx)
    deltas = []
    for i in sub:
        xi, yi, zi = arr[i]["x"], arr[i]["y"], arr[i]["z"]
        best_j, best_d = None, float('inf')
        for j in comp_set:
            d = euclid3(xi, yi, zi, arr[j]["x"], arr[j]["y"], arr[j]["z"])
            if d < best_d:
                best_j, best_d = j, d
        dx = arr[best_j]["x"] - xi
        dy = arr[best_j]["y"] - yi
        dz = arr[best_j]["z"] - zi
        deltas.append((dx, dy, dz))
    deltas = np.array(deltas, float)
    med = np.median(deltas, axis=0)
    offs = np.linalg.norm(deltas - med, axis=1)
    mad = float(np.median(offs))
    return med, mad

def apply_shift_to_subtree(arr: np.ndarray, root_idx: int, delta: np.ndarray):
    """Apply a rigid translation to the entire subtree and update x_str/y_str/z_str."""
    kids_map = build_children_map(arr)
    sub = subtree_indices(arr, root_idx)
    dx, dy, dz = delta.tolist()
    for i in sub:
        arr[i]["x"] += dx; arr[i]["y"] += dy; arr[i]["z"] += dz
        arr[i]["x_str"] = f"{arr[i]['x']:.12g}"
        arr[i]["y_str"] = f"{arr[i]['y']:.12g}"
        arr[i]["z_str"] = f"{arr[i]['z']:.12g}"

def reconnect_or_split_orphan(
    arr: np.ndarray, root_idx: int, shift_accept_mad: float
) -> Tuple[np.ndarray, Optional[np.ndarray], str, Dict[str, Any]]:
    """
    Returns (new_arr, split_subtree_or_None, action, details)
      action ∈ {"merged", "reconnected", "shifted_reconnected", "split", "skipped"}
    """
    # 0) zero-distance duplicate merge?
    arr2, merged, merge_details = resolve_zero_distance_orphan(arr, root_idx, eps=0.0)
    if merged:
        return arr2, None, "merged", merge_details

    # 1) soma component
    soma_comp = soma_component_indices(arr)
    if soma_comp is None:
        sub_idx = subtree_indices(arr, root_idx)
        sub = extract_subgraph(arr, sub_idx)
        mask = np.ones(len(arr), dtype=bool)
        mask[sub_idx] = False
        remaining = reindex_ids_preserve_strings(arr[mask])
        details = {
            "change": "split",
            "root_id": int(arr[root_idx]["id"]),
            "subtree_size": int(len(sub_idx)),
            "threshold_T": None,
            "distance_before_shift": None,
            "distance_after_shift": None
        }
        return remaining, sub, "split", details

    # 2) adaptive threshold
    L = edge_lengths_in_component(arr, soma_comp)
    T = robust_threshold_T(L)

    # nearest non-descendant BEFORE any shift
    parent_idx, d_min = nearest_non_descendant_in_soma(arr, root_idx, soma_comp)
    if parent_idx is not None and d_min <= T:
        old_parent = int(arr[root_idx]["parent"])
        arr[root_idx]["parent"] = int(arr[parent_idx]["id"])
        details = {
            "change": "reconnected",
            "root_id": int(arr[root_idx]["id"]),
            "new_parent_id": int(arr[parent_idx]["id"]),
            "distance_before_shift": float(d_min),
            "threshold_T": float(T),
            "distance_after_shift": None,
            "old_parent": int(old_parent)
        }
        return arr, None, "reconnected", details

    # 3) try rigid translation
    delta, mad = consistent_offset_for_subtree(arr, root_idx, soma_comp)
    best_d_after_shift = None
    if mad <= shift_accept_mad:
        xi = arr[root_idx]["x"] + float(delta[0])
        yi = arr[root_idx]["y"] + float(delta[1])
        zi = arr[root_idx]["z"] + float(delta[2])

        kids_map = build_children_map(arr)
        excl = descendants_of(arr, root_idx, kids_map); excl.add(root_idx)

        best, best_d = None, float('inf')
        for j in soma_comp:
            if j in excl:
                continue
            d = euclid3(xi, yi, zi, arr[j]["x"], arr[j]["y"], arr[j]["z"])
            if d < best_d:
                best, best_d = j, d

        best_d_after_shift = float(best_d) if best is not None else None

        if best is not None and best_d <= T:
            apply_shift_to_subtree(arr, root_idx, delta)
            old_parent = int(arr[root_idx]["parent"])
            arr[root_idx]["parent"] = int(arr[best]["id"])
            sub_size = int(len(subtree_indices(arr, root_idx)))
            details = {
                "change": "shifted_reconnected",
                "root_id": int(arr[root_idx]["id"]),
                "new_parent_id": int(arr[best]["id"]),
                "distance_before_shift": float(d_min),
                "distance_after_shift": float(best_d),
                "threshold_T": float(T),
                "shift_dx": float(delta[0]),
                "shift_dy": float(delta[1]),
                "shift_dz": float(delta[2]),
                "subtree_nodes_shifted": int(sub_size),
                "offset_mad": float(mad),
                "old_parent": int(old_parent)
            }
            return arr, None, "shifted_reconnected", details

    # 4) split
    sub_idx = subtree_indices(arr, root_idx)
    sub = extract_subgraph(arr, sub_idx)
    mask = np.ones(len(arr), dtype=bool)
    mask[sub_idx] = False
    remaining = reindex_ids_preserve_strings(arr[mask])
    details = {
        "change": "split",
        "root_id": int(arr[root_idx]["id"]),
        "subtree_size": int(len(sub_idx)),
        "threshold_T": float(T),
        "distance_before_shift": float(d_min) if math.isfinite(d_min) else None,
        "distance_after_shift": best_d_after_shift
    }
    return remaining, sub, "split", details

# ==========================================================
#                Reindexing / Subgraph Extract
# ==========================================================

def reindex_ids(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    new_ids = np.arange(1, len(arr) + 1, dtype=np.int64)
    old_ids = arr["id"].astype(np.int64)
    remap = {int(old): int(new) for old, new in zip(old_ids, new_ids)}
    out["id"] = new_ids
    new_parents = []
    for p in arr["parent"]:
        p = int(p)
        new_parents.append(-1 if p == -1 else remap.get(p, -1))
    out["parent"] = np.array(new_parents, dtype=np.int64)
    return out

def extract_subgraph(arr: np.ndarray, indices: List[int]) -> np.ndarray:
    sub = arr[indices]
    return reindex_ids_preserve_strings(sub)

# ==========================================================
#                     Per-file Pipeline
# ==========================================================

def process_file(
    in_path: str,
    out_root: str,
    shift_accept_mad: float
) -> Dict[str, Any]:
    """
    For each file:
      - If multiple roots (parent==-1), process from the second root onward.
      - Always write the remaining/main neuron to <out>/<name>.swc (reindexed).
    """
    base = os.path.basename(in_path)
    name_noext = os.path.splitext(base)[0]
    out_dir = out_root
    os.makedirs(out_dir, exist_ok=True)

    arr = read_swc_file(in_path)
    outputs = []
    actions = []
    change_details: List[Dict[str, Any]] = []

    def current_roots(a: np.ndarray) -> List[int]:
        return [i for i, n in enumerate(a) if int(a[i]["parent"]) == -1]

    soma_split_index = 0
    orphan_split_index = 0
    rpos = 1
    while True:
        roots = current_roots(arr)
        if len(roots) <= 1 or rpos >= len(roots):
            break
        r = roots[rpos]
        node_type = int(arr[r]["type"])

        if node_type == 1:
            sub_idxs = subtree_indices(arr, r)
            sub = extract_subgraph(arr, sub_idxs)
            mask = np.ones(len(arr), dtype=bool); mask[sub_idxs] = False
            arr = reindex_ids_preserve_strings(arr[mask])
            soma_split_index += 1
            out_file = os.path.join(out_dir, f"{name_noext}_soma_{soma_split_index}.swc")
            write_swc_file(sub, out_file)
            outputs.append(os.path.relpath(out_file, out_dir))
            actions.append(f"soma_split[{soma_split_index}]")
            change_details.append({
                "change": "soma_split",
                "split_index": soma_split_index,
                "subtree_size": int(len(sub_idxs)),
                "output": os.path.relpath(out_file, out_dir),
                "distance_before_shift": None,
                "distance_after_shift": None,
                "threshold_T": None
            })
            continue

        new_arr, sub, action, details = reconnect_or_split_orphan(arr, r, shift_accept_mad=shift_accept_mad)
        arr = new_arr

        if action == "merged":
            actions.append("merged_duplicate")
            change_details.append(details)

        elif action == "reconnected":
            actions.append("reconnected")
            change_details.append(details)

        elif action == "shifted_reconnected":
            actions.append("shifted_reconnected")
            change_details.append(details)

        elif action == "split":
            orphan_split_index += 1
            out_file = os.path.join(out_dir, f"{name_noext}_orphan_{orphan_split_index}.swc")
            write_swc_file(sub, out_file)
            outputs.append(os.path.relpath(out_file, out_dir))
            actions.append(f"orphan_split[{orphan_split_index}]")
            details = {**details, "split_index": orphan_split_index, "output": os.path.relpath(out_file, out_dir)}
            change_details.append(details)
        else:
            actions.append("skipped")

    main_out = os.path.join(out_dir, f"{name_noext}.swc")
    write_swc_file(reindex_ids_preserve_strings(arr), main_out)
    outputs.append(os.path.relpath(main_out, out_dir))

    changed = len(change_details) > 0

    return {
        "input_file": base,
        "changed": changed,
        "num_nodes_out": int(len(arr)),
        "num_outputs": len(outputs),
        "actions": ";".join(actions) if actions else "unchanged_or_single_root",
        "outputs": outputs,
        "change_details": change_details
    }

# ==========================================================
#                    CSV Aggregation Helpers
# ==========================================================

def _safe_stats(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    v = [float(x) for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return None, None, None
    return float(min(v)), float(sum(v)/len(v)), float(max(v))

# ==========================================================
#                         CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Repair/split SWC files with extra roots (parent==-1). Second and subsequent roots are split if soma, or reconnected adaptively, else shifted+reconnected, else split."
    )
    parser.add_argument("input_dir", help="Folder containing .swc files")
    parser.add_argument("output_dir", help="Folder where cleaned/split outputs will be written")
    parser.add_argument("--pattern", default="*.swc", help="Glob pattern (default: *.swc)")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders recursively (uses **/*.swc).")
    parser.add_argument("--shift-accept-mad", type=float, default=5.0,
                        help="Max MAD (in units of x/y/z) to accept a rigid translation shift for an orphan subtree (default: 5.0).")
    parser.add_argument("--log-csv", default="swc_changes.csv",
                        help="Name of CSV log file (written inside output_dir).")
    args = parser.parse_args([
        r"D:\Desktop\Projectome SWC Files",
        r"D:\Desktop\Projectome SWC Files_cleaned",
    ])

    pattern = "**/*.swc" if args.recursive else args.pattern
    files = sorted(glob.glob(os.path.join(args.input_dir, pattern), recursive=args.recursive))
    if not files:
        print("No SWC files found. Check --pattern/--recursive and input path.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logs = []
    for f in files:
        rec = process_file(f, args.output_dir, shift_accept_mad=args.shift_accept_mad)
        logs.append(rec)
        outs = ", ".join(rec["outputs"]) if rec["outputs"] else "(no output)"
        print(f"[{rec['actions']:^28}] {rec['input_file']} -> {outs}")

    # -------- CSV: ONLY changed files, clear structure --------
    csv_path = os.path.join(args.output_dir, args.log_csv)
    with open(csv_path, "w", newline="", encoding="utf-8") as out:
        fields = [
            "input_file", "actions", "num_outputs", "change_types",
            # merged
            "merged_count", "merged_children_reparented_total", "merged_nodes_removed_total",
            # reconnected
            "reconnected_count", "reconnected_dist_min", "reconnected_dist_mean", "reconnected_dist_max",
            "reconnected_T_mean",
            # shifted_reconnected
            "shifted_reconnected_count", "shift_nodes_total",
            "shift_dx_mean", "shift_dy_mean", "shift_dz_mean",
            "shift_dist_before_mean", "shift_dist_after_mean", "shift_T_mean",
            # split
            "split_count", "split_subtree_nodes_total",
            "split_dist_before_mean", "split_dist_after_mean", "split_T_mean",
            # readable summary
            "summary"
        ]
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()

        for r in logs:
            if not r.get("changed"):
                continue

            details: List[Dict[str, Any]] = r.get("change_details", [])
            change_types = [d.get("change", "unknown") for d in details]

            # MERGED
            merged = [d for d in details if d.get("change") == "merged_duplicate"]
            merged_count = len(merged)
            merged_children_total = sum(int(d.get("children_reparented", 0)) for d in merged)
            merged_removed_total = sum(int(d.get("nodes_removed", 0)) for d in merged)

            # RECONNECTED
            reconn = [d for d in details if d.get("change") == "reconnected"]
            reconn_count = len(reconn)
            reconn_dists = [d.get("distance_before_shift") for d in reconn]
            rd_min, rd_mean, rd_max = _safe_stats(reconn_dists)
            rT_mean = None
            rT_vals = [d.get("threshold_T") for d in reconn if d.get("threshold_T") is not None]
            if rT_vals:
                rT_mean = float(sum(rT_vals) / len(rT_vals))

            # SHIFTED_RECONNECTED
            sh = [d for d in details if d.get("change") == "shifted_reconnected"]
            sh_count = len(sh)
            sh_nodes_total = sum(int(d.get("subtree_nodes_shifted", 0)) for d in sh)
            sdx = [d.get("shift_dx") for d in sh]; _, sdx_mean, _ = _safe_stats(sdx)
            sdy = [d.get("shift_dy") for d in sh]; _, sdy_mean, _ = _safe_stats(sdy)
            sdz = [d.get("shift_dz") for d in sh]; _, sdz_mean, _ = _safe_stats(sdz)
            sdb = [d.get("distance_before_shift") for d in sh]; _, sdb_mean, _ = _safe_stats(sdb)
            sda = [d.get("distance_after_shift") for d in sh]; _, sda_mean, _ = _safe_stats(sda)
            sT_vals = [d.get("threshold_T") for d in sh if d.get("threshold_T") is not None]
            sT_mean = float(sum(sT_vals) / len(sT_vals)) if sT_vals else None

            # SPLIT
            sp = [d for d in details if d.get("change") == "split" or d.get("change") == "soma_split"]
            sp_count = len(sp)
            sp_nodes_total = sum(int(d.get("subtree_size", 0)) for d in sp)
            sp_db = [d.get("distance_before_shift") for d in sp]; _, sp_db_mean, _ = _safe_stats(sp_db)
            sp_da = [d.get("distance_after_shift") for d in sp]; _, sp_da_mean, _ = _safe_stats(sp_da)
            sp_T = [d.get("threshold_T") for d in sp if d.get("threshold_T") is not None]
            sp_T_mean = float(sum(sp_T) / len(sp_T)) if sp_T else None

            summary = []
            if merged_count:
                summary.append(f"merged {merged_count} dup(s)")
            if reconn_count:
                summary.append(f"reconnected {reconn_count} (mean d={rd_mean:.3f})" if rd_mean is not None else f"reconnected {reconn_count}")
            if sh_count:
                summary.append(f"shift+reconnected {sh_count} (mean shift=({sdx_mean:.3f},{sdy_mean:.3f},{sdz_mean:.3f}), after d≈{sda_mean:.3f})" if sda_mean is not None and sdx_mean is not None else f"shift+reconnected {sh_count}")
            if sp_count:
                summary.append(f"split {sp_count} (mean subtree={sp_nodes_total//sp_count if sp_count else 0})")

            w.writerow({
                "input_file": r["input_file"],
                "actions": r.get("actions", ""),
                "num_outputs": r.get("num_outputs", 0),
                "change_types": ";".join(change_types),
                "merged_count": merged_count,
                "merged_children_reparented_total": merged_children_total,
                "merged_nodes_removed_total": merged_removed_total,
                "reconnected_count": reconn_count,
                "reconnected_dist_min": rd_min,
                "reconnected_dist_mean": rd_mean,
                "reconnected_dist_max": rd_max,
                "reconnected_T_mean": rT_mean,
                "shifted_reconnected_count": sh_count,
                "shift_nodes_total": sh_nodes_total,
                "shift_dx_mean": sdx_mean,
                "shift_dy_mean": sdy_mean,
                "shift_dz_mean": sdz_mean,
                "shift_dist_before_mean": sdb_mean,
                "shift_dist_after_mean": sda_mean,
                "shift_T_mean": sT_mean,
                "split_count": sp_count,
                "split_subtree_nodes_total": sp_nodes_total,
                "split_dist_before_mean": sp_db_mean,
                "split_dist_after_mean": sp_da_mean,
                "split_T_mean": sp_T_mean,
                "summary": "; ".join(summary) if summary else ""
            })

    print(f"\nSummary (changed files only) written to: {csv_path}")

# if __name__ == "__main__":
    main()

