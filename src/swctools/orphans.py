# src/swctools/orphans.py
import os, glob, csv, math
from typing import Dict, Any, List, Tuple, Optional

from .config import load_section
from .io import read_swc_file, write_swc_file
from .graph import (
    build_children_map,
    descendants_of,
    connected_components,
    subtree_indices,
)
from .stats import euclid3, edge_lengths_in_component, robust_threshold_T


def reindex_ids_preserve_strings(arr):
    import numpy as np

    out = arr.copy()
    new_ids = np.arange(1, len(out) + 1, dtype=np.int64)
    old_ids = out["id"].astype(np.int64)
    remap = {int(o): int(n) for o, n in zip(old_ids, new_ids)}
    out["id"] = new_ids
    out["parent"] = np.array(
        [-1 if int(p) == -1 else remap.get(int(p), -1) for p in arr["parent"]],
        dtype=np.int64,
    )
    return out


def extract_subgraph(arr, indices):
    return reindex_ids_preserve_strings(arr[indices])


def soma_component_indices(arr):
    soma_roots = [
        i for i, n in enumerate(arr) if int(n["type"]) == 1 and int(n["parent"]) == -1
    ]
    if not soma_roots:
        return None
    sroot = soma_roots[0]
    for comp in connected_components(arr):
        if sroot in comp:
            return comp
    return None


# ---------- MERGE (zero-distance duplicate) ----------
def resolve_zero_distance_orphan(arr, orphan_idx, eps=0.0):
    xi, yi, zi = (
        arr[orphan_idx]["x"],
        arr[orphan_idx]["y"],
        arr[orphan_idx]["z"],
    )
    coinc = []
    for j in range(len(arr)):
        if j == orphan_idx:
            continue
        if (
            abs(arr[j]["x"] - xi) <= eps
            and abs(arr[j]["y"] - yi) <= eps
            and abs(arr[j]["z"] - zi) <= eps
        ):
            coinc.append(j)
    if not coinc:
        return arr, False, {}

    kids_map = build_children_map(arr)
    desc = descendants_of(arr, orphan_idx, kids_map)

    chosen = next((j for j in coinc if j not in desc), coinc[0])

    orphan_id = int(arr[orphan_idx]["id"])
    chosen_id = int(arr[chosen]["id"])
    reparented_child_ids: List[int] = []
    children_reparented = 0
    for i in range(len(arr)):
        if int(arr[i]["parent"]) == orphan_id:
            arr[i]["parent"] = chosen_id
            children_reparented += 1
            reparented_child_ids.append(int(arr[i]["id"]))

    import numpy as np

    mask = np.ones(len(arr), dtype=bool)
    mask[orphan_idx] = False
    arr2 = reindex_ids_preserve_strings(arr[mask])

    details = {
        "change": "merged_duplicate",
        "orphan_old_id": orphan_id,
        "coincident_id": chosen_id,
        "children_reparented": children_reparented,
        "reparented_child_ids": reparented_child_ids,  # for node-level log
        "nodes_removed": 1,
        "distance_before_shift": 0.0,
        "threshold_T": None,
        "distance_after_shift": None,
    }
    return arr2, True, details


def nearest_non_descendant_in_soma(arr, root_idx, soma_comp):
    kids_map = build_children_map(arr)
    excl = descendants_of(arr, root_idx, kids_map)
    excl.add(root_idx)
    cand = [j for j in soma_comp if j not in excl]
    if not cand:
        return None, float("inf")
    xi, yi, zi = arr[root_idx]["x"], arr[root_idx]["y"], arr[root_idx]["z"]
    best = None
    best_d = float("inf")
    for j in cand:
        d = euclid3(xi, yi, zi, arr[j]["x"], arr[j]["y"], arr[j]["z"])
        if d < best_d:
            best, best_d = j, d
    return best, best_d


def consistent_offset_for_subtree(arr, root_idx, soma_comp):
    import numpy as np

    comp = set(soma_comp)
    kids_map = build_children_map(arr)
    sub = subtree_indices(arr, root_idx)
    deltas = []
    for i in sub:
        xi, yi, zi = arr[i]["x"], arr[i]["y"], arr[i]["z"]
        best_j = None
        best_d = float("inf")
        for j in comp:
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


def apply_shift_to_subtree(arr, root_idx, delta):
    kids_map = build_children_map(arr)
    sub = subtree_indices(arr, root_idx)
    dx, dy, dz = delta.tolist()
    for i in sub:
        arr[i]["x"] += dx
        arr[i]["y"] += dy
        arr[i]["z"] += dz
        arr[i]["x_str"] = f"{arr[i]['x']:.12g}"
        arr[i]["y_str"] = f"{arr[i]['y']:.12g}"
        arr[i]["z_str"] = f"{arr[i]['z']:.12g}"


def reconnect_or_split_orphan(arr, root_idx, shift_accept_mad: float):
    """
    Returns (new_arr, split_subgraph_or_None, action, details)
    details contains metrics and, for splits, a 'split_nodes' list with:
        {"node_id_pre", "parent_before", "parent_after"}
    """
    # 0) zero-distance duplicate merge?
    arr2, merged, merge_details = resolve_zero_distance_orphan(arr, root_idx, eps=0.0)
    if merged:
        return arr2, None, "merged", merge_details

    # 1) if there's no soma component, split the subtree
    soma_comp = soma_component_indices(arr)
    if soma_comp is None:
        sub_idx = subtree_indices(arr, root_idx)
        sub = extract_subgraph(arr, sub_idx)

        # build per-node mapping
        old_ids = [int(arr[i]["id"]) for i in sub_idx]
        old_pars = [int(arr[i]["parent"]) for i in sub_idx]
        split_nodes = []
        for k, old_id in enumerate(old_ids):
            split_nodes.append(
                {
                    "node_id_pre": old_id,
                    "parent_before": old_pars[k],
                    "parent_after": int(sub[k]["parent"]),
                }
            )

        import numpy as np

        mask = np.ones(len(arr), dtype=bool)
        mask[sub_idx] = False
        remaining = reindex_ids_preserve_strings(arr[mask])
        details = {
            "change": "split",
            "root_id": int(arr[root_idx]["id"]),
            "subtree_size": int(len(sub_idx)),
            "threshold_T": None,
            "distance_before_shift": None,
            "distance_after_shift": None,
            "split_nodes": split_nodes,
        }
        return remaining, sub, "split", details

    # 2) adaptive threshold against soma component
    L = edge_lengths_in_component(arr, soma_comp)
    T = robust_threshold_T(L)

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
            "old_parent": int(old_parent),
        }
        return arr, None, "reconnected", details

    # 3) try rigid translation then reconnect
    delta, mad = consistent_offset_for_subtree(arr, root_idx, soma_comp)
    best_d_after_shift = None
    if mad <= shift_accept_mad:
        xi = arr[root_idx]["x"] + float(delta[0])
        yi = arr[root_idx]["y"] + float(delta[1])
        zi = arr[root_idx]["z"] + float(delta[2])

        kids_map = build_children_map(arr)
        excl = descendants_of(arr, root_idx, kids_map)
        excl.add(root_idx)

        best = None
        best_d = float("inf")
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
            sub_size = len(subtree_indices(arr, root_idx))
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
                "old_parent": old_parent,
            }
            return arr, None, "shifted_reconnected", details

    # 4) split (threshold failed or shift rejected)
    sub_idx = subtree_indices(arr, root_idx)
    sub = extract_subgraph(arr, sub_idx)

    old_ids = [int(arr[i]["id"]) for i in sub_idx]
    old_pars = [int(arr[i]["parent"]) for i in sub_idx]
    split_nodes = []
    for k, old_id in enumerate(old_ids):
        split_nodes.append(
            {
                "node_id_pre": old_id,
                "parent_before": old_pars[k],
                "parent_after": int(sub[k]["parent"]),
            }
        )

    import numpy as np

    mask = np.ones(len(arr), dtype=bool)
    mask[sub_idx] = False
    remaining = reindex_ids_preserve_strings(arr[mask])
    details = {
        "change": "split",
        "root_id": int(arr[root_idx]["id"]),
        "subtree_size": int(len(sub_idx)),
        "threshold_T": float(T),
        "distance_before_shift": float(d_min) if math.isfinite(d_min) else None,
        "distance_after_shift": best_d_after_shift,
        "split_nodes": split_nodes,
    }
    return remaining, sub, "split", details


def process_file(in_path: str, out_root: str, shift_accept_mad: float) -> Dict[str, Any]:
    base = os.path.basename(in_path)
    name_noext = os.path.splitext(base)[0]
    os.makedirs(out_root, exist_ok=True)

    arr = read_swc_file(in_path)
    outputs: List[str] = []
    actions: List[str] = []
    change_details: List[Dict[str, Any]] = []

    def current_roots(a):
        return [i for i in range(len(a)) if int(a[i]["parent"]) == -1]

    soma_split_index = 0
    orphan_split_index = 0
    rpos = 1
    while True:
        roots = current_roots(arr)
        if len(roots) <= 1 or rpos >= len(roots):
            break
        r = roots[rpos]
        node_type = int(arr[r]["type"])

        # -------- SOMA split (type==1) --------
        if node_type == 1:
            sub_idxs = subtree_indices(arr, r)
            sub = extract_subgraph(arr, sub_idxs)

            # build split_nodes mapping for unified CSV
            old_ids = [int(arr[idx]["id"]) for idx in sub_idxs]
            old_pars = [int(arr[idx]["parent"]) for idx in sub_idxs]
            split_nodes = []
            for k, old_id in enumerate(old_ids):
                split_nodes.append(
                    {
                        "node_id_pre": old_id,
                        "parent_before": old_pars[k],
                        "parent_after": int(sub[k]["parent"]),
                    }
                )

            import numpy as np

            mask = np.ones(len(arr), dtype=bool)
            mask[sub_idxs] = False
            arr = reindex_ids_preserve_strings(arr[mask])

            soma_split_index += 1
            out_file = os.path.join(out_root, f"{name_noext}_soma_{soma_split_index}.swc")
            write_swc_file(sub, out_file)
            outputs.append(os.path.relpath(out_file, out_root))
            actions.append(f"soma_split[{soma_split_index}]")
            change_details.append(
                {
                    "change": "soma_split",
                    "split_index": soma_split_index,
                    "subtree_size": int(len(sub_idxs)),
                    "output": os.path.relpath(out_file, out_root),
                    "split_nodes": split_nodes,
                    "distance_before_shift": None,
                    "distance_after_shift": None,
                    "threshold_T": None,
                }
            )
            continue

        # -------- Orphan: reconnect / shift+reconnect / split / merge --------
        new_arr, sub, action, details = reconnect_or_split_orphan(
            arr, r, shift_accept_mad
        )
        arr = new_arr

        if action in ("merged", "reconnected", "shifted_reconnected"):
            actions.append(action)
            change_details.append(details)

        elif action == "split":
            orphan_split_index += 1
            out_file = os.path.join(out_root, f"{name_noext}_orphan_{orphan_split_index}.swc")
            write_swc_file(sub, out_file)
            outputs.append(os.path.relpath(out_file, out_root))
            actions.append(f"orphan_split[{orphan_split_index}]")
            details = {
                **details,
                "split_index": orphan_split_index,
                "output": os.path.relpath(out_file, out_root),
            }
            change_details.append(details)
        else:
            actions.append("skipped")

    # write main (remaining) neuron
    main_out = os.path.join(out_root, f"{name_noext}.swc")
    write_swc_file(reindex_ids_preserve_strings(arr), main_out)
    outputs.append(os.path.relpath(main_out, out_root))

    return {
        "input_file": base,
        "changed": bool(change_details),
        "num_nodes_out": int(len(arr)),
        "num_outputs": len(outputs),
        "actions": ";".join(actions) if actions else "unchanged_or_single_root",
        "outputs": outputs,
        "change_details": change_details,
    }


# ---------- Unified node-level orphan changes CSV ----------
def write_orphan_changes_csv(logs: List[Dict[str, Any]], csv_path: str):
    """
    Single CSV with one row per node-level change across all orphan operations.
    Columns:
      input_file, output_file, node_id, change_type,
      parent_before, parent_after, trigger_root_id,
      distance_before, distance_after, threshold_T,
      shift_dx, shift_dy, shift_dz, coincident_id
    """
    fields = [
        "input_file",
        "output_file",
        "node_id",
        "change_type",
        "parent_before",
        "parent_after",
        "trigger_root_id",
        "distance_before",
        "distance_after",
        "threshold_T",
        "shift_dx",
        "shift_dy",
        "shift_dz",
        "coincident_id",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()

        for rec in logs:
            if not rec.get("changed"):
                continue

            input_file = rec.get("input_file")
            main_output = rec["outputs"][-1] if rec.get("outputs") else ""

            for d in rec.get("change_details", []):
                chg = d.get("change")

                # merged_duplicate: removed root + reparented children
                if chg == "merged_duplicate":
                    orphan_id = d.get("orphan_old_id")
                    coinc = d.get("coincident_id")

                    # removed orphan root
                    w.writerow(
                        {
                            "input_file": input_file,
                            "output_file": "",
                            "node_id": orphan_id,
                            "change_type": chg,
                            "parent_before": -1,
                            "parent_after": "",
                            "trigger_root_id": orphan_id,
                            "distance_before": d.get("distance_before_shift"),
                            "distance_after": d.get("distance_after_shift"),
                            "threshold_T": d.get("threshold_T"),
                            "shift_dx": "",
                            "shift_dy": "",
                            "shift_dz": "",
                            "coincident_id": coinc,
                        }
                    )

                    # each child reparented orphan->coincident
                    for cid in d.get("reparented_child_ids", []) or []:
                        w.writerow(
                            {
                                "input_file": input_file,
                                "output_file": main_output,
                                "node_id": cid,
                                "change_type": chg,
                                "parent_before": orphan_id,
                                "parent_after": coinc,
                                "trigger_root_id": orphan_id,
                                "distance_before": "",
                                "distance_after": "",
                                "threshold_T": "",
                                "shift_dx": "",
                                "shift_dy": "",
                                "shift_dz": "",
                                "coincident_id": coinc,
                            }
                        )

                # reconnected: root parent -1 -> new_parent_id
                elif chg == "reconnected":
                    root_id = d.get("root_id")
                    w.writerow(
                        {
                            "input_file": input_file,
                            "output_file": main_output,
                            "node_id": root_id,
                            "change_type": chg,
                            "parent_before": d.get("old_parent", -1),
                            "parent_after": d.get("new_parent_id"),
                            "trigger_root_id": root_id,
                            "distance_before": d.get("distance_before_shift"),
                            "distance_after": d.get("distance_after_shift"),
                            "threshold_T": d.get("threshold_T"),
                            "shift_dx": "",
                            "shift_dy": "",
                            "shift_dz": "",
                            "coincident_id": "",
                        }
                    )

                # shifted_reconnected: root parent change + shift metrics
                elif chg == "shifted_reconnected":
                    root_id = d.get("root_id")
                    w.writerow(
                        {
                            "input_file": input_file,
                            "output_file": main_output,
                            "node_id": root_id,
                            "change_type": chg,
                            "parent_before": d.get("old_parent", -1),
                            "parent_after": d.get("new_parent_id"),
                            "trigger_root_id": root_id,
                            "distance_before": d.get("distance_before_shift"),
                            "distance_after": d.get("distance_after_shift"),
                            "threshold_T": d.get("threshold_T"),
                            "shift_dx": d.get("shift_dx"),
                            "shift_dy": d.get("shift_dy"),
                            "shift_dz": d.get("shift_dz"),
                            "coincident_id": "",
                        }
                    )

                # split/soma_split: one row per node moved to split file
                elif chg in ("split", "soma_split"):
                    split_file = d.get("output", "")
                    trig = d.get("root_id")
                    nodes = d.get("split_nodes", []) or []
                    if nodes:
                        for n in nodes:
                            w.writerow(
                                {
                                    "input_file": input_file,
                                    "output_file": split_file,
                                    "node_id": n.get("node_id_pre"),
                                    "change_type": "split_orphan"
                                    if chg == "split"
                                    else "split_soma",
                                    "parent_before": n.get("parent_before"),
                                    "parent_after": n.get("parent_after"),
                                    "trigger_root_id": trig,
                                    "distance_before": d.get("distance_before_shift"),
                                    "distance_after": d.get("distance_after_shift"),
                                    "threshold_T": d.get("threshold_T"),
                                    "shift_dx": "",
                                    "shift_dy": "",
                                    "shift_dz": "",
                                    "coincident_id": "",
                                }
                            )
                    else:
                        # fallback: at least log the root
                        w.writerow(
                            {
                                "input_file": input_file,
                                "output_file": split_file,
                                "node_id": trig,
                                "change_type": "split_orphan"
                                if chg == "split"
                                else "split_soma",
                                "parent_before": -1,
                                "parent_after": -1,
                                "trigger_root_id": trig,
                                "distance_before": d.get("distance_before_shift"),
                                "distance_after": d.get("distance_after_shift"),
                                "threshold_T": d.get("threshold_T"),
                                "shift_dx": "",
                                "shift_dy": "",
                                "shift_dz": "",
                                "coincident_id": "",
                            }
                        )


# ---------- (existing) file-level summary ----------
def write_summary(logs: List[Dict[str, Any]], csv_path: str):
    def _safe_stats(vals):
        v = [
            float(x)
            for x in vals
            if x is not None and not (isinstance(x, float) and math.isnan(x))
        ]
        if not v:
            return None, None, None
        return min(v), sum(v) / len(v), max(v)

    with open(csv_path, "w", newline="", encoding="utf-8") as out:
        fields = [
            "input_file",
            "actions",
            "num_outputs",
            "change_types",
            "merged_count",
            "merged_children_reparented_total",
            "merged_nodes_removed_total",
            "reconnected_count",
            "reconnected_dist_min",
            "reconnected_dist_mean",
            "reconnected_dist_max",
            "reconnected_T_mean",
            "shifted_reconnected_count",
            "shift_nodes_total",
            "shift_dx_mean",
            "shift_dy_mean",
            "shift_dz_mean",
            "shift_dist_before_mean",
            "shift_dist_after_mean",
            "shift_T_mean",
            "split_count",
            "split_subtree_nodes_total",
            "split_dist_before_mean",
            "split_dist_after_mean",
            "split_T_mean",
            "summary",
        ]
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        for r in logs:
            if not r.get("changed"):
                continue
            details = r.get("change_details", [])
            change_types = [d.get("change", "unknown") for d in details]
            merged = [d for d in details if d.get("change") == "merged_duplicate"]
            reconn = [d for d in details if d.get("change") == "reconnected"]
            sh = [d for d in details if d.get("change") == "shifted_reconnected"]
            sp = [d for d in details if d.get("change") in ("split", "soma_split")]

            m_cnt = len(merged)
            m_kids = sum(int(d.get("children_reparented", 0)) for d in merged)
            m_rm = sum(int(d.get("nodes_removed", 0)) for d in merged)

            def mean_or_none(xs):
                xs = [
                    x
                    for x in xs
                    if x is not None and not (isinstance(x, float) and math.isnan(x))
                ]
                return sum(xs) / len(xs) if xs else None

            def stats_triplet(xs):
                a, b, c = _safe_stats(xs)
                return a, b, c

            rd_min, rd_mean, rd_max = stats_triplet(
                [d.get("distance_before_shift") for d in reconn]
            )
            rT_mean = mean_or_none(
                [d.get("threshold_T") for d in reconn if d.get("threshold_T") is not None]
            )

            sh_cnt = len(sh)
            sh_nodes = sum(int(d.get("subtree_nodes_shifted", 0)) for d in sh)
            sdx = mean_or_none([d.get("shift_dx") for d in sh])
            sdy = mean_or_none([d.get("shift_dy") for d in sh])
            sdz = mean_or_none([d.get("shift_dz") for d in sh])
            sdb = mean_or_none([d.get("distance_before_shift") for d in sh])
            sda = mean_or_none([d.get("distance_after_shift") for d in sh])
            sT_mean = mean_or_none(
                [d.get("threshold_T") for d in sh if d.get("threshold_T") is not None]
            )

            sp_cnt = len(sp)
            sp_nodes = sum(int(d.get("subtree_size", 0)) for d in sp)
            sp_db = mean_or_none([d.get("distance_before_shift") for d in sp])
            sp_da = mean_or_none([d.get("distance_after_shift") for d in sp])
            sp_T_mean = mean_or_none(
                [d.get("threshold_T") for d in sp if d.get("threshold_T") is not None]
            )

            summary = []
            if m_cnt:
                summary.append(f"merged {m_cnt}")
            if reconn:
                summary.append(
                    f"reconnected {len(reconn)} (mean d={rd_mean:.3f})"
                    if rd_mean is not None
                    else f"reconnected {len(reconn)}"
                )
            if sh_cnt:
                summary.append(f"shift+reconnected {sh_cnt}")
            if sp_cnt:
                summary.append(
                    f"split {sp_cnt} (mean subtree={sp_nodes // sp_cnt if sp_cnt else 0})"
                )

            w.writerow(
                {
                    "input_file": r["input_file"],
                    "actions": r.get("actions", ""),
                    "num_outputs": r.get("num_outputs", 0),
                    "change_types": ";".join(change_types),
                    "merged_count": m_cnt,
                    "merged_children_reparented_total": m_kids,
                    "merged_nodes_removed_total": m_rm,
                    "reconnected_count": len(reconn),
                    "reconnected_dist_min": rd_min,
                    "reconnected_dist_mean": rd_mean,
                    "reconnected_dist_max": rd_max,
                    "reconnected_T_mean": rT_mean,
                    "shifted_reconnected_count": sh_cnt,
                    "shift_nodes_total": sh_nodes,
                    "shift_dx_mean": sdx,
                    "shift_dy_mean": sdy,
                    "shift_dz_mean": sdz,
                    "shift_dist_before_mean": sdb,
                    "shift_dist_after_mean": sda,
                    "shift_T_mean": sT_mean,
                    "split_count": sp_cnt,
                    "split_subtree_nodes_total": sp_nodes,
                    "split_dist_before_mean": sp_db,
                    "split_dist_after_mean": sp_da,
                    "split_T_mean": sp_T_mean,
                    "summary": "; ".join(summary) if summary else "",
                }
            )


def process_dir(cfg: Dict[str, Any]):
    io = cfg["io"]
    pattern = "**/*.swc" if io["recursive"] else io["pattern"]
    files = sorted(
        glob.glob(os.path.join(io["input_dir"], pattern), recursive=io["recursive"])
    )
    if not files:
        raise FileNotFoundError("No SWC files found for orphan_clean")
    os.makedirs(io["output_dir"], exist_ok=True)

    logs = []
    for f in files:
        rec = process_file(f, io["output_dir"], cfg["reconnect"]["shift_accept_mad"])
        logs.append(rec)
        outs = ", ".join(rec["outputs"]) if rec["outputs"] else "(no output)"
        print(f"[{rec['actions']:^28}] {rec['input_file']} -> {outs}")

    # unified node-level changes CSV
    unified_csv = os.path.join(io["output_dir"], "orphan_changes_all.csv")
    write_orphan_changes_csv(logs, unified_csv)

    # keep existing file-level summary CSV
    write_summary(logs, os.path.join(io["output_dir"], io["log_csv"]))
    return logs


def cli_main():
    import argparse

    ap = argparse.ArgumentParser(description="Clean orphans / extra roots in SWC")
    ap.add_argument("--config", required=True, help="Path to config.json")
    ap.add_argument("--section", default="orphan_clean")
    args = ap.parse_args()
    cfg = load_section(args.config, args.section)
    process_dir(cfg)
