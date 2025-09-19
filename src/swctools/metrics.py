# src/swctools/metrics.py
from __future__ import annotations
import os, re
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import pandas as pd

from .io import read_swc_file

# -------------------------------------------------------------------
# Canonical type order (used everywhere for consistent plotting)
# -------------------------------------------------------------------
NEURON_TYPES = ["SUBdd", "ProSub", "SUBv", "SUBvv", "SUBdv"]

# soft pastel fill colors per type (header colors in your sheet)
TYPE_COLORS = {
    "SUBdd": "#e7c0c0",  # red-ish
    "ProSub": "#f4dfc7", # orange-ish
    "SUBv":   "#dfead7", # green-ish
    "SUBvv":  "#d6e2f6", # blue-ish
    "SUBdv":  "#e0d7ea", # purple-ish
}

# -------------------------------------------------------------------
# Excel reader: two-row header with merged top cells
# -------------------------------------------------------------------
def _normalize_type_header(raw: str) -> str | None:
    """
    Robust match for top header cells like:
    "SUBdd (Red)", "ProSub (Orange)", "SUBv(Green)", "SUBvv (Blue)", "SUBdv (Purple)"
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)        # remove (...) e.g. "(Red)"
    s = re.sub(r"[^a-z]+", " ", s)          # non letters -> space
    s = re.sub(r"\s+", " ", s).strip()

    if "prosub" in s:
        return "ProSub"
    if "subvv" in s:
        return "SUBvv"
    if "subdv" in s:
        return "SUBdv"
    # 'subv' (exact word) before 'subdd' to avoid partial overlaps
    if re.search(r"\bsubv\b", s):
        return "SUBv"
    if "subdd" in s:
        return "SUBdd"
    return None


def read_celltype_table(xlsx_path: str, sheet: str | int | None = 0) -> Dict[str, List[str]]:
    """
    Layout (like your screenshot):
      Row 1: merged headers = neuron types (with colors in names)
      Row 2: under each type: columns 'Layer 1', 'Layer 3', 'Layer 4', ...
      Body : SWC ids listed down each layer column
    Returns: { type -> [swc_id, ...] } (deduped, order-preserved)
    """
    df = pd.read_excel(xlsx_path, header=[0, 1], dtype=str, sheet_name=sheet)

    # Forward-fill merged top headers across their columns
    top_raw = [t if isinstance(t, str) and t.strip() else None for (t, _) in df.columns]
    for i in range(len(top_raw)):
        if top_raw[i] is None and i > 0:
            top_raw[i] = top_raw[i - 1]
    layers = [l for (_, l) in df.columns]
    df.columns = pd.MultiIndex.from_tuples(list(zip(top_raw, layers)))

    ids_by_type: Dict[str, List[str]] = {t: [] for t in NEURON_TYPES}

    for (raw_type, layer) in df.columns:
        if raw_type is None:
            continue
        t_norm = _normalize_type_header(raw_type)
        if t_norm is None:
            continue

        col = df[(raw_type, layer)].astype(str)
        vals: List[str] = []
        for v in col:
            s = (v or "").strip()
            if not s or s.lower() in ("nan", "none"):
                continue
            swc_id = s.split()[0]
            vals.append(swc_id)
        ids_by_type[t_norm].extend(vals)

    # De-duplicate (preserve order)
    for t in ids_by_type:
        seen: Set[str] = set()
        uniq: List[str] = []
        for v in ids_by_type[t]:
            if v not in seen:
                seen.add(v); uniq.append(v)
        ids_by_type[t] = uniq

    return ids_by_type

# -------------------------------------------------------------------
# Metric helpers on a single SWC (numpy structured array from read_swc_file)
# -------------------------------------------------------------------
def _parent_index(arr) -> np.ndarray:
    id2idx = {int(n["id"]): i for i, n in enumerate(arr)}
    par = np.full(len(arr), -1, dtype=np.int64)
    for i in range(len(arr)):
        p = int(arr[i]["parent"])
        if p != -1:
            par[i] = id2idx.get(p, -1)
    return par

def _children_lists(par_idx: np.ndarray) -> list[list[int]]:
    kids = [[] for _ in range(len(par_idx))]
    for i, p in enumerate(par_idx):
        if p >= 0:
            kids[p].append(i)
    return kids

def _euclid3(a, b) -> float:
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    dz = float(a["z"]) - float(b["z"])
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))

def _root_index(arr) -> int | None:
    # prefer soma root: type==1 & parent==-1
    soma = [i for i in range(len(arr)) if int(arr[i]["type"]) == 1 and int(arr[i]["parent"]) == -1]
    if soma:
        return soma[0]
    # else, any parent==-1
    roots = [i for i in range(len(arr)) if int(arr[i]["parent"]) == -1]
    return roots[0] if roots else None

def _tips(kids: list[list[int]]) -> list[int]:
    return [i for i in range(len(kids)) if len(kids[i]) == 0]

def _bifurcations(kids: list[list[int]]) -> list[int]:
    return [i for i in range(len(kids)) if len(kids[i]) >= 2]

def _tree_depth(arr) -> int:
    r = _root_index(arr)
    if r is None:
        return 0
    par = _parent_index(arr); kids = _children_lists(par)
    # BFS measuring depth in #edges from root
    depth = 0
    frontier = [r]
    seen = {r}
    while frontier:
        nxt = []
        for u in frontier:
            for v in kids[u]:
                if v not in seen:
                    seen.add(v); nxt.append(v)
        if nxt:
            depth += 1
        frontier = nxt
    return depth

def _total_length(arr, type_filter: int | None = None) -> float:
    par = _parent_index(arr)
    total = 0.0
    for i in range(len(arr)):
        p = par[i]
        if p < 0:
            continue
        if type_filter is not None and int(arr[i]["type"]) != type_filter:
            continue
        total += _euclid3(arr[i], arr[p])
    return float(total)

def _max_euclid_extent(arr) -> float:
    r = _root_index(arr)
    if r is None:
        return 0.0
    par = _parent_index(arr); kids = _children_lists(par)
    tips = _tips(kids)
    if not tips:
        return 0.0
    soma = arr[r]
    dmax = 0.0
    for t in tips:
        d = _euclid3(arr[t], soma)
        if d > dmax:
            dmax = d
    return float(dmax)

def _avg_radius(arr, include_types: set[int]) -> float:
    radii = [float(arr[i]["radius"]) for i in range(len(arr))
             if int(arr[i]["type"]) in include_types and float(arr[i]["radius"]) > 0.0]
    return float(np.mean(radii)) if radii else 0.0

# -------------------------------------------------------------------
# Public: compute metrics for a single SWC file
# -------------------------------------------------------------------
def metrics_for_swc(swc_path: str) -> Dict[str, float] | None:
    if not os.path.isfile(swc_path):
        return None
    arr = read_swc_file(swc_path)
    if len(arr) == 0:
        return None

    par = _parent_index(arr); kids = _children_lists(par)

    m: Dict[str, float] = {}
    m["n_nodes"]             = float(len(arr))
    m["n_bifurcations"]      = float(len(_bifurcations(kids)))
    m["n_tips"]              = float(len(_tips(kids)))
    m["total_length_all"]    = _total_length(arr, None)
    m["total_length_axon"]   = _total_length(arr, 2)
    m["total_length_basal"]  = _total_length(arr, 3)
    m["total_length_apical"] = _total_length(arr, 4)
    m["max_euclid_extent"]   = _max_euclid_extent(arr)
    m["avg_radius_non_soma"] = _avg_radius(arr, {0,2,3,4,5,6,7,8,9,10,11})
    m["avg_radius_soma"]     = _avg_radius(arr, {1})
    m["tree_depth"]          = float(_tree_depth(arr))
    return m

# -------------------------------------------------------------------
# Collect metrics for all SWCs listed in Excel (per type)
# Returns:
#   by_metric_values: {metric -> {type -> [values]}}
#   by_metric_points: {metric -> {type -> [(swc_id, value), ...]}}
#   ids_by_type:      {type -> [swc_id, ...]}
# -------------------------------------------------------------------
def collect_metrics_from_table(xlsx_path: str,
                               swc_dir: str,
                               *,
                               verbose: bool = False,
                               limit_per_type: int | None = None
                               ) -> Tuple[Dict[str, Dict[str, List[float]]],
                                          Dict[str, Dict[str, List[Tuple[str, float]]]],
                                          Dict[str, List[str]]]:
    ids_by_type = read_celltype_table(xlsx_path)

    metrics_keys = [
        "n_nodes","n_bifurcations","n_tips","total_length_all",
        "total_length_axon","total_length_basal","total_length_apical",
        "max_euclid_extent","avg_radius_non_soma","avg_radius_soma","tree_depth",
    ]
    by_metric_values: Dict[str, Dict[str, List[float]]] = {k: {t: [] for t in NEURON_TYPES} for k in metrics_keys}
    by_metric_points: Dict[str, Dict[str, List[Tuple[str, float]]]] = {k: {t: [] for t in NEURON_TYPES} for k in metrics_keys}

    for t in NEURON_TYPES:
        ids = ids_by_type.get(t, [])
        if limit_per_type is not None:
            ids = ids[:limit_per_type]

        if verbose:
            print(f"[{t}] {len(ids)} file(s) to process...", flush=True)

        for i, swc_id in enumerate(ids, 1):
            swc_path = os.path.join(swc_dir, f"{swc_id}.swc")
            try:
                m = metrics_for_swc(swc_path)
            except Exception as e:
                if verbose:
                    print(f"  - {swc_id}: ERROR {type(e).__name__}: {e}", flush=True)
                continue

            if m is None:
                if verbose:
                    print(f"  - {swc_id}: missing/empty", flush=True)
                continue

            for k in metrics_keys:
                val = float(m.get(k, np.nan))
                if np.isnan(val):
                    continue
                by_metric_values[k][t].append(val)
                by_metric_points[k][t].append((swc_id, val))

            if verbose and (i % 10 == 0 or i == len(ids)):
                print(f"  - processed {i}/{len(ids)}", flush=True)

    return by_metric_values, by_metric_points, ids_by_type

