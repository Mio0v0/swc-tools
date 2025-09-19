# src/swctools/stats.py
import numpy as np
from typing import Dict, Tuple, List
from .graph import id_to_index_map

def per_type_stats(arr: np.ndarray, p_low: float, p_high: float, min_count: int
                   ) -> Tuple[Dict[int,float], Dict[int,float], Dict[int,float]]:
    types = arr["type"].astype(int)
    mean_by_type, pL_by_type, pH_by_type = {}, {}, {}
    all_pos = np.array([float(r) for r in arr["radius"] if r > 0], dtype=float)
    g_pL = float(np.percentile(all_pos, p_low)) if all_pos.size else 0.0
    g_pH = float(np.percentile(all_pos, p_high)) if all_pos.size else 0.0
    g_mean = float(np.mean(all_pos)) if all_pos.size else 0.0
    for t in np.unique(types):
        vals = np.array([float(arr[i]["radius"]) for i in range(len(arr))
                         if types[i]==t and arr[i]["radius"]>0], dtype=float)
        if vals.size >= max(3, min_count):
            mean_by_type[t] = float(np.mean(vals))
            pL_by_type[t]   = float(np.percentile(vals, p_low))
            pH_by_type[t]   = float(np.percentile(vals, p_high))
        else:
            mean_by_type[t], pL_by_type[t], pH_by_type[t] = g_mean, g_pL, g_pH
    return mean_by_type, pL_by_type, pH_by_type

def euclid3(ax, ay, az, bx, by, bz) -> float:
    dx=ax-bx; dy=ay-by; dz=az-bz
    return float(np.sqrt(dx*dx+dy*dy+dz*dz))

# src/swctools/stats.py
def edge_lengths_in_component(arr: np.ndarray, comp_indices: List[int]) -> np.ndarray:
    id2idx = id_to_index_map(arr)
    comp = set(comp_indices)
    L: List[float] = []
    for i in comp_indices:
        p = int(arr[i]["parent"])
        if p == -1:
            continue
        j = id2idx.get(p)
        if j is None or j not in comp:
            continue
        L.append(
            euclid3(
                arr[i]["x"], arr[i]["y"], arr[i]["z"],
                arr[j]["x"], arr[j]["y"], arr[j]["z"]
            )
        )
    return np.array(L, dtype=float)


def robust_threshold_T(edge_lengths: np.ndarray) -> float:
    if edge_lengths.size == 0: return 0.0
    d_med = float(np.median(edge_lengths))
    mad   = float(np.median(np.abs(edge_lengths - d_med)))
    d_p95 = float(np.percentile(edge_lengths, 95))
    return max(1.25*d_p95, d_med + 3*mad)
