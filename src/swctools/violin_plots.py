from __future__ import annotations
import os, csv
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_section
from .metrics import (
    collect_metrics_from_table,
    NEURON_TYPES,
    TYPE_COLORS,
)


METRIC_KEYS = [
    "n_nodes",
    "n_bifurcations",
    "n_tips",
    "total_length_all",
    "total_length_axon",
    "total_length_basal",
    "total_length_apical",
    "max_euclid_extent",
    "avg_radius_non_soma",
    "avg_radius_soma",
    "tree_depth",
]

# --------------------- small helpers ---------------------

def _nan_safe_array(vals: List[float]) -> np.ndarray:
    if not vals:
        return np.array([], dtype=float)
    a = np.array([float(v) for v in vals], dtype=float)
    return a[~np.isnan(a)]

def _describe(vals: List[float]) -> Dict[str, Any]:
    """Return basic stats for a vector; empty -> NaNs."""
    a = _nan_safe_array(vals)
    if a.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan,
                    q1=np.nan, median=np.nan, q3=np.nan, max=np.nan, iqr=np.nan)
    q1 = float(np.percentile(a, 25))
    q3 = float(np.percentile(a, 75))
    return dict(
        n=int(a.size),
        mean=float(np.mean(a)),
        std=float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        min=float(np.min(a)),
        q1=q1,
        median=float(np.median(a)),
        q3=q3,
        max=float(np.max(a)),
        iqr=float(q3 - q1),
    )

# --------------------- plotting one metric ---------------------

def _draw_one_violin(metric: str,
                     by_type_values: Dict[str, List[float]],
                     by_type_points: Dict[str, List[Tuple[str, float]]],
                     out_svg: str,
                     width: float, height: float, dpi: int,
                     jitter: float, point_alpha: float,
                     point_edge: str, point_size: float,
                     sample: Tuple[str, str] | None,
                     mean_color: str = "#ff7f0e") -> Tuple[bool, str]:
    types = NEURON_TYPES
    positions = list(range(1, len(types) + 1))

    # Build dataset and skip entirely if all groups are empty
    data = [by_type_values.get(t, []) for t in types]
    if not any(len(v) for v in data):
        return False, "no data for this metric"

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    vp = ax.violinplot(
        data,
        positions=positions,
        showmeans=True,
        showmedians=True,
        widths=0.8,
    )

    # Color the violin "bodies"
    for body, t in zip(vp['bodies'], types):
        body.set_facecolor(TYPE_COLORS[t])
        body.set_edgecolor('black')
        body.set_alpha(0.55)

    # color the mean line differently
    if 'cmeans' in vp and vp['cmeans'] is not None:
        obj = vp['cmeans']
        try:
            obj.set_color(mean_color)
            obj.set_linewidth(2.0)
            obj.set_zorder(6)
        except AttributeError:
            for o in obj:
                o.set_color(mean_color)
                o.set_linewidth(2.0)
                o.set_zorder(6)

    # Keep other violin artists black
    for k in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if k in vp and vp[k] is not None:
            obj = vp[k]
            try:
                obj.set_color('black')
            except AttributeError:
                try:
                    for o in obj:
                        o.set_color('black')
                except TypeError:
                    pass

    # Overlay all SWC points with jitter
    rng = np.random.default_rng(0)  # stable jitter
    for x, t in zip(positions, types):
        pts = by_type_points.get(t, [])
        if not pts:
            continue
        xs = x + rng.uniform(-jitter, jitter, size=len(pts))
        ys = [v for _, v in pts]
        ax.scatter(xs, ys,
                   s=point_size,
                   facecolor='white',
                   edgecolor=point_edge,
                   alpha=point_alpha,
                   linewidths=0.7,
                   zorder=5)

    # Highlight optional sample
    if sample is not None:
        sample_type, sample_id = sample
        if sample_type in types:
            pts = dict(by_type_points.get(sample_type, []))
            if sample_id in pts:
                xi = positions[types.index(sample_type)]
                yi = pts[sample_id]
                ax.scatter([xi], [yi], s=65, marker='o',
                           facecolor='yellow', edgecolor='black',
                           linewidths=1.0, zorder=6, label=f"sample {sample_id}")
                ax.legend(frameon=False, loc='best')

    ax.set_xticks(positions)
    ax.set_xticklabels(types)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_ylabel(metric)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    return True, ""

# --------------------- main driver ---------------------

def process_dir(cfg: Dict[str, Any]):
    io   = cfg["io"]
    plot = cfg.get("plot", {})
    dbg  = cfg.get("debug", {})

    xlsx    = io["xlsx"]
    swc_dir = io["swc_dir"]
    out_dir = io["out_dir"]
    stats_csv = io.get("stats_csv", "violin_summary_stats.csv")

    width  = float(plot.get("width", 8.5))
    height = float(plot.get("height", 4.5))
    dpi    = int(plot.get("dpi", 150))
    jitter = float(plot.get("jitter", 0.06))
    point_alpha = float(plot.get("point_alpha", 0.6))
    point_edge  = str(plot.get("point_edge", "#000000"))
    point_size  = float(plot.get("point_size", 18.0))
    mean_color = str(plot.get("mean_color", "#ff7f0e"))

    verbose = bool(dbg.get("verbose", True))
    limit_per_type = dbg.get("limit_per_type", None)

    # Compute metrics (with progress + limiting)
    by_metric_values, by_metric_points, ids_by_type = collect_metrics_from_table(
        xlsx, swc_dir, verbose=verbose, limit_per_type=limit_per_type
    )

    # Optional sample overlay
    sample_id = (cfg.get("sample") or {}).get("swc_id")
    sample_as = None
    if sample_id:
        for typ, ids in ids_by_type.items():
            if sample_id in ids:
                sample_as = (typ, sample_id)
                break
        if verbose:
            print(f"Sample: {sample_as}", flush=True)

    # --- plots ---
    results = []
    for metric in METRIC_KEYS:
        if metric not in by_metric_values:
            if verbose: print(f"⚠️ {metric}: no data key", flush=True)
            continue
        out_svg = os.path.join(out_dir, f"violin_{metric}.svg")
        ok, err = _draw_one_violin(
            metric,
            by_metric_values[metric],
            by_metric_points[metric],
            out_svg,
            width, height, dpi,
            jitter, point_alpha, point_edge, point_size,
            sample_as,
            mean_color=mean_color
        )
        results.append({"metric": metric, "output": out_svg, "ok": ok, "error": err})
        if ok:
            print(f"✓ {metric} -> {out_svg}", flush=True)
        else:
            print(f"⚠️ {metric} skipped: {err}", flush=True)

    # --- one summary CSV with per-type stats for each metric ---
    stats_path = os.path.join(out_dir, stats_csv)
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "metric", "type", "n",
            "mean", "std", "min", "q1", "median", "q3", "max", "iqr",
            "sample_id", "sample_value"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        # build a quick sample lookup: metric -> type -> {id: value}
        sample_lookup: Dict[str, Dict[str, Dict[str, float]]] = {}
        for metric in METRIC_KEYS:
            pts = by_metric_points.get(metric, {})
            sample_lookup[metric] = {t: dict(pts.get(t, [])) for t in NEURON_TYPES}

        for metric in METRIC_KEYS:
            vals_by_type = by_metric_values.get(metric, {})
            for t in NEURON_TYPES:
                vals = vals_by_type.get(t, [])
                ds = _describe(vals)
                s_id = sample_id if (sample_as and sample_as[0] == t) else ""
                s_val = ""
                if s_id:
                    s_val = sample_lookup.get(metric, {}).get(t, {}).get(s_id, "")
                w.writerow({
                    "metric": metric,
                    "type": t,
                    **ds,
                    "sample_id": s_id,
                    "sample_value": s_val
                })

    if verbose:
        print(f"Summary stats -> {stats_path}", flush=True)

    return results

def cli_main():
    import argparse
    ap = argparse.ArgumentParser(
        description="SVG violin plots per metric (five neuron types), overlay all SWCs as points, plus per-type summary stats."
    )
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--section", default="violin_metrics", help="Config section name")
    args = ap.parse_args()
    cfg = load_section(args.config, args.section)
    process_dir(cfg)
