# src/swctools/dendrograms.py
from __future__ import annotations
import os, glob, csv
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .config import load_section
from .io import read_swc_file

# --------------------------- coloring -----------------------------

DEFAULT_COLORS = {
    "undefined":      "#9467bd",
    "soma":           "#808080",
    "axon":           "#d62728",
    "basal_dendrite": "#2ca02c",
    "apical_dendrite":"#1f77b4",
    "custom":         "#ff7f00",
}
CATEGORY_ORDER = ["undefined", "soma", "axon", "basal_dendrite", "apical_dendrite", "custom"]

def _colors_from_cfg(c: Dict[str, str] | None) -> Dict[str, str]:
    out = DEFAULT_COLORS.copy()
    if c:
        for k in out.keys():
            if k in c and c[k]:
                out[k] = c[k]
    return out

def _category_for_type(t: int) -> str:
    if t == 0: return "undefined"
    if t == 1: return "soma"
    if t == 2: return "axon"
    if t == 3: return "basal_dendrite"
    if t == 4: return "apical_dendrite"
    return "custom"

def _legend_label(cat: str) -> str:
    return {
        "undefined":       "undefined (0)",
        "soma":            "soma (1)",
        "axon":            "axon (2)",
        "basal_dendrite":  "basal dendrite (3)",
        "apical_dendrite": "apical dendrite (4)",
        "custom":          "custom (≥5)",
    }[cat]

# --------------------------- tree utils ---------------------------

def _build_id2idx(arr: np.ndarray) -> Dict[int, int]:
    return {int(n["id"]): i for i, n in enumerate(arr)}

def _children_lists(arr: np.ndarray) -> List[List[int]]:
    id2idx = _build_id2idx(arr)
    N = len(arr)
    kids = [[] for _ in range(N)]
    for i in range(N):
        p = int(arr[i]["parent"])
        if p == -1:
            continue
        j = id2idx.get(p)
        if j is None or j == i:
            continue  # skip invalid/self loops
        kids[j].append(i)
    return kids

def _edge_length(arr: np.ndarray, i: int, j: int) -> float:
    dx = float(arr[i]["x"]) - float(arr[j]["x"])
    dy = float(arr[i]["y"]) - float(arr[j]["y"])
    dz = float(arr[i]["z"]) - float(arr[j]["z"])
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))

def _subtree_size(kids: List[List[int]], root: int) -> int:
    count = 0
    stack = [root]
    seen = set()
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        count += 1
        stack.extend(kids[u])
    return count

def _pick_root(arr: np.ndarray, kids: List[List[int]]) -> int:
    roots = [i for i in range(len(arr)) if int(arr[i]["parent"]) == -1]
    if not roots:
        return 0
    soma_roots = [i for i in roots if int(arr[i]["type"]) == 1]
    if soma_roots:
        return soma_roots[0]
    # else choose the one with largest subtree (iterative, no recursion)
    best = roots[0]; best_size = -1
    for r in roots:
        sz = _subtree_size(kids, r)
        if sz > best_size:
            best, best_size = r, sz
    return best

def _cumlens_from_root(arr: np.ndarray, kids: List[List[int]], root: int) -> List[float]:
    """Iterative path lengths from root to each node (sum of Euclidean edge lengths)."""
    cum = [0.0] * len(arr)
    stack = [root]
    seen = set([root])
    while stack:
        u = stack.pop()
        for v in kids[u]:
            if v not in seen:
                cum[v] = cum[u] + _edge_length(arr, v, u)
                seen.add(v)
                stack.append(v)
    return cum

def _layout_y_positions_iterative(kids: List[List[int]], root: int) -> Tuple[List[float], List[int]]:
    """
    Iterative DFS to assign y:
      - Leaves get increasing integers left->right
      - Internal y = mean(child y)
    Returns (y_positions, leaves_in_order).
    """
    N = len(kids)
    y = [0.0] * N
    leaves_order: List[int] = []
    cursor = 0

    # state 0 = enter, 1 = exit (post-order)
    stack: List[Tuple[int, int]] = [(root, 0)]
    onstack = set([root])
    visited = set()

    while stack:
        u, state = stack.pop()
        if state == 0:
            visited.add(u)
            if not kids[u]:
                # leaf
                y[u] = float(cursor)
                leaves_order.append(u)
                cursor += 1
                # no exit state needed for leaf
            else:
                # exit later after children processed
                stack.append((u, 1))
                # push children in reverse for stable left->right
                for v in reversed(kids[u]):
                    if v == u or v in onstack:
                        # guard against cycles/back-edges: treat as no-child
                        continue
                    stack.append((v, 0))
                    onstack.add(v)
        else:
            # exit: average child y
            ch = kids[u]
            if ch:
                y[u] = float(sum(y[v] for v in ch) / len(ch))
            onstack.discard(u)

    # In case graph had isolated nodes or cycles we skipped, normalize range
    if not leaves_order:
        leaves_order = [root]
        y[root] = 0.0

    return y, leaves_order

# ------------------------- drawing core ---------------------------

def _draw_dendrogram_matplotlib(
    arr: np.ndarray,
    ax: plt.Axes,
    colors: Dict[str, str],
    line_width: float = 1.4,
    connector_color: str = "#444444",
) -> List[str]:
    """
    Draw right-angle dendrogram for chosen component:
      - vertical gray connectors at branch x
      - horizontal edges colored by child category
    Returns list of categories used (for legend).
    """
    if len(arr) == 0:
        return []

    kids = _children_lists(arr)
    root = _pick_root(arr, kids)
    cum = _cumlens_from_root(arr, kids, root)
    y, _ = _layout_y_positions_iterative(kids, root)

    used_categories: List[str] = []
    used_set = set()

    # vertical connectors (branch unions)
    for u in range(len(arr)):
        if not kids[u]:
            continue
        ys = [y[v] for v in kids[u]]
        x = cum[u]
        ax.plot([x, x], [min(ys), max(ys)], color=connector_color, linewidth=line_width, solid_capstyle="butt")

    # horizontal edges (child colored by its SWC type category)
    for u in range(len(arr)):
        for v in kids[u]:
            xv0, xv1 = cum[u], cum[v]
            yv = y[v]
            cat = _category_for_type(int(arr[v]["type"]))
            col = colors.get(cat, DEFAULT_COLORS[cat])
            ax.plot([xv0, xv1], [yv, yv], color=col, linewidth=line_width, solid_capstyle="butt")
            if cat not in used_set:
                used_set.add(cat)
                used_categories.append(cat)

    # style
    ymin, ymax = min(y), max(y)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    xmax = max(cum) if cum else 0.0
    ax.set_xlim(0.0, xmax * 1.02 if xmax > 0 else 1.0)
    ax.grid(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    return used_categories

# --------------------------- per-file/dir/CLI ---------------------

def process_file(in_path: str, out_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = os.path.basename(in_path)
    stem, _ = os.path.splitext(base)

    io   = cfg["io"]
    draw = cfg.get("draw", {})
    colors_cfg = _colors_from_cfg(cfg.get("colors"))
    line_width = float(draw.get("line_width", 1.4))
    width      = float(draw.get("width", 10.0))
    height     = float(draw.get("height", 7.0))
    dpi        = int(draw.get("dpi", 300))

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{stem}_dendrogram.svg")  # always SVG

    arr = read_swc_file(in_path)

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    try:
        used_cats = _draw_dendrogram_matplotlib(arr, ax, colors_cfg, line_width=line_width)

        # legend (only categories that appeared)
        handles = [Line2D([0],[0], color=colors_cfg[c], lw=line_width, label=_legend_label(c))
                   for c in CATEGORY_ORDER if c in used_cats]
        if handles:
            ax.legend(handles=handles, loc="best", frameon=False)

        ax.set_title(f"Dendrogram: {stem}", pad=10)
        ax.set_xlabel("Path length")

        fig.tight_layout()
        fig.savefig(out_file, format="svg", facecolor="white", edgecolor="white")
        ok, err = True, ""
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    finally:
        plt.close(fig)

    return {"input_file": base, "ok": ok, "error": err, "output": out_file if ok else ""}

def process_dir(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    io = cfg["io"]
    pattern = "**/*.swc" if io.get("recursive", False) else io.get("pattern", "*.swc")
    files = sorted(glob.glob(os.path.join(io["input_dir"], pattern), recursive=io.get("recursive", False)))
    if not files:
        raise FileNotFoundError(f"No SWC files found under: {io['input_dir']} (pattern={pattern})")

    os.makedirs(io["output_dir"], exist_ok=True)
    results: List[Dict[str, Any]] = []
    for i, f in enumerate(files, 1):
        rec = process_file(f, io["output_dir"], cfg)
        results.append(rec)
        if rec["ok"]:
            print(f"[{i}/{len(files)}] ✓ {rec['input_file']} -> {os.path.relpath(rec['output'], io['output_dir'])}")
        else:
            print(f"[{i}/{len(files)}] ⚠️ {rec['input_file']} -> {rec['error']}")

    # CSV log
    log_csv = io.get("log_csv", "dendrogram_errors.csv")
    log_path = os.path.join(io["output_dir"], log_csv)
    with open(log_path, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["input_file", "ok", "output", "error"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nLog written: {log_path}")
    return results

def cli_main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Draw Matplotlib dendrograms colored by SWC type (0..4 fixed, ≥5 custom) as SVG."
    )
    ap.add_argument("--config", required=True, help="Path to config.json")
    ap.add_argument("--section", default="dendrograms", help="Config section (default: dendrograms)")
    args = ap.parse_args()
    cfg = load_section(args.config, args.section)
    process_dir(cfg)
