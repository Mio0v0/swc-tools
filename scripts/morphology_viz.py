import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Tuple

# --- Input / Output ---
swc_dir = Path(r"D:\Desktop\Projectome SWC Files")
out_dir = Path(r"D:\Desktop\SWC Output\Plot")
out_dir.mkdir(parents=True, exist_ok=True)

# --- List of SWC files ---
swc_list = [
    "202225_062.swc"
]

# --------------------------- category + legend ---------------------------

def _category_for_type(t: int) -> str:
    if t == 0: return "undefined"
    if t == 1: return "soma"
    if t == 2: return "axon"
    if t == 3: return "basal dendrite"
    if t == 4: return "apical dendrite"
    return "custom"

# --- consistent colors ---
DEFAULT_COLORS = {
    "undefined": "gray",
    "soma": "green",
    "axon": "blue",
    "basal dendrite": "red",
    "apical dendrite": "pink",
    "custom": "orange",
}

# --------------------------- tree utils for dendrogram ---------------------------

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
            continue
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
    best = roots[0]; best_size = -1
    for r in roots:
        sz = _subtree_size(kids, r)
        if sz > best_size:
            best, best_size = r, sz
    return best

def _cumlens_from_root(arr: np.ndarray, kids: List[List[int]], root: int) -> List[float]:
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
    N = len(kids)
    y = [0.0] * N
    leaves_order: List[int] = []
    cursor = 0
    stack: List[Tuple[int, int]] = [(root, 0)]
    onstack = set([root])

    while stack:
        u, state = stack.pop()
        if state == 0:
            if not kids[u]:
                y[u] = float(cursor)
                leaves_order.append(u)
                cursor += 1
            else:
                stack.append((u, 1))
                for v in reversed(kids[u]):
                    if v == u or v in onstack:
                        continue
                    stack.append((v, 0))
                    onstack.add(v)
        else:
            ch = kids[u]
            if ch:
                y[u] = float(sum(y[v] for v in ch) / len(ch))
            onstack.discard(u)
    if not leaves_order:
        leaves_order = [root]
        y[root] = 0.0
    return y, leaves_order

# ------------------------- dendrogram drawing ---------------------------

def _draw_dendrogram_matplotlib(
    arr: np.ndarray,
    ax: plt.Axes,
    colors: Dict[str, str],
    line_width: float = 1.5,
    connector_color: str = "#444444",
) -> List[str]:
    if len(arr) == 0:
        return []
    kids = _children_lists(arr)
    root = _pick_root(arr, kids)
    cum = _cumlens_from_root(arr, kids, root)
    y, _ = _layout_y_positions_iterative(kids, root)

    used_categories: List[str] = []
    used_set = set()

    # vertical connectors
    for u in range(len(arr)):
        if not kids[u]:
            continue
        ys = [y[v] for v in kids[u]]
        x = cum[u]
        ax.plot([x, x], [min(ys), max(ys)], color=connector_color,
                linewidth=line_width)

    # horizontal edges
    for u in range(len(arr)):
        for v in kids[u]:
            xv0, xv1 = cum[u], cum[v]
            yv = y[v]
            cat = _category_for_type(int(arr[v]["type"]))
            col = colors.get(cat, "gray")
            ax.plot([xv0, xv1], [yv, yv], color=col, linewidth=line_width)
            if cat not in used_set:
                used_set.add(cat)
                used_categories.append(cat)

    ax.set_ylim(min(y)-0.5, max(y)+0.5)
    xmax = max(cum) if cum else 0.0
    ax.set_xlim(0.0, xmax*1.02 if xmax > 0 else 1.0)
    ax.tick_params(axis="y", left=False, labelleft=False)
    for side in ("left","right","top"):
        ax.spines[side].set_visible(False)

    return used_categories

# ------------------------- main plotting ---------------------------

def plot_all(file_path: Path, out_dir: Path):
    cols = ["id", "type", "x", "y", "z", "radius", "parent"]
    df = pd.read_csv(file_path, delim_whitespace=True, comment="#", names=cols)
    arr = df.to_records(index=False)
    fname = file_path.stem

    # --- 2D plot ---
    plt.figure(figsize=(12,10))
    for _, row in df.iterrows():
        if row["parent"] == -1: continue
        parent = df.loc[df["id"] == row["parent"]]
        if parent.empty: continue
        cat = _category_for_type(int(row["type"]))
        plt.plot([row["x"], parent["x"].values[0]],
                 [row["y"], parent["y"].values[0]],
                 color=DEFAULT_COLORS.get(cat,"gray"), linewidth=1.5)
    plt.title(f"2D Morphology", fontsize=20)
    plt.xlabel("X (µm)", fontsize=20)
    plt.ylabel("Y (µm)", fontsize=20)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)

    plt.savefig(out_dir / f"{fname}_2D.svg", format="svg")
    plt.close()

    # --- 3D plot ---
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection="3d")
    for _, row in df.iterrows():
        if row["parent"] == -1: continue
        parent = df.loc[df["id"] == row["parent"]]
        if parent.empty: continue
        cat = _category_for_type(int(row["type"]))
        ax.plot([row["x"], parent["x"].values[0]],
                [row["y"], parent["y"].values[0]],
                [row["z"], parent["z"].values[0]],
                color=DEFAULT_COLORS.get(cat,"gray"), linewidth=1.2)
    ax.set_title(f"3D Morphology", fontsize=20)
    ax.set_xlabel("X (µm)", fontsize=20, labelpad=15)
    ax.set_ylabel("Y (µm)", fontsize=20, labelpad=15)
    ax.set_zlabel("Z (µm)", fontsize=20, labelpad=15)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="z", labelsize=14)
    ax.zaxis.set_tick_params(pad=10)
    plt.savefig(out_dir / f"{fname}_3D.svg", format="svg")
    plt.close()

    # --- Dendrogram ---
    fig, ax = plt.subplots(figsize=(18, 10))
    used_cats = _draw_dendrogram_matplotlib(arr, ax, DEFAULT_COLORS, line_width=1.2)

    ax.set_title(f"Dendrogram", fontsize=22)
    ax.set_xlabel("Path length (µm)", fontsize=20)
    ax.tick_params(axis="x", labelsize=14)

    # Ensure soma is included in legend if present in file
    if 1 in df["type"].values and "soma" not in used_cats:
        used_cats.insert(0, "soma")

    # Map text for legend
    def legend_label(cat: str) -> str:
        if cat in ("basal dendrite", "apical dendrite"):
            return "dendrite"
        return cat

    # Build legend handles with remapped labels
    legend_handles = [
        plt.Line2D([0], [0], color=DEFAULT_COLORS[c], lw=3, label=legend_label(c))
        for c in used_cats
    ]

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=14)

    plt.savefig(out_dir / f"{fname}_Dendrogram.svg", format="svg")
    plt.close()


# --- Loop ---
for swc_file in swc_list:
    fpath = swc_dir / swc_file
    if fpath.exists():
        print(f"Processing {fpath.name}...")
        plot_all(fpath, out_dir)
    else:
        print(f"⚠️ File not found: {fpath}")

print("✅ Done! 2D, 3D, and Dendrogram plots saved in:", out_dir)
