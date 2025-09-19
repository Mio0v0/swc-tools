# -*- coding: utf-8 -*-
"""
Robust SWC → dendrogram batch plotter (no recursion; cycle-safe).

- Reads Excel mapping: swc_id -> (type, layer, color)
- Parses each SWC, rebuilds children AFTER reading all nodes
- Iterative postorder for y layout (no recursion)
- Iterative drawing (no recursion)
- Detects and breaks parent->child cycles
- Saves one PNG per SWC + metrics.csv

Requires: pandas, numpy, matplotlib, openpyxl
"""

import os, re, math, glob
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- USER PATHS ----------------------
excel_path = r"D:\Desktop\SUB_CellTypes.xlsx"
swc_dir    = r"D:\Desktop\Projectome SWC Files_cleaned"
outdir     = r"D:\Desktop\SWC_Dendrograms"
os.makedirs(outdir, exist_ok=True)

# ----------------- Excel mapping helpers ----------------
def parse_excel_mapping(xlsx_path):
    df = pd.read_excel(xlsx_path, header=None, engine="openpyxl")
    if df.shape[0] < 3:
        raise ValueError("Excel must have ≥3 rows: header, layer, IDs.")

    top = df.iloc[0].ffill()
    layers = df.iloc[1]
    ids_block = df.iloc[2:]

    def split_type_color(s):
        if not isinstance(s, str):
            return str(s), None
        m = re.match(r"\s*(.+?)\s*\((.+?)\)\s*$", s)
        return (m.group(1).strip(), m.group(2).strip()) if m else (s.strip(), None)

    color_map = {
        "Red":"red","Orange":"orange","Green":"green","Blue":"blue","Purple":"purple",
        "Magenta":"magenta","Cyan":"cyan","Black":"black","Gray":"gray","Grey":"gray",
    }

    mapping = {}
    for j in range(df.shape[1]):
        tname_raw = top.iloc[j]
        layer_val = layers.iloc[j]
        layer_str = "" if (isinstance(layer_val, float) and math.isnan(layer_val)) else str(layer_val).strip()
        tname, tcolor_name = split_type_color(tname_raw)
        tcolor = color_map.get(tcolor_name) if tcolor_name else None

        ids = ids_block.iloc[:, j].dropna().astype(str).str.strip()
        for sid in ids:
            if sid and sid.lower() != "nan":
                mapping[sid] = {
                    "type_name": tname, "layer": layer_str,
                    "color": tcolor, "color_name": tcolor_name
                }
    return mapping

id_meta = parse_excel_mapping(excel_path)

# ---------------------- SWC parsing ---------------------
def load_swc_build_children(path):
    """
    Returns:
      nodes: dict[id] = {type,x,y,z,r,parent}
      children: dict[parent] -> [child,...] (only valid parents)
      roots: list of ids with parent == -1 OR parent missing (fixed)
      orphan_count: # of nodes whose parent was missing and got promoted to root
    """
    nodes = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"): continue
            parts = s.split()
            if len(parts) < 7: continue
            nid = int(float(parts[0])); p = int(float(parts[6]))
            nodes[nid] = {
                "type": int(float(parts[1])),
                "x": float(parts[2]), "y": float(parts[3]), "z": float(parts[4]),
                "r": float(parts[5]), "parent": p
            }

    # rebuild children only for valid parents
    children = defaultdict(list)
    orphan_count = 0
    for nid, d in nodes.items():
        p = d["parent"]
        if p == -1 or p not in nodes:
            if p not in (-1,):
                orphan_count += 1
            d["parent"] = -1
        else:
            children[p].append(nid)
    # sort for stability
    for k in children:
        children[k].sort()

    roots = [nid for nid, d in nodes.items() if d["parent"] == -1]
    return nodes, children, roots, orphan_count

def seg_len(nodes, child_id):
    p = nodes[child_id]["parent"]
    if p == -1: return 0.0
    cx, cy, cz = nodes[child_id]["x"], nodes[child_id]["y"], nodes[child_id]["z"]
    px, py, pz = nodes[p]["x"], nodes[p]["y"], nodes[p]["z"]
    return math.dist((cx,cy,cz), (px,py,pz))

def cumulative_x(nodes, children, roots):
    """
    Iterative cumulative path length from soma; supports multiple roots via "__ROOT__".
    """
    vchildren = {k: list(v) for k, v in children.items()}
    if len(roots) == 0:
        raise ValueError("No root (-1 parent) after rebuild.")
    if len(roots) == 1:
        root_id = roots[0]
        start = [root_id]
    else:
        root_id = "__ROOT__"
        vchildren[root_id] = list(roots)
        start = [root_id]

    xdist = {}
    dq = deque(start)
    for r in start: xdist[r] = 0.0

    while dq:
        u = dq.popleft()
        for v in vchildren.get(u, []):
            xdist[v] = 0.0 if u == "__ROOT__" else (xdist[u] + seg_len(nodes, v))
            dq.append(v)
    return xdist, root_id, vchildren

# ----------------- Cycle detection / breaking -----------
def break_cycles(vchildren):
    """
    Remove back-edges detected via iterative DFS (state: 0=unseen,1=visiting,2=done).
    Returns number of removed edges.
    """
    # Collect all nodes
    nodeset = set(vchildren.keys())
    for kids in vchildren.values():
        nodeset.update(kids)

    state = {n: 0 for n in nodeset}
    removed = 0

    for start in list(nodeset):
        if state.get(start,0) != 0: continue
        stack = [(start, iter(vchildren.get(start, [])))]
        state[start] = 1
        while stack:
            u, it = stack[-1]
            try:
                w = next(it)
                st = state.get(w, 0)
                if st == 0:
                    state[w] = 1
                    stack.append((w, iter(vchildren.get(w, []))))
                elif st == 1:
                    # back-edge u->w, remove w from u's children
                    lst = vchildren.get(u, [])
                    if w in lst:
                        lst2 = [c for c in lst if c != w]
                        vchildren[u] = lst2
                        removed += 1
                    # continue without pushing
                else:
                    # already done
                    continue
            except StopIteration:
                state[u] = 2
                stack.pop()
    return removed

# ---------------------- Y assignment --------------------
def assign_y_iterative(vchildren, root_id):
    """
    Postorder layout without recursion:
      - leaves get consecutive integers
      - internal nodes get mean of child y's
    """
    y = {}
    next_y = 0

    # Postorder using explicit stack with 'processed' flag
    stack = [(root_id, False)]
    visited = set()
    while stack:
        u, processed = stack.pop()
        if not processed:
            stack.append((u, True))
            for k in vchildren.get(u, []):
                if (u, k) in visited:  # already queued this edge
                    continue
                visited.add((u, k))
                stack.append((k, False))
        else:
            kids = vchildren.get(u, [])
            if not kids:
                y[u] = float(next_y); next_y += 1
            else:
                y[u] = float(np.mean([y[k] for k in kids if k in y])) if any(k in y for k in kids) else float(next_y)

    return y

# ---------------------- Drawing -------------------------
def draw_dendrogram_iterative(xdist, ycoord, vchildren, root_id, color, title, outpath, dpi=220):
    fig, ax = plt.subplots(figsize=(10, 6))

    # We need to draw vertical at parent x spanning children y's, then horizontals
    stack = [root_id]
    seen = set()
    while stack:
        u = stack.pop()
        if u in seen: continue
        seen.add(u)
        kids = vchildren.get(u, [])
        if kids:
            x_parent = xdist[u]
            ys = [ycoord[k] for k in kids if k in ycoord]
            if ys:
                ax.plot([x_parent, x_parent], [min(ys), max(ys)], lw=1.0, color=color)
            for k in kids:
                if k in xdist and k in ycoord:
                    ax.plot([x_parent, xdist[k]], [ycoord[k], ycoord[k]], lw=1.0, color=color)
                stack.append(k)

    # mark root
    if root_id in xdist and root_id in ycoord:
        ax.scatter([xdist[root_id]], [ycoord[root_id]], s=10, color=color, zorder=3)

    ax.set_xlabel("Path length from soma")
    ax.set_ylabel("Branch tips")
    if title: ax.set_title(title)
    ax.spines[['top','right']].set_visible(False)

    if len(ycoord):
        ax.set_ylim(min(ycoord.values()) - 1, max(ycoord.values()) + 1)
    if len(xdist):
        xmin = min(xdist.values()); xmax = max(xdist.values())
        ax.set_xlim(max(0.0, xmin - 0.05*(xmax - xmin + 1e-9)), xmax + 0.05*(xmax - xmin + 1e-9))
    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)

# --------------------- Summaries ------------------------
def summarize_tree(nodes, vchildren):
    n_nodes = len(nodes)
    all_parents = set(vchildren.keys())
    all_children = set(c for lst in vchildren.values() for c in lst)
    leaves = [u for u in nodes if u not in vchildren or len(vchildren.get(u, [])) == 0]
    bifurc = [u for u in all_parents if isinstance(u, int) and len(vchildren.get(u, [])) >= 2]
    total_len = sum(seg_len(nodes, u) for u in nodes if nodes[u]["parent"] != -1)
    return n_nodes, len(leaves), len(bifurc), total_len

# ------------------------ Main --------------------------
def main():
    meta = id_meta
    swc_files = sorted(glob.glob(os.path.join(swc_dir, "*.swc"))) + \
                sorted(glob.glob(os.path.join(swc_dir, "*.SWC")))
    if not swc_files:
        print(f"No .swc files found in {swc_dir}")
        return

    rows = []
    for swc_path in swc_files:
        base = os.path.splitext(os.path.basename(swc_path))[0]
        m = meta.get(base, {})
        type_name = m.get("type_name", "UnknownType")
        layer = m.get("layer", "")
        color = m.get("color", "black")
        color_name = m.get("color_name", "")

        try:
            nodes, children, roots, orphan_ct = load_swc_build_children(swc_path)
            xdist, root_id, vchildren = cumulative_x(nodes, children, roots)
            removed_edges = break_cycles(vchildren)  # prevent infinite loops
            ycoord = assign_y_iterative(vchildren, root_id)

            title = f"{base} – {type_name}"
            if layer: title += f" ({layer})"
            if color_name: title += f" [{color_name}]"

            safe_t = re.sub(r"[^A-Za-z0-9_.-]+","_", type_name)
            safe_l = re.sub(r"[^A-Za-z0-9_.-]+","_", layer)
            outname = f"{base}__{safe_t}__{safe_l}_dendrogram.png"
            outpath = os.path.join(outdir, outname)

            draw_dendrogram_iterative(xdist, ycoord, vchildren, root_id, color, title, outpath)

            n_nodes, n_leaves, n_bifurc, total_len = summarize_tree(nodes, vchildren)
            rows.append({
                "swc_id": base, "type": type_name, "layer": layer, "color": color_name,
                "nodes": n_nodes, "leaves": n_leaves, "bifurcations_(>=2_children)": n_bifurc,
                "total_path_length": total_len, "orphaned_parents_promoted_to_root": orphan_ct,
                "cycle_edges_removed": removed_edges, "png_path": outpath
            })
            print(f"✓ Saved {outpath} | orphans:{orphan_ct} cycles_broken:{removed_edges}")

        except Exception as e:
            print(f"✗ Failed for {base}: {e}")
            rows.append({
                "swc_id": base, "type": type_name, "layer": layer, "color": color_name,
                "nodes": "", "leaves": "", "bifurcations_(>=2_children)": "",
                "total_path_length": "", "orphaned_parents_promoted_to_root": "",
                "cycle_edges_removed": "", "png_path": "", "error": str(e)
            })

    if rows:
        metrics_csv = os.path.join(outdir, "metrics.csv")
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)
        print(f"\nMetrics written to: {metrics_csv}")

if __name__ == "__main__":
    main()
