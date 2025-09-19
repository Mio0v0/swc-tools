# src/swctools/graph.py
import numpy as np
from typing import Dict, List

def id_to_index_map(arr: np.ndarray) -> Dict[int, int]:
    return {int(n["id"]): i for i, n in enumerate(arr)}

def build_parent_index(arr: np.ndarray) -> np.ndarray:
    id2idx = id_to_index_map(arr)
    parent_idx = np.full(len(arr), -1, dtype=np.int64)
    for i in range(len(arr)):
        p = int(arr[i]["parent"])
        if p != -1:
            parent_idx[i] = id2idx.get(p, -1)
    return parent_idx

def build_children_lists(parent_idx: np.ndarray) -> List[List[int]]:
    kids = [[] for _ in range(len(parent_idx))]
    for i, p in enumerate(parent_idx):
        if p >= 0: kids[p].append(i)
    return kids

def build_adj(parent_idx: np.ndarray, children: List[List[int]]) -> List[List[int]]:
    adj = [[] for _ in range(len(parent_idx))]
    for i, p in enumerate(parent_idx):
        if p >= 0: adj[i].append(p)
        adj[i].extend(children[i])
    return adj

def precompute_neighbors(adj: List[List[int]], depth: int, max_n: int) -> List[List[int]]:
    if depth <= 1:
        return [nbrs[:max_n] for nbrs in adj]
    out = []
    for u in range(len(adj)):
        seen = {u}; acc = []
        for v in adj[u]:
            if v not in seen: seen.add(v); acc.append(v)
        for v in adj[u]:
            for w in adj[v]:
                if w not in seen:
                    seen.add(w); acc.append(w)
                    if len(acc) >= max_n: break
            if len(acc) >= max_n: break
        out.append(acc[:max_n])
    return out

# Orphan helpers reused by orphan cleaner
def build_undirected_adjacency(arr: np.ndarray) -> List[List[int]]:
    N = len(arr); adj=[[] for _ in range(N)]
    id2idx = id_to_index_map(arr)
    for i in range(N):
        p = int(arr[i]["parent"])
        if p == -1: continue
        j = id2idx.get(p);
        if j is None: continue
        adj[i].append(j); adj[j].append(i)
    return adj

def connected_components(arr: np.ndarray) -> List[List[int]]:
    N=len(arr); adj=build_undirected_adjacency(arr)
    seen=[False]*N; comps=[]
    for i in range(N):
        if seen[i]: continue
        stack=[i]; seen[i]=True; comp=[]
        while stack:
            u=stack.pop(); comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v]=True; stack.append(v)
        comps.append(sorted(comp))
    return comps

def build_children_map(arr: np.ndarray) -> Dict[int, List[int]]:
    id2idx = id_to_index_map(arr)
    kids = {i: [] for i in range(len(arr))}
    for i in range(len(arr)):
        p = int(arr[i]["parent"])
        if p == -1: continue
        j = id2idx.get(p)
        if j is not None: kids[j].append(i)
    return kids

def descendants_of(arr: np.ndarray, root_idx: int, kids_map: Dict[int, List[int]]) -> set:
    out=set(); stack=list(kids_map.get(root_idx, []))
    while stack:
        u=stack.pop()
        if u in out: continue
        out.add(u); stack.extend(kids_map.get(u, []))
    return out

def subtree_indices(arr: np.ndarray, root_idx: int) -> List[int]:
    kids_map = build_children_map(arr)
    return [root_idx] + sorted(list(descendants_of(arr, root_idx, kids_map)))
