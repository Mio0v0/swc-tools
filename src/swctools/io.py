# src/swctools/io.py
import os, numpy as np
from typing import Dict

def _safe_int(x, default=-1):
    try:
        s = str(x).strip().lower()
        if s in ("", "na", "nan"): return default
        return int(float(x))
    except Exception:
        return default

def _safe_float(x, default=0.0):
    try:
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().lower()
        if s in ("", "na", "nan"): return default
        return float(x)
    except Exception:
        return default

def fmt_radius(x: float) -> str:
    return f"{float(x):.13e}"

def read_swc_file(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            p = s.split()
            if len(p) < 7: continue
            rid=_safe_int(p[0]); rtype=_safe_int(p[1])
            x_str,y_str,z_str = p[2],p[3],p[4]
            rrad_str = p[5]; rpar=_safe_int(p[6])
            rx=_safe_float(x_str); ry=_safe_float(y_str); rz=_safe_float(z_str)
            rrad=_safe_float(rrad_str)
            rows.append((rid, rtype, rx, ry, rz, rpar, x_str, y_str, z_str, rrad, rrad_str))
    dtype = np.dtype([
        ("id","i8"),("type","i4"),("x","f8"),("y","f8"),("z","f8"),("parent","i8"),
        ("x_str","U128"),("y_str","U128"),("z_str","U128"),("radius","f8"),("radius_str","U128"),
    ])
    return np.array(rows, dtype=dtype) if rows else np.empty((0,), dtype=dtype)

def write_swc_file(arr: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ID type x y z radius parent\n")
        for n in arr:
            f.write(f"{int(n['id'])} {int(n['type'])} {n['x_str']} {n['y_str']} {n['z_str']} {n['radius_str']} {int(n['parent'])}\n")
