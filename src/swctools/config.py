# src/swctools/config.py
import os, json, copy

DEFAULTS = {
  "radius_qc": {
    "io": {"input_dir":"", "output_dir":"", "log_csv":"radius_changes_percentile.csv",
           "log_summary":"files_changed_summary.csv", "pattern":"*.swc", "recursive":False},
    "compute": {"workers": 4},
    "thresholds": {"p_low":2.5, "p_high":97.5, "min_per_type_count":20},
    "neighbors": {"depth":1, "neighbor_max":16},
    "policy": {"skip_soma_changes":True, "eps_min_radius":1e-9}
  },
  "orphan_clean": {
    "io": {"input_dir":"", "output_dir":"", "log_csv":"swc_changes.csv",
           "pattern":"*.swc", "recursive":False},
    "reconnect": {"shift_accept_mad": 5.0}
  },
  "dendrograms": {
    "io": {"input_dir": "", "output_dir": "", "pattern": "*.swc", "recursive": False,
           "log_csv": "dendrogram_errors.csv"},
    "draw": {"show_diameters": True, "width": 10.0, "height": 7.0, "dpi": 300, "format": "svg"},
    "colors": {
      "undefined":      "#9467bd",
      "soma":           "#808080",
      "axon":           "#d62728",
      "basal_dendrite": "#2ca02c",
      "apical_dendrite":"#1f77b4",
      "custom":         "#ff7f00"   # used for any type >= 5
    },
    "min_nodes_per_type": 2
  },
  "violin_metrics": {
    "io": { "xlsx": "", "swc_dir": "", "out_dir": "", "log_csv": "violin_metrics_log.csv"},
    "plot": {"width": 8.5,"height": 4.5,"dpi": 150,"jitter": 0.06,"point_alpha": 0.6,
            "point_edge": "#000000","point_size": 18.0},
    "sample": {"swc_id": None}
  },
    "metrics": {
        "feature_file": "",
        "excel_file": "",
        "output_dir": "analysis_out",
        "neuron_types": ["SUBdd", "ProSub", "SUBv", "SUBvv", "SUBdv"],
        "colors": {"SUBdd": "#1f77b4",  "ProSub": "#ff7f0e", "SUBv": "#2ca02c", "SUBvv": "#d62728", "SUBdv": "#9467bd"},
        "plot": {"width": 8, "height": 6, "dpi": 300, "format": "svg"},
        "features": None
    }
}

def _deep_merge(a,b):
    out=copy.deepcopy(a)
    for k,v in (b or {}).items():
        if isinstance(v,dict) and isinstance(out.get(k),dict):
            out[k]=_deep_merge(out[k],v)
        else:
            out[k]=v
    return out

def _resolve(base,p):
    if not p: return p
    p=os.path.expandvars(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base,p))

def load_section(config_path: str, section: str):
    base_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path,"r",encoding="utf-8") as f:
        user = json.load(f)
    cfg = _deep_merge(DEFAULTS.get(section, {}), user.get(section, {}))

    if section == "radius_qc":
        io=cfg["io"]; thr=cfg["thresholds"]; nb=cfg["neighbors"]; pol=cfg["policy"]
        thr["p_low"]=float(thr["p_low"]); thr["p_high"]=float(thr["p_high"])
        if not (0<=thr["p_low"]<thr["p_high"]<=100): raise ValueError("p_low<p_high in [0,100]")
        cfg["compute"]["workers"]=int(cfg["compute"]["workers"])
        nb["depth"]=int(nb["depth"]); nb["neighbor_max"]=int(nb["neighbor_max"])
        pol["skip_soma_changes"]=bool(pol["skip_soma_changes"])
        # resolve paths
        io["input_dir"]=_resolve(base_dir, io["input_dir"])
        io["output_dir"]=_resolve(base_dir, io["output_dir"])
        io["log_csv"]    =_resolve(io["output_dir"], io["log_csv"])
        io["log_summary"]=_resolve(io["output_dir"], io["log_summary"])
    elif section == "orphan_clean":
        io=cfg["io"]; rec=cfg["reconnect"]
        rec["shift_accept_mad"]=float(rec["shift_accept_mad"])
        io["input_dir"]=_resolve(base_dir, io["input_dir"])
        io["output_dir"]=_resolve(base_dir, io["output_dir"])
        io["log_csv"]    =_resolve(io["output_dir"], io["log_csv"])
    elif section == "dendrograms":
        io = cfg["io"];
        draw = cfg["draw"]
        io["input_dir"] = _resolve(base_dir, io["input_dir"])
        io["output_dir"] = _resolve(base_dir, io["output_dir"])
        io["log_csv"] = _resolve(io["output_dir"], io["log_csv"])
        draw["width"] = float(draw.get("width", 10.0))
        draw["height"] = float(draw.get("height", 7.0))
        draw["dpi"] = int(draw.get("dpi", 300))
        fmt = (draw.get("format", "svg") or "svg").lower()
        if fmt not in ("svg", "pdf", "png"): fmt = "svg"
        draw["format"] = fmt
        cfg["min_nodes_per_type"] = int(cfg.get("min_nodes_per_type", 2))
    elif section == "violin_metrics":
        io = cfg["io"]
        io["xlsx"] = _resolve(base_dir, io["xlsx"])
        io["swc_dir"] = _resolve(base_dir, io["swc_dir"])
        io["out_dir"] = _resolve(base_dir, io["out_dir"])
        io["log_csv"] = _resolve(io["out_dir"], io["log_csv"])
    elif section == "metrics":
        cfg["feature_file"] = _resolve(base_dir, cfg["feature_file"])
        cfg["excel_file"]   = _resolve(base_dir, cfg["excel_file"])
        cfg["output_dir"]   = _resolve(base_dir, cfg["output_dir"])
        os.makedirs(cfg["output_dir"], exist_ok=True)
    else:
        raise ValueError(f"Unknown section: {section}")
    return cfg
