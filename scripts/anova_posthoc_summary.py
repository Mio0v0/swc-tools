"""
Summarize ANOVA + posthoc JSON into CSVs and a readable Markdown report
with aligned (padded) tables.

Inputs:
  - D:\Desktop\SWC Output\analysis\anova_posthoc_results.json

Outputs (same folder):
  - anova_summary.csv
  - posthoc_all.csv
  - posthoc_significant.csv
  - anova_posthoc_summary.md
"""

import json, os, re
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- config ----------
IN_PATH = r"D:\Desktop\SWC Output\analysis\anova_posthoc_results.json"
ALPHA = 0.05  # significance threshold on ANOVA p-unc and posthoc p-corr

# ---------- helpers ----------
def to_float(x):
    """Safely convert numbers/strings like '1.411e+07' to float; keep NaN as np.nan."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return np.nan
    try:
        return float(s)
    except Exception:
        s2 = re.sub(r"[^0-9eE+\-\.]", "", s)
        try:
            return float(s2)
        except Exception:
            return np.nan

def p_stars(p):
    if pd.isna(p): return ""
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

def format_p(p):
    if pd.isna(p): return "NA"
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"

def eta2_label(np2):
    if pd.isna(np2): return "NA"
    # Conventional (rule-of-thumb) partial eta^2 cutoffs
    if np2 >= 0.14: return "large"
    if np2 >= 0.06: return "medium"
    if np2 >= 0.01: return "small"
    return "very small"

def cohen_label(d):
    if pd.isna(d): return "NA"
    ad = abs(d)
    if ad >= 0.8: return "large"
    if ad >= 0.5: return "medium"
    if ad >= 0.2: return "small"
    return "very small"

def format_md_table(rows, headers, aligns=None):
    """
    Create an aligned Markdown table (cells padded to equal width).
    rows:    list[list[str]]
    headers: list[str]
    aligns:  list of 'l' | 'c' | 'r'
    """
    if aligns is None:
        aligns = ['l'] * len(headers)

    srows = [[("" if v is None else str(v)) for v in row] for row in rows]
    sheaders = [("" if h is None else str(h)) for h in headers]

    ncol = len(sheaders)
    widths = []
    for j in range(ncol):
        col_vals = [sheaders[j]] + [r[j] for r in srows]
        widths.append(max(len(x) for x in col_vals))

    def _pad(s, w, a):
        if a == 'r':
            return s.rjust(w)
        elif a == 'c':
            left = (w - len(s)) // 2
            right = w - len(s) - left
            return ' ' * left + s + ' ' * right
        else:
            return s.ljust(w)

    def _sep(w, a):
        if w < 2:  # guard tiny widths
            w = 2
        if a == 'r':
            return "-" * (w - 1) + ":"
        elif a == 'c':
            return ":" + "-" * (w - 2) + ":"
        else:  # 'l'
            return ":" + "-" * (w - 1)

    header_line = "| " + " | ".join(_pad(sheaders[j], widths[j], aligns[j]) for j in range(ncol)) + " |"
    sep_line    = "| " + " | ".join(_sep(widths[j], aligns[j]) for j in range(ncol)) + " |"
    row_lines   = [
        "| " + " | ".join(_pad(srows[i][j], widths[j], aligns[j]) for j in range(ncol)) + " |"
        for i in range(len(srows))
    ]
    return "\n".join([header_line, sep_line] + row_lines)

# ---------- load ----------
in_path = Path(IN_PATH)
out_dir = in_path.parent
with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------- collect ANOVA summary & posthoc ----------
anova_rows = []
posthoc_rows = []

for metric, blocks in data.items():
    # ANOVA: look for Source 'type'
    anova = blocks.get("anova", [])
    arow = next((a for a in anova if str(a.get("Source","")).lower() == "type"), None)
    if arow:
        F   = to_float(arow.get("F"))
        p   = to_float(arow.get("p-unc"))
        np2 = to_float(arow.get("np2"))
        anova_rows.append({
            "metric": metric,
            "F": F,
            "p_unc": p,
            "sig_at_alpha": (p < ALPHA) if pd.notna(p) else False,
            "stars": p_stars(p),
            "eta2_p": np2,
            "eta2_label": eta2_label(np2),
        })
    else:
        anova_rows.append({
            "metric": metric, "F": np.nan, "p_unc": np.nan,
            "sig_at_alpha": False, "stars": "", "eta2_p": np.nan, "eta2_label": "NA"
        })

    # Posthoc
    for ph in blocks.get("posthoc", []):
        A = ph.get("A"); B = ph.get("B")
        T = to_float(ph.get("T"))
        dof = to_float(ph.get("dof"))
        p_unc = to_float(ph.get("p-unc"))
        p_corr = to_float(ph.get("p-corr"))
        d = to_float(ph.get("cohen"))
        # BF10 might be scientific string
        BF10 = to_float(ph.get("BF10"))
        padj_method = ph.get("p-adjust", "")
        direction = "A > B" if (pd.notna(T) and T > 0) else ("A < B" if pd.notna(T) else "NA")

        posthoc_rows.append({
            "metric": metric, "A": A, "B": B,
            "T": T, "dof": dof,
            "p_unc": p_unc, "p_corr": p_corr, "stars": p_stars(p_corr),
            "p_adjust": padj_method,
            "cohen_d": d, "cohen_label": cohen_label(d),
            "BF10": BF10, "direction": direction
        })

anova_df = pd.DataFrame(anova_rows).sort_values(
    ["sig_at_alpha", "eta2_p", "F"], ascending=[False, False, False]
)
posthoc_df = pd.DataFrame(posthoc_rows)

# all pairs
posthoc_df_all = posthoc_df.sort_values(["metric", "p_corr"], ascending=[True, True])

# only significant at ALPHA on adjusted p
posthoc_df_sig = posthoc_df[posthoc_df["p_corr"] < ALPHA].copy()
posthoc_df_sig = posthoc_df_sig.sort_values(["metric", "p_corr"], ascending=[True, True])

# ---------- save CSVs ----------
anova_csv = out_dir / "anova_summary.csv"
posthoc_all_csv = out_dir / "posthoc_all.csv"
posthoc_sig_csv = out_dir / "posthoc_significant.csv"

anova_df.to_csv(anova_csv, index=False)
posthoc_df_all.to_csv(posthoc_all_csv, index=False)
posthoc_df_sig.to_csv(posthoc_sig_csv, index=False)

# ---------- build Markdown ----------
md_lines = []
md_lines.append("# ANOVA & Posthoc Summary\n")
md_lines.append(f"- Source file: `{in_path}`")
md_lines.append(f"- Alpha (significance on adjusted p): {ALPHA}\n")

# ANOVA table
md_lines.append("## ANOVA (by metric)\n")
anova_rows_print = []
for _, r in anova_df.iterrows():
    Ftxt   = "" if pd.isna(r["F"]) else f"{r['F']:.3f}"
    ptxt   = format_p(r["p_unc"])
    stars  = r["stars"]
    etatxt = "" if pd.isna(r["eta2_p"]) else f"{r['eta2_p']:.3f}"
    elabel = r["eta2_label"]
    anova_rows_print.append([r["metric"], Ftxt, ptxt, stars, etatxt, elabel])

headers = ["metric", "F", "p-unc", "sig", "η²ₚ", "effect"]
aligns  = ['l', 'r', 'r', 'c', 'r', 'l']  # numbers right, stars centered
md_lines.append(format_md_table(anova_rows_print, headers, aligns))
md_lines.append("")

# Significant posthoc by metric
md_lines.append(f"## Significant Posthoc Comparisons (adjusted p < {ALPHA:g})\n")
if posthoc_df_sig.empty:
    md_lines.append("_No significant pairwise differences after correction._\n")
else:
    for metric in posthoc_df_sig["metric"].unique():
        sdf = posthoc_df_sig[posthoc_df_sig["metric"] == metric]
        rows = []
        for _, r in sdf.iterrows():
            t_val = "" if pd.isna(r["T"]) else f"{r['T']:.3f}"
            dof   = "" if pd.isna(r["dof"]) else f"{r['dof']:.1f}"
            d_val = "" if pd.isna(r["cohen_d"]) else f"{r['cohen_d']:.3f}"
            # BF10 formatting: scientific if very large/small
            if pd.isna(r["BF10"]):
                BF = ""
            else:
                BF = f"{r['BF10']:.2e}" if (r["BF10"] >= 1e4 or r["BF10"] < 1e-3) else f"{r['BF10']:.3f}"

            rows.append([
                r["A"], r["B"], r["direction"],
                t_val, dof, format_p(r["p_corr"]), r["stars"],
                d_val, r["cohen_label"], BF
            ])

        headers = ["A", "B", "direction", "t", "df", "p-corr", "sig", "Cohen d", "effect", "BF10"]
        aligns  = ['l', 'l', 'c', 'r', 'r', 'r', 'c', 'r', 'l', 'r']
        md_lines.append(f"### {metric}\n")
        md_lines.append(format_md_table(rows, headers, aligns))
        md_lines.append("")

# Biggest effects (by partial eta^2)
md_lines.append("## Biggest Effects (by partial η²)\n")
top = anova_df.dropna(subset=["eta2_p"]).copy()
top = (anova_df.dropna(subset=["eta2_p"])
               .query("eta2_p >= 0.06")        # keep small-ish and up
               .sort_values("eta2_p", ascending=False))

if top.empty:
    md_lines.append("_No ANOVA rows with η²ₚ available._\n")
else:
    # Render as padded table too
    rows = []
    for _, r in top.iterrows():
        rows.append([
            r["metric"],
            "" if pd.isna(r["eta2_p"]) else f"{r['eta2_p']:.3f}",
            r["eta2_label"],
            "" if pd.isna(r["F"]) else f"{r['F']:.2f}",
            format_p(r["p_unc"]),
            r["stars"],
        ])
    headers = ["metric", "η²ₚ", "effect", "F", "p-unc", "sig"]
    aligns  = ['l', 'r', 'l', 'r', 'r', 'c']
    md_lines.append(format_md_table(rows, headers, aligns))
    md_lines.append("")

# Write Markdown
md_path = out_dir / "anova_posthoc_summary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("Wrote:")
print(f"- {anova_csv}")
print(f"- {posthoc_all_csv}")
print(f"- {posthoc_sig_csv}")
print(f"- {md_path}")