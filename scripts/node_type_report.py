import os
import pandas as pd
from pathlib import Path

# --- Input / Output ---
swc_dir = Path(r"D:\Desktop\Projectome SWC Files")
out_csv = Path(r"D:\Desktop\SWC Output\swc_node_summary.csv")
out_txt = Path(r"D:\Desktop\SWC Output\swc_node_summary.txt")
out_xlsx = Path(r"D:\Desktop\SWC Output\swc_node_summary.xlsx")

# --- Define type names ---
TYPE_MAP = {
    0: "unknown",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite"
}

# --- Helper to process one SWC file ---
def process_swc(fpath: Path):
    cols = ["id", "type", "x", "y", "z", "radius", "parent"]
    df = pd.read_csv(
        fpath,
        sep=r"\s+",  # avoids FutureWarning
        comment="#",
        names=cols
    )

    total = len(df)
    type_counts = df["type"].value_counts().to_dict()

    row = {"file": fpath.name}

    # Add counts + percentages for fixed types (0–4)
    for t in range(0, 5):
        name = TYPE_MAP[t]
        count = type_counts.get(t, 0)
        perc = (count / total * 100) if total > 0 else 0
        row[f"{name}_count"] = count
        row[f"{name}_perc"] = f"{perc:.2f}%"

    # Handle custom types >=5
    custom_entries = []
    for t, count in sorted(type_counts.items()):
        if t >= 5:
            perc = (count / total * 100) if total > 0 else 0
            custom_entries.append(f"custom {t}: count {count}, {perc:.2f}%")

    row["customized"] = "; ".join(custom_entries) if custom_entries else ""

    return row

# --- Process all files ---
records = []
for fname in os.listdir(swc_dir):
    if fname.endswith(".swc"):
        fpath = swc_dir / fname
        records.append(process_swc(fpath))

# --- Save to CSV and Excel ---
df_out = pd.DataFrame(records)
df_out.to_csv(out_csv, index=False)
df_out.to_excel(out_xlsx, index=False, engine="openpyxl")  # Excel output

# --- Pretty print table ---
headers = list(df_out.columns)
col_widths = [max(len(str(x)) for x in df_out[col].astype(str).tolist() + [col]) for col in headers]

def format_row(row):
    return " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))

line = "-+-".join("-" * w for w in col_widths)

lines = []
lines.append(format_row(headers))
lines.append(line)
for _, row in df_out.iterrows():
    lines.append(format_row(row))

# Write to text file
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ Done!")
print("CSV saved to:", out_csv)
print("Excel saved to:", out_xlsx)
print("Text table saved to:", out_txt)
