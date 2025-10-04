import os
from pathlib import Path
import csv

# --- Config ---
SWC_DIR = Path(r"D:\Desktop\Projectome SWC Files")
OUT_DIR = Path(r"D:\Desktop\SWC Output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.10  # 10%
EXCLUDE_TYPE_MINUS1 = False  # set True if you want to ignore type == -1 rows

CSV_REPORT = OUT_DIR / "unknown_type_report.csv"
TXT_OVER_THRESHOLD = OUT_DIR / "files_unknown_over_10pct.txt"

def parse_swc_count_types(swc_path: Path):
    """
    Parse an SWC file and return (total_nodes_considered, unknown_nodes).
    SWC columns: id type x y z radius parent
    Lines starting with '#' are comments.
    """
    total = 0
    unknown = 0
    with swc_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t = int(float(parts[1]))  # robust cast in case of "0.0"
            except ValueError:
                continue

            if EXCLUDE_TYPE_MINUS1 and t == -1:
                continue

            total += 1
            if t == 0:
                unknown += 1
    return total, unknown

def main():
    swc_files = sorted(SWC_DIR.glob("*.swc"))
    rows = []
    over_threshold_files = []

    for fp in swc_files:
        total, unknown = parse_swc_count_types(fp)
        if total == 0:
            pct = 0.0
        else:
            pct = unknown / total

        rows.append({
            "filename": fp.name,
            "total_nodes": total,
            "unknown_nodes": unknown,
            "percent_unknown": round(pct * 100.0, 4)  # as %
        })

        if pct > THRESHOLD:
            over_threshold_files.append(fp.name)

    # Write CSV report
    with CSV_REPORT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "total_nodes", "unknown_nodes", "percent_unknown"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write TXT list of files over threshold
    with TXT_OVER_THRESHOLD.open("w", encoding="utf-8") as f:
        f.write(f"Files with > {int(THRESHOLD*100)}% unknown-type nodes (type == 0)\n")
        f.write(f"Directory: {SWC_DIR}\n\n")
        for name in over_threshold_files:
            f.write(name + "\n")

    print(f"Done.\nCSV report: {CSV_REPORT}\nList (>10%): {TXT_OVER_THRESHOLD}")

if __name__ == "__main__":
    main()
