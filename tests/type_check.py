#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv

# ========================
# CONFIG
# ========================
INPUT_DIR = r"D:\Desktop\Projectome SWC Files"
OUTPUT_CSV = r"D:\Desktop\Projectome_SWCTypes_Report.csv"
PATTERN = "*.swc"
RECURSIVE = False

# Allowed types
ALLOWED_TYPES = set(range(0, 8))  # {0,1,2,3,4,5,6,7}

def _safe_int(x, default=-1):
    try:
        return int(float(x))
    except Exception:
        return default

def check_file(path):
    bad_types = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            t = _safe_int(parts[1], default=-1)
            if t not in ALLOWED_TYPES:
                bad_types.add(t)
    return bad_types

def main():
    print("=== Checking SWC types ===")
    pattern = "**/*.swc" if RECURSIVE else PATTERN
    files = sorted(glob.glob(os.path.join(INPUT_DIR, pattern), recursive=RECURSIVE))
    if not files:
        print("No SWC files found.")
        return

    rows = []
    for i, f in enumerate(files, 1):
        bad = check_file(f)
        if bad:
            rows.append({
                "file": os.path.basename(f),
                "path": f,
                "unexpected_types": ";".join(map(str, sorted(bad)))
            })
            print(f"[{i}/{len(files)}] {os.path.basename(f)} -> Unexpected types: {sorted(bad)}")
        else:
            print(f"[{i}/{len(files)}] {os.path.basename(f)} -> OK")

    # Write CSV
    if rows:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=["file", "path", "unexpected_types"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nReport written to: {OUTPUT_CSV}")
    else:
        print("\nAll files OK. No unexpected types found.")

if __name__ == "__main__":
    main()
