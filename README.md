# File Description

swctools/io.py – Robust SWC I/O: safely parse an SWC into a structured NumPy array (preserving original numeric strings) and write it back unchanged except for edits.

swctools/graph.py – Graph helpers for SWC trees: parent/child maps, adjacency, neighbor sets, connected components, and subtree extraction.

swctools/stats.py – Numeric utilities: per-type radius percentiles/means, 3D distances, component edge lengths, and a robust MAD/percentile threshold.

swctools/orphans.py – Cleans extra roots (“orphans”): merge duplicates, reconnect (with optional rigid shift), or split subtrees; writes the cleaned SWCs plus a unified node-level change log (orphan_changes_all.csv) and a file-level summary.

swctools/radius-qc.py – Flags outlier radii per SWC type using percentile thresholds, repairs them via local neighbor/type means, and records all adjustments to CSV.

swctools/dendrograms.py – Renders per-neuron dendrograms (SVG) with segments colored by SWC type (0..4 fixed, ≥5 custom); X-axis = path length from soma.

# How to Run

.\venv\Scripts\python.exe -m pip install -e .

## Clean orphans / extra roots
swc-clean-orphans --config D:\Desktop\SWC\config\config.json

## or run the radius QC pass
swc-radius-qc --config D:\Desktop\SWC\config\config.json

## create denfrogram
swc-dendrograms --config D:\Desktop\SWC\config\config.json

## create violin plots
swc-violin-metrics --config D:\Desktop\SWC\config\config.json

