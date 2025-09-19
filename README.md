# SWC Tools

A small toolkit for cleaning and analyzing neuron morphologies in **SWC** format.

---

## Contents (what each file does)

- `swctools/io.py` – **Robust SWC I/O.** Safely parses an SWC into a structured NumPy array (preserving original numeric strings) and writes it back unchanged except for your edits.

- `swctools/graph.py` – **Tree helpers.** Parent/child maps, adjacency, neighbor sets, connected components, and subtree extraction for SWC trees.

- `swctools/stats.py` – **Numeric utilities.** Per-type radius percentiles/means, 3D distances, component edge lengths, and a robust MAD/percentile threshold.

- `swctools/orphans.py` – **Clean extra roots (“orphans”).** Merges exact duplicates, reconnects to soma (optionally after a rigid shift), or splits subtrees. Writes:
  - cleaned SWCs
  - **node-level change log**: `orphan_changes_all.csv`
  - **file-level summary** CSV

- `swctools/radius_qc.py` – **Radius quality control.** Flags outlier radii per SWC node **by SWC type** using percentile thresholds; repairs using local neighbor/type means; logs node-level adjustments to CSV.

- `swctools/dendrograms.py` – **Dendrogram renderer (SVG).** Per-neuron dendrograms with segments colored by SWC type:
  - `0` = undefined
  - `1` = soma
  - `2` = axon
  - `3` = (basal) dendrite
  - `4` = apical dendrite
  - `5+` = custom  
  *X-axis = path length from soma.*

- `swctools/metrics.py` + `swctools/violin_plots.py` – **Metrics & violin plots.** Computes per-neuron metrics and draws SVG violin plots per metric across five neuron types listed in an Excel mapping (see below). Overlays each SWC as a jittered point; optional sample SWC highlight.

---

## Prerequisites

- **Python 3.9+**
- A virtual environment is recommended.

---

## Install (editable)

From the project root (the folder that contains `pyproject.toml`):

```bash
# Windows PowerShell
python -m venv .\venv
.\venv\Scripts\activate

python -m pip install -U pip
python -m pip install -e .


