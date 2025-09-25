'''
NeuroM Checkers
'''
import morphio
from neurom.core import Morphology
from neurom.check import morphology_checks as checks
import json

morphio.set_maximum_warnings(1)

raw = morphio.Morphology(
    "D:/Desktop/SWC_Fixed/200783_024.swc",
    options=morphio.Option.allow_unifurcated_section_change
)

morph = Morphology(raw)

# 2. Run all check functions starting with 'has_'
results = {}
for name in dir(checks):
    if name.startswith("has_"):
        func = getattr(checks, name)
        if callable(func):
            try:
                if "neurite_filter" in func.__code__.co_varnames:
                    results[name] = bool(func(morph, neurite_filter=None))
                else:
                    results[name] = bool(func(morph))
            except Exception as e:
                results[name] = f"ERROR: {e}"

# 3. Save results
with open("D:/Desktop/SWC/summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("Checks finished → results saved to summary.json")

'''
Plots
'''
from neurom.view import plot_morph, plot_morph3d, plot_dendrogram
import matplotlib.pyplot as plt

plt.figure(figsize=(25, 20))
plot_morph(morph)
plt.savefig("..._morph.png", dpi=300)

plt.figure(figsize=(25, 20))
plot_morph3d(morph)
plt.savefig("..._morph3d.png", dpi=300)

fig, ax = plt.subplots(figsize=(25, 30), dpi=200)
plot_dendrogram(morph, ax=ax)   # ensure it draws on your custom axes
fig.tight_layout()
plt.savefig("..._dendrogram.png", dpi=300)
plt.close('all')

import neurom
from neurom import features
m = neurom.load_morphology("tests/data/swc/Neuron.swc")
result = features.get('soma_surface_area', m)

'''
Stats
'''
import os
import glob
import pandas as pd
import morphio
from neurom.core import Morphology
from neurom import features
import json
import tempfile
# ------------------------
# PARAMETERS
# ------------------------
INPUT_DIR = r"D:\Desktop\Projectome SWC Files"
OUTPUT_FILE = r"D:\Desktop\SWC Output\partial_features.json"
N_FILES = None    # set to 10, 100, or None for ALL files
# ------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
FEATURE_LIST = [
    "aspect_ratio",
    "circularity",
    "length_fraction_above_soma",
    "max_radial_distance",
    "neurite_volume_density",
    "number_of_neurites",
    "number_of_sections_per_neurite",
    # "section_bif_radial_distances",
    # "section_radial_distances",
    # "section_term_radial_distances",
    # "segment_radial_distances",
    "shape_factor",
    # "sholl_crossings",
    # "sholl_frequency",
    "soma_radius",
    "soma_surface_area",
    "soma_volume",
    "total_area_per_neurite",
    "total_depth",
    "total_height",
    "total_length_per_neurite",
    "total_volume_per_neurite",
    "total_width",
    "trunk_angles",
    # "trunk_angles_from_vector",
    # "trunk_angles_inter_types",
    # "trunk_origin_azimuths",
    # "trunk_origin_elevations",
    "trunk_origin_radii",
    "trunk_section_lengths",
    # "trunk_vectors",
    "volume_density",
]
# Collect files
all_files = glob.glob(os.path.join(INPUT_DIR, "*.swc"))
if N_FILES is not None:
    files_to_process = all_files[:N_FILES]
else:
    files_to_process = all_files
print(f"Found {len(all_files)} SWC files, processing {len(files_to_process)}...")
all_results = {}
for swc_file in files_to_process:
    print(f"Processing {swc_file}")
    # --- Step 1: Fix invalid types in memory ---
    df = pd.read_csv(swc_file, sep=r"\s+", comment="#", header=None, engine="python")
    df[1] = df[1].apply(lambda t: 7 if (t == 0 or t > 7) else t)
    # --- Step 2: Write to a temporary file ---
    with tempfile.NamedTemporaryFile(suffix=".swc", delete=False) as tmp:
        df.to_csv(tmp.name, sep=" ", header=False, index=False)
        tmp_path = tmp.name
    # --- Step 3: Load with MorphIO (relaxed option) ---
    raw = morphio.Morphology(
        tmp_path,
        options=morphio.Option.allow_unifurcated_section_change
    )
    morph = Morphology(raw)
    # --- Step 4: Compute features ---
    file_results = {}
    for feat in FEATURE_LIST:
        try:
            file_results[feat] = features.get(feat, morph)
        except Exception as e:
            file_results[feat] = f"ERROR: {e}"
    # Add results
    all_results[os.path.basename(swc_file)] = file_results
    # --- 🔥 Step 5: SAVE immediately after this file ---
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    # Clean up
    os.remove(tmp_path)
    # Optional: show progress
    print(f"✔ Saved progress for {swc_file}")

