import os, json, argparse
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pingouin as pg
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatter

from .config import load_section
from .metrics import read_celltype_table


def process_metrics(cfg):
    feature_file = cfg["feature_file"]
    excel_file = cfg["excel_file"]
    out_dir = cfg["output_dir"]
    neuron_types = cfg["neuron_types"]
    colors = cfg["colors"]
    features = cfg["features"]

    os.makedirs(out_dir, exist_ok=True)

    # --- Load feature JSON ---
    with open(feature_file, "r") as f:
        feat_data = json.load(f)

    df = pd.DataFrame.from_dict(feat_data, orient="index")
    df.index = df.index.str.replace(".swc", "")

    # Flatten lists (e.g. total_length_per_neurite = [val])
    for col in df.columns:
        df[col] = df[col].apply(lambda v: v[0] if isinstance(v, list) and len(v) == 1 else v)

    # --- Load Excel type mapping ---
    ids_by_type = read_celltype_table(excel_file)
    rows = []
    for t, ids in ids_by_type.items():
        for sid in ids:
            if sid in df.index:
                row = df.loc[sid].to_dict()
                row["cell_id"] = sid
                row["type"] = t
                rows.append(row)

    data = pd.DataFrame(rows)

    # --- Ensure all feature cols are numeric ---
    for feat in features:
        data[feat] = pd.to_numeric(data[feat], errors="coerce")

    # --- Save descriptive stats ---
    desc = data.groupby("type")[features].agg(["mean", "median", "std"])
    desc.to_csv(os.path.join(out_dir, "descriptive_stats.csv"))

    # --- ANOVA for each feature ---
    anova_results = {}
    for feat in features:
        try:
            # skip constant features
            if data[feat].nunique() <= 1:
                anova_results[feat] = {"note": "constant feature"}
                continue

            # One-way ANOVA with effect size (partial eta squared)
            aov = pg.anova(data=data, dv=feat, between="type", detailed=True, effsize="np2")
            aov_out = aov[["Source", "F", "p-unc", "np2"]].to_dict(orient="records")

            # Posthoc pairwise tests (Bonferroni corrected, Cohen’s d effect size)
            posthoc = pg.pairwise_tests(
                data=data,
                dv=feat,
                between="type",
                padjust="bonferroni",
                effsize="cohen"
            )
            posthoc_out = posthoc.to_dict(orient="records")

            anova_results[feat] = {"anova": aov_out, "posthoc": posthoc_out}
        except Exception as e:
            anova_results[feat] = {"error": str(e)}

    with open(os.path.join(out_dir, "anova_posthoc_results.json"), "w") as f:
        json.dump(anova_results, f, indent=2)

    # --- Plots ---
    # --- Violin plots ---
    n_feats = len(features)
    n_cols = 3  # number of plots per row
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cfg["plot"]["width"] * n_cols, cfg["plot"]["height"] * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        sns.violinplot(x="type", y=feat, hue="type", data=data, palette=colors, ax=ax, legend=False)
        sns.stripplot(x="type", y=feat, hue="type", data=data, palette=colors,
                      size=3, jitter=0.25, alpha=0.6, ax=ax, dodge=False, legend=False)

        ax.set_title(feat, fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "violin_all_features.svg"), dpi=cfg["plot"]["dpi"])
    plt.close()

    # Radar plot (per type mean values)
    means = desc.xs("mean", level=1, axis=1)  # mean values per feature

    # scale each feature across types to 0–1
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(means.values)

    features_norm = means.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(features_norm), endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    plt.figure(figsize=(20, 20))
    ax = plt.subplot(111, polar=True)
    for i, t in enumerate(neuron_types):
        vals = scaled_values[i].tolist()
        vals += vals[:1]  # close the polygon
        ax.plot(angles, vals, label=t, color=colors[t], linewidth=3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_norm, fontsize=8)
    plt.legend()
    plt.title("Radar Plot (Mean Feature Values, normalized 0–1)")
    plt.savefig(os.path.join(out_dir, "radar_plot.svg"), dpi=cfg["plot"]["dpi"])
    plt.close()

    # Scatter matrix
    sns.pairplot(data[features + ["type"]], hue="type", palette=colors)
    plt.savefig(os.path.join(out_dir, "scatter_matrix.svg"), dpi=cfg["plot"]["dpi"])
    plt.close()

    # --- Scatter plots (all features in one canvas) ---
    n_feats = len(features)
    n_cols = 3  # you can change depending on how many plots per row
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cfg["plot"]["width"] * n_cols, cfg["plot"]["height"] * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        sns.scatterplot(
            x=feat, y="cell_id", hue="type", data=data, palette=colors, s=30, alpha=0.7, ax=ax
        )
        ax.set_yticks([])  # hide x labels since they’re just IDs
        ax.set_xlabel(feat)
        ax.set_ylabel("Sample ID")
        ax.set_title(feat, fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_all_features.svg"), dpi=cfg["plot"]["dpi"])
    plt.close()

    # Heatmap of mean feature values
    vmin, vmax = 1e-1, 1e7  # 10^-1 to 10^7

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        means.T,
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),  # fixed log range
        annot=True, fmt=".2f",
        cbar_kws={"label": "Mean (log scale)"}
    )

    # Nice decade ticks: 10^-1, 10^0, 10^1, ..., 10^7
    cbar = ax.collections[0].colorbar
    cbar.locator = LogLocator(base=10, subs=(1.0,))  # decades only
    cbar.formatter = LogFormatter(base=10)
    cbar.update_ticks()

    plt.title("Mean Feature Values per Type")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_means_log.svg"), dpi=cfg["plot"]["dpi"])
    plt.close()

    # --- Clustering (KMeans) ---
    X = data[features].dropna().values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=5, random_state=42).fit(X_scaled)
    labels = km.labels_
    sil = silhouette_score(X_scaled, labels)
    with open(os.path.join(out_dir, "clustering_results.json"), "w") as f:
        json.dump({"silhouette_score": sil}, f, indent=2)

    print(f"✔ Analysis complete. Results in {out_dir}")


def cli_main():
    ap = argparse.ArgumentParser(
        description="Morphology feature analysis: stats, plots, ANOVA, clustering"
    )
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--section", default="metrics", help="Config section name")
    args = ap.parse_args()
    cfg = load_section(args.config, args.section)
    process_metrics(cfg)
