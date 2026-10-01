"""
==========================================================
Build Manuscript Figure 4: Latent PCA and Clustering

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Two-panel figure built directly from the saved latent
activation dataset, replacing the hand-edited
paper/fig4_pca.svg:
  Panel I  - PC1-PC2 projection coloured by perturbation
             condition.
  Panel II - the same projection coloured by K-means
             cluster label (K=4).

Fonts are set to Arial 10pt throughout (COSYNE convention).
Panel labels use Roman numerals (I, II) rather than letters.

Usage
-----
python scripts/build_fig4_pca.py
==========================================================
"""

from pathlib import Path

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"   # editable text in Inkscape
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["figure.dpi"] = 150

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path("experiments/version_1_0/results")
OUTPUT_PNG = Path("paper/fig4_pca.png")
OUTPUT_SVG = Path("paper/figures/fig4_pca.svg")

N_CLUSTERS = 4
RANDOM_STATE = 0

CONDITION_COLOURS = {
    "P0": "#555555",
    "L1": "#f4a9c4",
    "L2": "#e05c9a",
    "L3": "#a8004e",
    "R1": "#a0d8c8",
    "R2": "#3aab8c",
    "R3": "#005f47",
}

CLUSTER_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def panel_label(ax, label):
    ax.text(
        0.0, 1.03, label,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        va="bottom", ha="left",
    )


def main():

    dataset = np.load(ROOT / "latent_dataset.npz", allow_pickle=True)
    activations = dataset["activations"]
    condition = dataset["condition"]
    timestep = dataset["timestep"]

    # Units are z-scored before PCA since raw hidden-unit variance is
    # highly heterogeneous; this matches the standardized-PCA convention
    # used in the rest of the neural analysis pipeline.
    activations_z = StandardScaler().fit_transform(activations)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(activations_z)

    var_pc1 = 100 * pca.explained_variance_ratio_[0]
    var_pc2 = 100 * pca.explained_variance_ratio_[1]
    var_total = var_pc1 + var_pc2

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    raw_cluster_labels = kmeans.fit_predict(scores)

    # Renumber clusters by mean timestep so cluster 1 = earliest movement
    # phase and cluster N = latest, matching the manuscript narrative.
    order = np.argsort([timestep[raw_cluster_labels == k].mean() for k in range(N_CLUSTERS)])
    relabel = np.zeros(N_CLUSTERS, dtype=int)
    relabel[order] = np.arange(N_CLUSTERS)
    cluster_labels = relabel[raw_cluster_labels]

    fig, (ax_i, ax_ii) = plt.subplots(1, 2, figsize=(9.5, 4.4))

    for cond in sorted(np.unique(condition), key=lambda c: (c != "P0", c)):
        idx = condition == cond
        ax_i.scatter(
            scores[idx, 0], scores[idx, 1],
            s=5, alpha=0.35, label=cond, color=CONDITION_COLOURS[cond],
            linewidths=0,
        )
    ax_i.set_xlabel(f"PC1 ({var_pc1:.1f}%)")
    ax_i.set_ylabel(f"PC2 ({var_pc2:.1f}%)")
    ax_i.legend(frameon=False, fontsize=8, markerscale=2, loc="best")
    ax_i.spines[["top", "right"]].set_visible(False)
    panel_label(ax_i, "I")

    for cluster_id in range(N_CLUSTERS):
        idx = cluster_labels == cluster_id
        ax_ii.scatter(
            scores[idx, 0], scores[idx, 1],
            s=5, alpha=0.35, label=f"Cluster {cluster_id + 1}",
            color=CLUSTER_COLOURS[cluster_id % len(CLUSTER_COLOURS)],
            linewidths=0,
        )
    ax_ii.set_xlabel(f"PC1 ({var_pc1:.1f}%)")
    ax_ii.set_ylabel(f"PC2 ({var_pc2:.1f}%)")
    ax_ii.legend(frameon=False, fontsize=8, markerscale=2, loc="best")
    ax_ii.spines[["top", "right"]].set_visible(False)
    panel_label(ax_ii, "II")

    fig.tight_layout()

    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"PC1 variance: {var_pc1:.1f}%  PC2 variance: {var_pc2:.1f}%  Total: {var_total:.1f}%")
    print(f"Figure 4 saved to:\n{OUTPUT_PNG.resolve()}\n{OUTPUT_SVG.resolve()}")


if __name__ == "__main__":
    main()
