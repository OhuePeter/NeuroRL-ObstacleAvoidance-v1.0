"""
==========================================================
Latent Hierarchical Clustering

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Hierarchical clustering of the PPO latent units
based on their pairwise correlations.

Version:
1.0
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram
)


class LatentClustering:

    def __init__(self, results_root):

        self.root = Path(results_root)

    def analyse(self):

        corr = np.load(
            self.root /
            "latent_correlation_matrix.npy"
        )

        # Distance = 1 - correlation
        distance = 1 - corr

        linkage_matrix = linkage(
            distance,
            method="average"
        )

        plt.figure(figsize=(14, 7))

        dendrogram(
            linkage_matrix,
            no_labels=True,
            color_threshold=0.7 * linkage_matrix[:, 2].max()
        )

        plt.title(
            "Hierarchical Clustering of PPO Latent Units",
            fontsize=18,
            fontweight="bold"
        )

        plt.xlabel(
            "Latent Unit",
            fontsize=12
        )

        plt.ylabel(
            "Cluster Distance",
            fontsize=12
        )

        plt.tight_layout()

        output = (
            self.root /
            "LatentDendrogram.png"
        )

        plt.savefig(
            output,
            dpi=600,
            bbox_inches="tight"
        )

        plt.close()

        print()

        print("=" * 60)
        print("LATENT CLUSTERING COMPLETE")
        print("=" * 60)

        print(f"Saved:\n{output}")