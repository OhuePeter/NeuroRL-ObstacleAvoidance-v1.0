"""
==========================================================
Latent PCA Analysis

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Performs PCA on the latent neural activity of the
trained PPO policy and generates publication-
quality figures.

Version:
1.0
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class LatentPCA:

    def __init__(self, results_root):

        self.root = Path(results_root)

    def analyse(self):

        dataset = np.load(
            self.root / "latent_dataset.npz",
            allow_pickle=True
        )

        X = dataset["activations"]
        condition = dataset["condition"]

        print("\n" + "=" * 60)
        print("LATENT PCA")
        print("=" * 60)

        print(f"Samples : {X.shape[0]}")
        print(f"Neurons : {X.shape[1]}")

        pca = PCA(n_components=2)

        scores = pca.fit_transform(X)

        print("\nExplained variance")

        for i, var in enumerate(
            pca.explained_variance_ratio_
        ):

            print(
                f"PC{i+1}: {100*var:.2f}%"
            )

        colors = {
            "P0": "#000000",
            "L1": "#1f77b4",
            "L2": "#4fa3ff",
            "L3": "#9ecae1",
            "R1": "#d62728",
            "R2": "#ff7f0e",
            "R3": "#ffbb78",
        }

        plt.figure(figsize=(9, 7))

        for c in np.unique(condition):

            idx = condition == c

            plt.scatter(
                scores[idx, 0],
                scores[idx, 1],
                s=6,
                alpha=0.35,
                label=c,
                color=colors[c]
            )

        plt.xlabel(
            f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)",
            fontsize=12
        )

        plt.ylabel(
            f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)",
            fontsize=12
        )

        plt.title(
            "Latent Neural State Space Across Perturbation Conditions",
            fontsize=15,
            fontweight="bold"
        )

        plt.legend()

        plt.grid(alpha=0.3)

        output = (
            self.root /
            "LatentPCA.png"
        )

        plt.savefig(
            output,
            dpi=600,
            bbox_inches="tight"
        )

        plt.close()

        print(f"\nSaved to:\n{output}")