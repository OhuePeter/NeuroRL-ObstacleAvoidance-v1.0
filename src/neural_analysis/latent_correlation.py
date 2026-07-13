"""
==========================================================
Latent Correlation Analysis

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Computes the correlation matrix of PPO latent
representations and generates a publication-
quality heatmap.

Version:
1.0
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class LatentCorrelation:

    def __init__(self, results_root):

        self.root = Path(results_root)

    def analyse(self):

        dataset = np.load(
            self.root / "latent_dataset.npz",
            allow_pickle=True
        )

        X = dataset["activations"]

        print("\n" + "=" * 60)
        print("LATENT CORRELATION ANALYSIS")
        print("=" * 60)

        print(f"Samples : {X.shape[0]}")
        print(f"Latent Units : {X.shape[1]}")

        corr = np.corrcoef(
            X.T
        )

        np.save(
            self.root / "latent_correlation_matrix.npy",
            corr
        )

        plt.figure(
            figsize=(10, 8)
        )

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={
                "label": "Pearson Correlation"
            }
        )

        plt.title(
            "Functional Correlation Structure of PPO Latent Units",
            fontsize=16,
            fontweight="bold"
        )

        plt.xlabel(
            "Latent Unit",
            fontsize=12
        )

        plt.ylabel(
            "Latent Unit",
            fontsize=12
        )

        output = (
            self.root /
            "LatentCorrelationMatrix.png"
        )

        plt.savefig(
            output,
            dpi=600,
            bbox_inches="tight"
        )

        plt.close()

        print("\nSaved:")
        print(output)