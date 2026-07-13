"""
==========================================================
Latent Trajectory Analysis

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Visualizes neural population trajectories in the
first two principal components.

==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class LatentTrajectory:

    def __init__(self, results_root):

        self.root = Path(results_root)

        self.conditions = [
            "P0",
            "L1",
            "L2",
            "L3",
            "R1",
            "R2",
            "R3",
        ]

        self.colors = {
            "P0": "black",
            "L1": "#6baed6",
            "L2": "#3182bd",
            "L3": "#08519c",
            "R1": "#fcae91",
            "R2": "#fb6a4a",
            "R3": "#cb181d",
        }

    def analyse(self):

        dataset = np.load(
            self.root / "latent_dataset.npz",
            allow_pickle=True
        )

        X = dataset["activations"]

        pca = PCA(n_components=2)
        pca.fit(X)

        plt.figure(figsize=(10, 8))

        latent_root = self.root / "neural_activity"

        for condition in self.conditions:

            file = latent_root / condition / "episode_000.npy"

            activity = np.load(file)

            trajectory = pca.transform(activity)

            plt.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                linewidth=2,
                color=self.colors[condition],
                label=condition,
            )

            # Start marker
            plt.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                marker="o",
                s=60,
                color=self.colors[condition],
            )

            # End marker
            plt.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                marker="X",
                s=80,
                color=self.colors[condition],
            )

        plt.xlabel(
            f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)"
        )

        plt.ylabel(
            f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)"
        )

        plt.title(
            "Evolution of Latent Neural States During Adaptive Behaviour",
            fontsize=15,
            fontweight="bold",
        )

        plt.grid(alpha=0.3)

        plt.legend(title="Condition")

        output = self.root / "LatentTrajectory.png"

        plt.savefig(
            output,
            dpi=600,
            bbox_inches="tight",
        )

        plt.close()

        print(f"\nSaved to:\n{output}")