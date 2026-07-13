"""
==========================================================
Latent Dataset Builder

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Builds a single analysis-ready dataset from the
latent activity extracted from all perturbation
conditions.

Version:
1.0
==========================================================
"""

from pathlib import Path

import numpy as np


class LatentDatasetBuilder:
    """
    Combine latent activity from all conditions
    into a single dataset.
    """

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

    def build(self):

        activations = []
        conditions = []
        episodes = []
        timesteps = []

        latent_root = self.root / "neural_activity"

        print()
        print("=" * 60)
        print("BUILDING LATENT DATASET")
        print("=" * 60)

        for condition in self.conditions:

            condition_dir = latent_root / condition

            files = sorted(
                condition_dir.glob("episode_*.npy")
            )

            print(
                f"{condition}: {len(files)} episodes"
            )

            for episode_id, file in enumerate(files):

                data = np.load(file)

                for timestep, latent in enumerate(data):

                    activations.append(latent)
                    conditions.append(condition)
                    episodes.append(episode_id)
                    timesteps.append(timestep)

        activations = np.asarray(
            activations,
            dtype=np.float32
        )

        conditions = np.asarray(conditions)

        episodes = np.asarray(
            episodes,
            dtype=np.int32
        )

        timesteps = np.asarray(
            timesteps,
            dtype=np.int32
        )

        output = self.root / "latent_dataset.npz"

        np.savez_compressed(
            output,
            activations=activations,
            condition=conditions,
            episode=episodes,
            timestep=timesteps,
        )

        print()
        print("=" * 60)
        print("LATENT DATASET COMPLETE")
        print("=" * 60)
        print(f"Samples : {len(activations)}")
        print(f"Latent shape : {activations.shape}")
        print(f"Saved to : {output}")

        return output