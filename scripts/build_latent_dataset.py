"""
==========================================================
Build Latent Dataset

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from src.neural_analysis.latent_dataset import (
    LatentDatasetBuilder,
)

builder = LatentDatasetBuilder(
    "experiments/version_1_0/results"
)

builder.build()

print("\nLatent dataset successfully created.")