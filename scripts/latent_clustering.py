"""
==========================================================
Run Latent Clustering
==========================================================
"""

from src.neural_analysis.latent_clustering import (
    LatentClustering
)

analysis = LatentClustering(
    "experiments/version_1_0/results"
)

analysis.analyse()

print("\nLatent clustering complete.")