"""
==========================================================
Run Latent PCA
==========================================================
"""

from src.neural_analysis.latent_pca import LatentPCA

analysis = LatentPCA(
    "experiments/version_1_0/results"
)

analysis.analyse()

print("\nLatent PCA complete.")