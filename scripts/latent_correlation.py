"""
==========================================================
Run Latent Correlation Analysis
==========================================================
"""

from src.neural_analysis.latent_correlation import (
    LatentCorrelation
)

analysis = LatentCorrelation(
    "experiments/version_1_0/results"
)

analysis.analyse()

print("\nLatent correlation analysis complete.")