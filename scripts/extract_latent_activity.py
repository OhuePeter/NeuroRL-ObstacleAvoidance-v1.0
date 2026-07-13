"""
==========================================================
Extract PPO Latent Activity

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Extracts the latent policy activations from a
trained PPO controller for all perturbation
conditions.

Version:
1.0
==========================================================
"""

from src.neural_analysis.activation_extractor import (
    ActivationExtractor,
)

MODEL = (
    "experiments/version_1_0/checkpoints/"
    "ppo_final.zip"
)

extractor = ActivationExtractor(MODEL)

# --------------------------------------------------
# Test on a single condition first
# --------------------------------------------------

extractor.extract_all(
    episodes=20
)

print("\nLatent extraction complete.")
