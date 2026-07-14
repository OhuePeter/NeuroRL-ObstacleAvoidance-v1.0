"""
Experiment 2 Configuration

Robustness analysis under increased perturbation strength.

The PPO controller is NOT retrained.
Only evaluation conditions change.
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================

ROOT = Path("experiments/version_2_0")

CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
VIDEOS_DIR = ROOT / "videos"
LOG_DIR = ROOT / "logs"

# ============================================================
# Evaluation
# ============================================================

N_EPISODES = 20

SAVE_VIDEO = True
SAVE_TRAJECTORIES = True
SAVE_LATENTS = True
SAVE_FIGURES = True

# ============================================================
# Noise
# ============================================================

STATE_NOISE_STD = 0.005
ACTION_NOISE_STD = 0.01

PERTURBATION_JITTER_STD = 0.02

# ============================================================
# Perturbation Levels
# ============================================================

PERTURBATIONS = {

    "L1": -0.8,
    "L2": -1.5,
    "L3": -2.6,

    "R1":  0.8,
    "R2":  1.5,
    "R3":  2.6,
}