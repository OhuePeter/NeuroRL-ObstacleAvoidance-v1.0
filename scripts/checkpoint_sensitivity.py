"""
==========================================================
Checkpoint Sensitivity of the Closed-Loop Fixed Points

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
1. Checks which numbered checkpoint best_model.zip is, by
   comparing parameters.
2. Recomputes the closed-loop fixed point and relaxation time
   (src/neural_analysis/attractor_analysis.py) at y = 5 m for
   every saved 13-D checkpoint, for both detour routes, so the
   manuscript can report how much these numbers move.

The forward-speed plateau is fixed at the value measured from
the best_model rollouts, so only the policy changes.

Usage
-----
python -m scripts.checkpoint_sensitivity
==========================================================
"""

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from scripts.cosyne_figure_data import Y_PROBE, analyse, load_rollouts
from src.neural_analysis.attractor_analysis import AttractorAnalysis

CKPT = Path("experiments/version_1_0/checkpoints")
ORDER = [
    "ppo_400000_steps", "ppo_800000_steps", "ppo_1200000_steps",
    "ppo_1600000_steps", "ppo_2000000_steps", "ppo_2400000_steps",
    "ppo_2800000_steps", "ppo_final", "best_model",
]


def same_weights(a, b):

    pa = PPO.load(CKPT / f"{a}.zip").policy.state_dict()
    pb = PPO.load(CKPT / f"{b}.zip").policy.state_dict()

    return all(torch.equal(pa[k], pb[k]) for k in pa)


def main():

    print("best_model identical to:")
    for name in ORDER[:-1]:
        print(f"  {name:20s} {same_weights('best_model', name)}")

    vy = analyse(load_rollouts())["vy_plateau"]
    print(f"\nforward-speed plateau used: {vy:.3f}\n")
    print(f"{'checkpoint':20s} {'x*_L':>6s} {'tau_L':>7s} {'x*_R':>6s} {'tau_R':>7s} "
          f"{'mirror x*_L+x*_R':>17s}")

    for name in ORDER:

        analysis = AttractorAnalysis(str(CKPT / f"{name}.zip"))
        row = []

        for route in ("left", "right"):

            x_star = analysis.find_lateral_fixed_point(Y_PROBE, vy, route=route)

            if x_star is None:
                row += [np.nan, np.nan]
                continue

            _, eig = analysis.local_jacobian(x_star, Y_PROBE, vy, route=route)
            tau = max(analysis.time_constant(e, analysis.dt) for e in eig)
            row += [x_star, tau]

        print(f"{name:20s} {row[0]:6.2f} {row[1]:6.2f}s {row[2]:6.2f} {row[3]:6.2f}s "
              f"{row[0] + row[2]:17.2f}")


if __name__ == "__main__":
    main()
