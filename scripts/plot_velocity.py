"""
==========================================================
Plot Velocity Profiles

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Plots the velocity profile across all evaluation
episodes for the baseline (P0) condition.

Version:
1.0
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():

    results_dir = Path(
        "experiments/version_1_0/results/evaluation_P0"
    )

    figures_dir = Path(
        "experiments/version_1_0/figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(results_dir.glob("kinematics_*.csv"))

    plt.figure(figsize=(8, 5))

    for file in files:

        df = pd.read_csv(file)

        plt.plot(
            df["step"],
            df["speed"],
            color="royalblue",
            alpha=0.35,
            linewidth=1.2
        )

    plt.xlabel("Step")
    plt.ylabel("Speed (units/s)")
    plt.title("Velocity Profiles (P0)")
    plt.grid(True)

    output = figures_dir / "velocity_profile_P0.png"

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

    print("=" * 60)
    print("VELOCITY PROFILE CREATED")
    print("=" * 60)
    print(f"Saved to:\n{output}")


if __name__ == "__main__":
    main()