"""
==========================================================
Velocity Profile Plotter

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Plots velocity profiles for all perturbation
conditions.

Version:
1.0
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class VelocityProfilePlotter:

    def __init__(self, experiment_path):

        self.root = Path(experiment_path)

    def plot(self, condition):

        folder = self.root / f"evaluation_{condition}"

        files = sorted(folder.glob("kinematics_*.csv"))

        plt.figure(figsize=(8,5))

        for file in files:

            df = pd.read_csv(file)

            plt.plot(
                df["step"],
                df["speed"],
                alpha=0.25,
                color="steelblue"
            )

        mean_speed = None

        for i, file in enumerate(files):

            df = pd.read_csv(file)

            if mean_speed is None:

                mean_speed = df["speed"].values

            else:

                mean_speed += df["speed"].values

        mean_speed /= len(files)

        plt.plot(
            df["step"],
            mean_speed,
            linewidth=3,
            color="black",
            label="Mean"
        )

        plt.axvline(
            x=40,
            color="red",
            linestyle="--",
            label="Perturbation"
        )

        plt.xlabel("Time Step")

        plt.ylabel("Speed")

        plt.title(f"Velocity Profile ({condition})")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            folder / f"velocity_profile_{condition}.png",
            dpi=300
        )

        plt.close()