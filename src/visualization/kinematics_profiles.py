"""
==========================================================
Kinematic Profile Plotter

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Plots mean kinematic profiles for each condition.

Version:
1.0
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class KinematicProfilePlotter:

    def __init__(self, experiment_path):

        self.root = Path(experiment_path)

    def plot(self, condition):

        folder = self.root / f"evaluation_{condition}"

        files = sorted(folder.glob("kinematics_*.csv"))

        if len(files) == 0:
            print(f"No files found for {condition}")
            return

        dfs = [pd.read_csv(f) for f in files]

        min_len = min(len(df) for df in dfs)

        variables = [
            ("vx", "Velocity X"),
            ("vy", "Velocity Y"),
            ("speed", "Speed"),
            ("heading", "Heading (rad)"),
            ("goal_distance", "Distance to Goal"),
        ]

        fig, axes = plt.subplots(
            len(variables),
            1,
            figsize=(9, 12),
            sharex=True
        )

        for ax, (var, title) in zip(axes, variables):

            data = np.array([
                df[var].values[:min_len]
                for df in dfs
            ])

            mean = data.mean(axis=0)
            std = data.std(axis=0)

            x = np.arange(min_len)

            ax.plot(
                x,
                mean,
                color="black",
                linewidth=2
            )

            ax.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=0.3
            )

            ax.axvline(
                40,
                color="red",
                linestyle="--",
                linewidth=1.5
            )

            ax.set_ylabel(title)

        axes[-1].set_xlabel("Time Step")

        fig.suptitle(
            f"Kinematic Profiles ({condition})",
            fontsize=16
        )

        plt.tight_layout()

        plt.savefig(
            folder / f"kinematic_profiles_{condition}.png",
            dpi=300
        )

        plt.close()