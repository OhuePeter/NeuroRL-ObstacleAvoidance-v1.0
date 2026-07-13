"""
==========================================================
Behaviour Summary Plot

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BehaviourSummaryPlotter:

    def __init__(self, results_root):

        self.root = Path(results_root)

        self.conditions = [
            "P0",
            "L1",
            "L2",
            "L3",
            "R1",
            "R2",
            "R3",
        ]

    def load(self):

        frames = []

        for condition in self.conditions:

            df = pd.read_csv(
                self.root /
                f"evaluation_{condition}" /
                "summary.csv"
            )

            df["condition"] = condition

            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def plot(self):

        df = self.load()

        metrics = [
            ("reward", "Reward"),
            ("steps", "Steps"),
            ("peak_lateral_velocity", "Peak Lateral Velocity"),
            ("max_heading_deviation", "Heading Deviation"),
            ("final_lateral_error", "Final Lateral Error"),
        ]

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(12, 12)
        )

        axes = axes.flatten()

        for ax, (metric, title) in zip(axes, metrics):

            sns.boxplot(
                data=df,
                x="condition",
                y=metric,
                ax=ax,
            )

            sns.stripplot(
                data=df,
                x="condition",
                y=metric,
                color="black",
                alpha=0.5,
                size=3,
                ax=ax,
            )

            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot
        axes[-1].axis("off")

        plt.tight_layout()

        output = self.root / "BehaviourSummary.png"

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {output}")