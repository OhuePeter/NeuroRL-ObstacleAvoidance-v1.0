"""
==========================================================
Behaviour Summary Plot

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Creates a publication-quality summary figure of the
behavioural effects of perturbations.

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

        self.condition_labels = [
            "Control",
            "Left\nWeak",
            "Left\nModerate",
            "Left\nStrong",
            "Right\nWeak",
            "Right\nModerate",
            "Right\nStrong",
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

        return pd.concat(
            frames,
            ignore_index=True
        )

    def plot(self):

        df = self.load()

        metrics = [

            (
                "reward",
                "Task Performance (Reward)"
            ),

            (
                "steps",
                "Movement Duration (Steps)"
            ),

            (
                "peak_lateral_velocity",
                "Peak Lateral Velocity"
            ),

            (
                "max_heading_deviation",
                "Maximum Heading Deviation"
            ),

            (
                "final_lateral_error",
                "Final Lateral Error"
            ),

        ]

        plt.style.use("seaborn-v0_8-whitegrid")

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(15, 12),
            constrained_layout=True
        )

        axes = axes.flatten()

        palette = sns.color_palette(
            "viridis",
            7
        )

        panels = [
            "A",
            "B",
            "C",
            "D",
            "E"
        ]

        for i, (metric, title) in enumerate(metrics):

            ax = axes[i]

            sns.boxplot(
                data=df,
                x="condition",
                y=metric,
                palette=palette,
                linewidth=1.5,
                width=0.60,
                showfliers=False,
                ax=ax
            )

            sns.stripplot(
                data=df,
                x="condition",
                y=metric,
                color="black",
                alpha=0.65,
                size=4,
                jitter=0.15,
                ax=ax
            )

            ax.set_xticklabels(
                self.condition_labels,
                fontsize=10
            )

            ax.set_xlabel(
                "Perturbation Condition",
                fontsize=11,
                fontweight="bold"
            )

            ax.set_ylabel(
                title,
                fontsize=11
            )

            ax.set_title(
                f"{panels[i]}. {title}",
                fontsize=13,
                fontweight="bold",
                pad=12
            )

            ax.grid(
                linestyle="--",
                alpha=0.30
            )

        axes[-1].axis("off")

        fig.suptitle(
            "Adaptive Motor Behaviour Under Lateral Perturbations",
            fontsize=20,
            fontweight="bold"
        )

        output = (
            self.root /
            "BehaviourSummary.png"
        )

        plt.savefig(
            output,
            dpi=600,
            bbox_inches="tight"
        )

        plt.close()

        print("\nSaved figure to:")
        print(output)