"""
==========================================================
Publication Figures

Experiment 2

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Generates publication-quality figures from the behavioural
summary files produced during Experiment 2.

Outputs
-------
behaviour_boxplots.png
reward_boxplot.png
steps_boxplot.png
path_length_boxplot.png
mean_speed_boxplot.png
max_speed_boxplot.png
peak_lateral_velocity_boxplot.png
heading_deviation_boxplot.png
lateral_error_boxplot.png

==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------
# Directories
# ----------------------------------------------------------

ROOT = Path("experiments/version_2_0/results")

OUTPUT_DIR = ROOT / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ----------------------------------------------------------
# Conditions
# ----------------------------------------------------------

CONDITIONS = [
    "P0",
    "L1",
    "L2",
    "L3",
    "R1",
    "R2",
    "R3",
]

CONDITION_NAMES = {
    "P0": "Control",
    "L1": "Small\nLeft",
    "L2": "Medium\nLeft",
    "L3": "Large\nLeft",
    "R1": "Small\nRight",
    "R2": "Medium\nRight",
    "R3": "Large\nRight",
}

# ----------------------------------------------------------
# Variables
# ----------------------------------------------------------

VARIABLES = {

    "reward": "Episode Reward",

    "steps": "Episode Duration",

    "path_length": "Path Length",

    "mean_speed": "Mean Speed",

    "max_speed": "Maximum Speed",

    "peak_lateral_velocity": "Peak Lateral Velocity",

    "max_heading_deviation": "Maximum Heading Deviation",

    "final_lateral_error": "Final Lateral Error",

}

# ----------------------------------------------------------
# Load all summary files
# ----------------------------------------------------------

def load_data():

    frames = []

    for condition in CONDITIONS:

        filename = (
            ROOT
            / f"evaluation_{condition}"
            / "summary.csv"
        )

        if not filename.exists():
            continue

        df = pd.read_csv(filename)

        df["condition"] = CONDITION_NAMES[condition]

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )

# ----------------------------------------------------------
# Combined publication figure
# ----------------------------------------------------------

def combined_boxplots(data):

    sns.set_theme(
        style="whitegrid",
        context="paper",
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18,10),
        constrained_layout=True,
    )

    axes = axes.flatten()

    panel_labels = list("ABCDEFGH")

    for ax, variable, panel in zip(
        axes,
        VARIABLES.keys(),
        panel_labels,
    ):

        sns.boxplot(
            data=data,
            x="condition",
            y=variable,
            palette="Set2",
            linewidth=1.5,
            ax=ax,
        )

        sns.stripplot(
            data=data,
            x="condition",
            y=variable,
            color="black",
            alpha=0.45,
            size=3,
            jitter=True,
            ax=ax,
        )

        ax.set_title(
            f"({panel}) {VARIABLES[variable]}",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel("")

        ax.tick_params(
            axis="x",
            labelrotation=0,
        )

    plt.suptitle(
        "Behavioural Performance Under Increasing Lateral Perturbations",
        fontsize=18,
        fontweight="bold",
    )

    plt.savefig(
        OUTPUT_DIR / "behaviour_boxplots.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved behaviour_boxplots.png")

# ----------------------------------------------------------
# Individual publication figures
# ----------------------------------------------------------

def individual_boxplots(data):

    sns.set_theme(
        style="whitegrid",
        context="paper",
    )

    for variable in VARIABLES:

        plt.figure(figsize=(7,6))

        sns.boxplot(
            data=data,
            x="condition",
            y=variable,
            palette="Set2",
            linewidth=1.5,
        )

        sns.stripplot(
            data=data,
            x="condition",
            y=variable,
            color="black",
            alpha=0.45,
            size=4,
            jitter=True,
        )

        plt.title(
            VARIABLES[variable],
            fontsize=15,
            fontweight="bold",
        )

        plt.xlabel(
            "Perturbation Condition",
            fontsize=12,
        )

        plt.ylabel(
            VARIABLES[variable],
            fontsize=12,
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{variable}_boxplot.png",
            dpi=600,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved {variable}_boxplot.png")

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    print("=" * 70)
    print("Generating Publication Figures")
    print("=" * 70)

    data = load_data()

    combined_boxplots(data)

    individual_boxplots(data)

    print()
    print("=" * 70)
    print("PUBLICATION FIGURES COMPLETE")
    print("=" * 70)
    print(f"Figures saved to:\n{OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()