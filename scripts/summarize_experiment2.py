"""
==========================================================
Experiment 2 Robustness Summary

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Summarizes robustness results across all perturbation
conditions and generates publication-quality figures.

Outputs
-------
- robustness_summary.csv
- success_failure_rate.png
- reward.png
- final_error.png

==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# Paths
# ==========================================================

ROOT = Path("experiments/version_2_0/results")


# ==========================================================
# Publication Labels
# ==========================================================

CONDITION_LABELS = {
    "P0": "No Perturbation",
    "L1": "Small Left Perturbation",
    "L2": "Medium Left Perturbation",
    "L3": "Large Left Perturbation",
    "R1": "Small Right Perturbation",
    "R2": "Medium Right Perturbation",
    "R3": "Large Right Perturbation",
}


# ==========================================================
# Load Results
# ==========================================================

def load_condition(condition):

    summary_file = ROOT / f"evaluation_{condition}" / "summary.csv"

    df = pd.read_csv(summary_file)

    success_rate = 100 * df["success"].mean()
    failure_rate = 100 - success_rate

    return {
        "Condition": condition,
        "Label": CONDITION_LABELS[condition],
        "Success Rate": success_rate,
        "Failure Rate": failure_rate,
        "Collision Rate": 100 * df["collision"].mean(),
        "Reward": df["reward"].mean(),
        "Steps": df["steps"].mean(),
        "Path Length": df["path_length"].mean(),
        "Mean Speed": df["mean_speed"].mean(),
        "Final Error": df["final_lateral_error"].mean(),
    }


# ==========================================================
# Main
# ==========================================================

def main():

    conditions = [
        "P0",
        "L1",
        "L2",
        "L3",
        "R1",
        "R2",
        "R3",
    ]

    rows = [load_condition(c) for c in conditions]

    results = pd.DataFrame(rows)

    output_dir = ROOT / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "robustness_summary.csv"

    results.to_csv(csv_file, index=False)

    # ======================================================
    # Figure 1
    # Success vs Failure
    # ======================================================

    plt.figure(figsize=(11, 6))

    success = results["Success Rate"]
    failure = results["Failure Rate"]

    bars1 = plt.bar(
        results["Label"],
        success,
        color="forestgreen",
        edgecolor="black",
        linewidth=1.2,
        label="Success",
    )

    bars2 = plt.bar(
        results["Label"],
        failure,
        bottom=success,
        color="crimson",
        edgecolor="black",
        linewidth=1.2,
        label="Failure",
    )

    for bar, value in zip(bars1, success):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            value/2,
            f"{value:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    for bar, s, f in zip(bars2, success, failure):

        if f > 0:

            plt.text(
                bar.get_x() + bar.get_width()/2,
                s + f/2,
                f"{f:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

    plt.title(
        "Experiment 2: Policy Robustness Under External Perturbations",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel("Perturbation Condition", fontsize=12)
    plt.ylabel("Episodes (%)", fontsize=12)

    plt.ylim(0, 100)

    plt.xticks(rotation=20, ha="right")

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "success_failure_rate.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ======================================================
    # Figure 2
    # Reward
    # ======================================================

    plt.figure(figsize=(11, 6))

    plt.plot(
        results["Label"],
        results["Reward"],
        marker="o",
        linewidth=3,
        markersize=8,
        color="royalblue",
    )

    plt.title(
        "Average Episode Reward Across Perturbation Conditions",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel("Perturbation Condition")
    plt.ylabel("Average Reward")

    plt.xticks(rotation=20, ha="right")

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "reward.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ======================================================
    # Figure 3
    # Final Lateral Error
    # ======================================================

    plt.figure(figsize=(11, 6))

    plt.plot(
        results["Label"],
        results["Final Error"],
        marker="o",
        linewidth=3,
        markersize=8,
        color="crimson",
    )

    plt.title(
        "Final Lateral Error Across Perturbation Conditions",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel("Perturbation Condition")
    plt.ylabel("Final Lateral Error")

    plt.xticks(rotation=20, ha="right")

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "final_error.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    print()
    print("=" * 70)
    print("Experiment 2 Robustness Summary")
    print("=" * 70)
    print(results)
    print("=" * 70)
    print(f"CSV Summary : {csv_file}")
    print(f"Figures      : {output_dir}")
    print("=" * 70)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()