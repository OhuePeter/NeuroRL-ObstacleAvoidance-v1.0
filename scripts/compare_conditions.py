"""
==========================================================
Compare Experimental Conditions

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Loads evaluation summaries from all experimental
conditions and produces a comparison table.

Version:
1.0
==========================================================
"""

from pathlib import Path

import pandas as pd


CONDITIONS = [
    "P0",
    "L1",
    "L2",
    "L3",
    "R1",
    "R2",
    "R3",
]


def main():

    base_dir = Path(
        "experiments/version_1_0/results"
    )

    results = []

    for condition in CONDITIONS:

        summary_file = (
            base_dir /
            f"evaluation_{condition}" /
            "summary.csv"
        )

        if not summary_file.exists():
            print(f"Skipping {condition} (not found)")
            continue

        df = pd.read_csv(summary_file)

        results.append({
            "Condition": condition,
            "Success Rate (%)": 100 * df["success"].mean(),
            "Collision Rate (%)": 100 * df["collision"].mean(),
            "Mean Reward": df["reward"].mean(),
            "Mean Steps": df["steps"].mean(),
            "Mean Path Length": df["path_length"].mean(),
            "Mean Speed": df["mean_speed"].mean(),
            "Peak Speed": df["max_speed"].mean(),
        })

    comparison = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("EXPERIMENTAL COMPARISON")
    print("=" * 70)
    print(comparison)

    output_dir = Path(
        "experiments/version_1_0/results"
    )

    comparison.to_csv(
        output_dir / "condition_comparison.csv",
        index=False,
    )

    print("\nSaved:")
    print(output_dir / "condition_comparison.csv")


if __name__ == "__main__":
    main()