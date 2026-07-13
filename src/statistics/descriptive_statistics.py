"""
==========================================================
Descriptive Statistics

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Computes descriptive statistics across all
evaluation conditions.

Version:
2.0
==========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


class DescriptiveStatistics:

    def __init__(self, results_root):

        self.root = Path(results_root)

    def analyse_condition(self, condition):

        summary = pd.read_csv(
            self.root /
            f"evaluation_{condition}" /
            "summary.csv"
        )

        results = {}

        # --------------------------------------------
        # Metrics to analyse
        # --------------------------------------------
        metrics = [
            "reward",
            "steps",
            "path_length",
            "mean_speed",
            "max_speed",
            "peak_lateral_velocity",
            "max_heading_deviation",
            "final_lateral_error",
        ]

        for column in metrics:

            values = summary[column].values

            results[f"{column}_mean"] = np.mean(values)

            results[f"{column}_sd"] = np.std(
                values,
                ddof=1
            )

            results[f"{column}_sem"] = (
                results[f"{column}_sd"] /
                np.sqrt(len(values))
            )

            ci95 = (
                1.96 *
                results[f"{column}_sem"]
            )

            results[f"{column}_ci95"] = ci95

            results[f"{column}_min"] = np.min(values)

            results[f"{column}_max"] = np.max(values)

        results["success_rate"] = (
            100 *
            summary["success"].mean()
        )

        results["collision_rate"] = (
            100 *
            summary["collision"].mean()
        )

        return results

    def analyse_all(self):

        conditions = "P0"

        rows = []

        for condition in conditions:

            stats = self.analyse_condition(
                condition
            )

            stats["condition"] = condition

            rows.append(stats)

        df = pd.DataFrame(rows)

        df = df[
            ["condition"] +
            [c for c in df.columns if c != "condition"]
        ]

        output = (
            self.root /
            "descriptive_statistics.csv"
        )

        df.to_csv(
            output,
            index=False
        )

        print("\n" + "=" * 70)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 70)
        print(df)

        print(
            f"\nSaved to:\n{output}"
        )

        return df