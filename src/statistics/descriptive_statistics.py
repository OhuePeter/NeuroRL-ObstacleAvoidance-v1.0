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
2.1
==========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


class DescriptiveStatistics:

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

        self.metrics = [
            "reward",
            "steps",
            "path_length",
            "mean_speed",
            "max_speed",
            "peak_lateral_velocity",
            "max_heading_deviation",
            "final_lateral_error",
        ]

    def analyse_condition(self, condition):

        summary = pd.read_csv(
            self.root /
            f"evaluation_{condition}" /
            "summary.csv"
        )

        results = {
            "condition": condition
        }

        for metric in self.metrics:

            values = summary[metric].values

            results[f"{metric}_mean"] = np.mean(values)
            results[f"{metric}_sd"] = np.std(values, ddof=1)
            results[f"{metric}_sem"] = (
                results[f"{metric}_sd"] /
                np.sqrt(len(values))
            )

            results[f"{metric}_ci95"] = (
                1.96 *
                results[f"{metric}_sem"]
            )

            results[f"{metric}_min"] = np.min(values)
            results[f"{metric}_max"] = np.max(values)

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

        rows = []

        for condition in self.conditions:

            rows.append(
                self.analyse_condition(condition)
            )

        df = pd.DataFrame(rows)

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

        print(f"\nSaved to:\n{output}")

        return df