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
1.0
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

        for column in [
            "reward",
            "steps",
            "path_length",
            "mean_speed",
            "max_speed"
        ]:

            values = summary[column].values

            results[f"{column}_mean"] = np.mean(values)
            results[f"{column}_sd"] = np.std(values, ddof=1)
            results[f"{column}_sem"] = (
                np.std(values, ddof=1) /
                np.sqrt(len(values))
            )

            ci95 = (
                1.96 *
                results[f"{column}_sem"]
            )

            results[f"{column}_ci95"] = ci95

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

        conditions = [
            "P0",
            "L1",
            "L2",
            "L3",
            "R1",
            "R2",
            "R3"
        ]

        rows = []

        for condition in conditions:

            stats = self.analyse_condition(condition)

            stats["condition"] = condition

            rows.append(stats)

        df = pd.DataFrame(rows)

        df.to_csv(

            self.root /
            "descriptive_statistics.csv",

            index=False
        )

        print(df)

        return df