"""
==========================================================
Inferential Statistics

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Performs one-way ANOVA and Tukey HSD comparisons
across perturbation conditions.

Version:
1.0
==========================================================
"""

from pathlib import Path

import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class InferentialStatistics:

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

    def load_metric(self, metric):

        data = []

        for condition in self.conditions:

            df = pd.read_csv(
                self.root /
                f"evaluation_{condition}" /
                "summary.csv"
            )

            df["condition"] = condition

            data.append(
                df[["condition", metric]]
            )

        return pd.concat(data, ignore_index=True)

    def analyse_metric(self, metric):

        df = self.load_metric(metric)

        groups = [
            df[df.condition == c][metric]
            for c in self.conditions
        ]

        F, p = f_oneway(*groups)

        print("=" * 60)
        print(metric)
        print("=" * 60)
        print(f"ANOVA F = {F:.4f}")
        print(f"p-value  = {p:.6f}")

        tukey = pairwise_tukeyhsd(
            endog=df[metric],
            groups=df["condition"],
            alpha=0.05
        )

        print()
        print(tukey)

        with open(
            self.root / f"{metric}_tukey.txt",
            "w"
        ) as f:

            f.write(str(tukey))