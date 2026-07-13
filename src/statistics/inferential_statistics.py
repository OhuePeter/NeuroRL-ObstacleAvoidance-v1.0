"""
==========================================================
Inferential Statistics

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Performs one-way ANOVA and Tukey HSD for all
behavioural metrics.

Version:
2.0
==========================================================
"""

from pathlib import Path

import numpy as np
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

        self.metrics = [
            "reward",
            "steps",
            "path_length",
            "mean_speed",
            "peak_lateral_velocity",
            "max_heading_deviation",
            "final_lateral_error",
        ]

    def load_metric(self, metric):

        frames = []

        for condition in self.conditions:

            df = pd.read_csv(
                self.root /
                f"evaluation_{condition}" /
                "summary.csv"
            )

            df["condition"] = condition

            frames.append(
                df[["condition", metric]]
            )

        return pd.concat(frames, ignore_index=True)

    def analyse_metric(self, metric):

        df = self.load_metric(metric)

        groups = [
            df[df.condition == c][metric].values
            for c in self.conditions
        ]

        # Skip metrics with no variance
        if np.allclose(df[metric].std(), 0):

            print(f"Skipping {metric} (no variance).")

            return None, None

        F, p = f_oneway(*groups)

        grand_mean = df[metric].mean()

        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2
            for g in groups
        )

        ss_total = np.sum(
            (df[metric] - grand_mean) ** 2
        )

        eta2 = ss_between / ss_total

        anova = pd.DataFrame({
            "metric": [metric],
            "F": [F],
            "p": [p],
            "eta_squared": [eta2],
        })

        tukey = pairwise_tukeyhsd(
            endog=df[metric],
            groups=df["condition"],
            alpha=0.05,
        )

        tukey_df = pd.DataFrame(
            tukey.summary().data[1:],
            columns=tukey.summary().data[0],
        )

        anova.to_csv(
            self.root / f"{metric}_anova.csv",
            index=False,
        )

        tukey_df.to_csv(
            self.root / f"{metric}_tukey.csv",
            index=False,
        )

        print("\n" + "=" * 60)
        print(metric)
        print("=" * 60)
        print(anova)

        return anova, tukey_df

    def analyse_all(self):

        for metric in self.metrics:

            self.analyse_metric(metric)