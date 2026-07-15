"""
==========================================================
Statistical Analysis

Experiment 2

Authors:
Peter Ohue
Gunnar Blohm

Description
"""

from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import (
    shapiro,
    levene,
    sem,
    t,
    f_oneway,
    kruskal,
)

from statsmodels.stats.multicomp import (
    pairwise_tukeyhsd,
)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from openpyxl import Workbook

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path("experiments/version_2_0/results")

OUTPUT_DIR = ROOT / "statistics"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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
    "L1": "Small Left",
    "L2": "Medium Left",
    "L3": "Large Left",
    "R1": "Small Right",
    "R2": "Medium Right",
    "R3": "Large Right",
}

# ---------------------------------------------------------
# Behavioural variables
# ---------------------------------------------------------

VARIABLES = {

    "reward":
        "Episode Reward",

    "steps":
        "Episode Duration",

    "path_length":
        "Path Length",

    "mean_speed":
        "Mean Speed",

    "max_speed":
        "Maximum Speed",

    "peak_lateral_velocity":
        "Peak Lateral Velocity",

    "max_heading_deviation":
        "Maximum Heading Deviation",

    "final_lateral_error":
        "Final Lateral Error",

}

# ---------------------------------------------------------
# Load all summary.csv files
# ---------------------------------------------------------

def load_data():

    frames = []

    print("=" * 70)
    print("Loading behavioural summaries")
    print("=" * 70)

    for condition in CONDITIONS:

        filename = (
            ROOT
            / f"evaluation_{condition}"
            / "summary.csv"
        )

        if not filename.exists():

            print(f"Missing: {filename}")

            continue

        df = pd.read_csv(filename)

        df["condition"] = condition
        df["condition_name"] = CONDITION_NAMES[condition]

        frames.append(df)

        print(
            f"{condition:<3}"
            f" {len(df):>3} episodes"
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print(
        f"Loaded {len(data)} episodes."
    )

    return data

# ---------------------------------------------------------
# 95% confidence interval
# ---------------------------------------------------------

def confidence_interval(values):

    values = np.asarray(values)

    n = len(values)

    mean = np.mean(values)

    interval = sem(values) * t.ppf(
        0.975,
        n - 1,
    )

    return mean - interval, mean + interval

# ---------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------

def descriptive_statistics(data):

    print()
    print("=" * 70)
    print("Computing descriptive statistics")
    print("=" * 70)

    rows = []

    for condition in CONDITIONS:

        subset = data[
            data["condition"] == condition
        ]

        for variable in VARIABLES:

            values = subset[variable]

            ci_low, ci_high = (
                confidence_interval(values)
            )

            rows.append({

                "Condition":
                    CONDITION_NAMES[condition],

                "Variable":
                    VARIABLES[variable],

                "N":
                    len(values),

                "Mean":
                    values.mean(),

                "Median":
                    values.median(),

                "Std":
                    values.std(),

                "Minimum":
                    values.min(),

                "Maximum":
                    values.max(),

                "IQR":
                    values.quantile(0.75)
                    -
                    values.quantile(0.25),

                "CI Lower":
                    ci_low,

                "CI Upper":
                    ci_high,

            })

    statistics = pd.DataFrame(rows)

    statistics.to_csv(

        OUTPUT_DIR /
        "descriptive_statistics.csv",

        index=False,

    )

    print(
        "Saved descriptive_statistics.csv"
    )

    return statistics

# ---------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------

def confidence_intervals(data):

    rows = []

    for condition in CONDITIONS:

        subset = data[
            data["condition"] == condition
        ]

        for variable in VARIABLES:

            low, high = confidence_interval(
                subset[variable]
            )

            rows.append({

                "Condition":
                    CONDITION_NAMES[condition],

                "Variable":
                    VARIABLES[variable],

                "Lower":
                    low,

                "Upper":
                    high,

            })

    ci = pd.DataFrame(rows)

    ci.to_csv(

        OUTPUT_DIR /
        "confidence_intervals.csv",

        index=False,

    )

    print(
        "Saved confidence_intervals.csv"
    )

    return ci

# ---------------------------------------------------------
# Statistical assumptions
# ---------------------------------------------------------

def assumption_tests(data):

    print()
    print("=" * 70)
    print("Checking statistical assumptions")
    print("=" * 70)

    rows = []

    for variable in VARIABLES:

        groups = [

            data[
                data["condition"] == c
            ][variable]

            for c in CONDITIONS

        ]

        # -----------------------------
        # Shapiro-Wilk
        # -----------------------------

        shapiro_p = min(

            shapiro(group)[1]

            for group in groups

        )

        # -----------------------------
        # Levene
        # -----------------------------

        levene_p = levene(
            *groups
        ).pvalue

        rows.append({

            "Variable":
                VARIABLES[variable],

            "Shapiro p":
                shapiro_p,

            "Normal":
                shapiro_p > 0.05,

            "Levene p":
                levene_p,

            "Equal Variance":
                levene_p > 0.05,

        })

    assumptions = pd.DataFrame(rows)

    assumptions.to_csv(

        OUTPUT_DIR /
        "assumption_tests.csv",

        index=False,

    )

    print(
        "Saved assumption_tests.csv"
    )

    return assumptions

# ---------------------------------------------------------
# Inferential Statistics
# ---------------------------------------------------------

def inferential_statistics(data, assumptions):
    """
    Performs either One-way ANOVA or
    Kruskal-Wallis depending on whether
    the assumptions are satisfied.
    """

    print()
    print("=" * 70)
    print("Inferential Statistics")
    print("=" * 70)

    rows = []

    for variable in VARIABLES:

        groups = [
            data[data["condition"] == c][variable]
            for c in CONDITIONS
        ]

        assumption = assumptions[
            assumptions["Variable"] == VARIABLES[variable]
        ].iloc[0]

        use_anova = (
            assumption["Normal"]
            and
            assumption["Equal Variance"]
        )

        pooled = np.concatenate([
            np.asarray(g)
            for g in groups
        ])

        if np.all(pooled == pooled[0]):

            statistic = 0.0
            p = 1.0
            test = "No-variance (all values identical)"
            eta2 = np.nan

            rows.append({

                "Variable": VARIABLES[variable],

                "Test": test,

                "Statistic": statistic,

                "p-value": p,

                "Effect Size (Eta²)": eta2,

                "Significant": False,

            })

            continue

        if use_anova:

            statistic, p = f_oneway(*groups)

            test = "One-way ANOVA"

            grand_mean = data[variable].mean()

            ss_between = sum(
                len(g) * (g.mean() - grand_mean) ** 2
                for g in groups
            )

            ss_total = np.sum(
                (data[variable] - grand_mean) ** 2
            )

            eta2 = ss_between / ss_total

        else:

            statistic, p = kruskal(*groups)

            test = "Kruskal-Wallis"

            eta2 = np.nan

        rows.append({

            "Variable": VARIABLES[variable],

            "Test": test,

            "Statistic": statistic,

            "p-value": p,

            "Effect Size (Eta²)": eta2,

            "Significant": p < 0.05,

        })

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_DIR / "anova_results.csv",
        index=False,
    )

    print("Saved anova_results.csv")

    return results

# ---------------------------------------------------------
# Tukey HSD
# ---------------------------------------------------------

def tukey_analysis(data, assumptions):

    print()
    print("=" * 70)
    print("Tukey HSD")
    print("=" * 70)

    tables = []

    for variable in VARIABLES:

        assumption = assumptions[
            assumptions["Variable"] == VARIABLES[variable]
        ].iloc[0]

        if not (
            assumption["Normal"]
            and
            assumption["Equal Variance"]
        ):
            continue

        tukey = pairwise_tukeyhsd(

            endog=data[variable],

            groups=data["condition_name"],

            alpha=0.05,

        )

        table = pd.DataFrame(

            data=tukey.summary().data[1:],

            columns=tukey.summary().data[0],

        )

        table.insert(
            0,
            "Variable",
            VARIABLES[variable],
        )

        tables.append(table)

    if len(tables):

        results = pd.concat(
            tables,
            ignore_index=True,
        )

    else:

        results = pd.DataFrame()

    results.to_csv(
        OUTPUT_DIR / "tukey_hsd_results.csv",
        index=False,
    )

    print("Saved tukey_hsd_results.csv")

    return results

# ---------------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------------

def correlation_matrix(data):

    print()
    print("=" * 70)
    print("Correlation Matrix")
    print("=" * 70)

    columns = list(VARIABLES.keys())

    corr = data[columns].corr()

    corr.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv"
    )

    print("Saved correlation_matrix.csv")

    return corr



def main():

    data = load_data()

    descriptive = descriptive_statistics(data)

    ci = confidence_intervals(data)

    assumptions = assumption_tests(data)

    inferential = inferential_statistics(
        data,
        assumptions,
    )

    tukey = tukey_analysis(
        data,
        assumptions,
    )

    correlation = correlation_matrix(data)

    print()

    print("=" * 70)
    print("PART 2 COMPLETE")
    print("=" * 70)

    print("Generated")

    print("✓ descriptive_statistics.csv")

    print("✓ confidence_intervals.csv")

    print("✓ assumption_tests.csv")

    print("✓ anova_results.csv")

    print("✓ tukey_hsd_results.csv")

    print("✓ correlation_matrix.csv")

    print("=" * 70)


if __name__ == "__main__":
    main()