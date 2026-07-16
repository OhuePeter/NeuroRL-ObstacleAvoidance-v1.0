"""
==========================================================
Manuscript Statistical Tables
Experiment 2

Description
-----------
Builds manuscript-ready statistical tables for Experiment 2
from behavioural summary files and previously generated
statistics outputs.

Outputs
-------
paper/tables/
  - table1_descriptive_statistics.tex
  - table2_assumption_tests.tex
  - table3_omnibus_tests.tex
  - table4_pairwise_posthoc.tex
  - table_manifest.csv

==========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

RESULTS_ROOT = Path("experiments/version_2_0/results")
STATS_DIR = RESULTS_ROOT / "statistics"
TABLE_DIR = Path("paper/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P0", "L1", "L2", "L3", "R1", "R2", "R3"]
CONDITION_NAMES = {
    "P0": "Control",
    "L1": "Small Left",
    "L2": "Medium Left",
    "L3": "Large Left",
    "R1": "Small Right",
    "R2": "Medium Right",
    "R3": "Large Right",
}
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


def _escape(text):

    return str(text).replace("_", r"\_")


def _format_p(p):

    if pd.isna(p):
        return "--"

    if p < 0.001:
        return "< 0.001"

    return f"{p:.3f}"


def _format_num(value, digits=3):

    if pd.isna(value):
        return "--"

    return f"{value:.{digits}f}"


def _latex_table(df, column_format, caption, label):

    header = " & ".join(_escape(col) for col in df.columns) + r" \\" 
    body = "\n".join(
        " & ".join(_escape(v) for v in row) + r" \\" 
        for row in df.astype(str).values.tolist()
    )

    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{column_format}}}\n"
        "\\hline\n"
        f"{header}\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def load_raw_data():

    frames = []

    for condition in CONDITIONS:
        filename = RESULTS_ROOT / f"evaluation_{condition}" / "summary.csv"
        if not filename.exists():
            continue
        df = pd.read_csv(filename)
        df["condition"] = condition
        df["condition_name"] = CONDITION_NAMES[condition]
        frames.append(df)

    if not frames:
        raise RuntimeError("No summary.csv files found for Experiment 2.")

    return pd.concat(frames, ignore_index=True)


def load_statistics_outputs():

    required = {
        "descriptive": STATS_DIR / "descriptive_statistics.csv",
        "assumptions": STATS_DIR / "assumption_tests.csv",
        "omnibus": STATS_DIR / "anova_results.csv",
    }

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing statistical output files:\n{joined}")

    return {
        name: pd.read_csv(path)
        for name, path in required.items()
    }


def build_descriptive_table(desc):

    rows = []

    for variable_name in VARIABLES.values():
        subset = desc[desc["Variable"] == variable_name]

        row = {"Variable": variable_name}

        for condition in CONDITION_NAMES.values():
            item = subset[subset["Condition"] == condition].iloc[0]
            row[condition] = (
                f"{item['Mean']:.3f} ± {item['Std']:.3f} "
                f"[{item['CI Lower']:.3f}, {item['CI Upper']:.3f}]"
            )

        rows.append(row)

    table = pd.DataFrame(rows)

    tex = _latex_table(
        table,
        column_format="l" + "p{2.5cm}" * len(CONDITION_NAMES),
        caption=(
            "Descriptive statistics for behavioural outcome measures across "
            "perturbation conditions. Entries report mean $\\pm$ standard deviation "
            "with 95\\% confidence intervals in brackets."
        ),
        label="tab:descriptive_statistics",
    )

    return table, tex


def build_assumptions_table(assumptions):

    table = assumptions.copy()
    table["Shapiro p"] = table["Shapiro p"].map(_format_p)
    table["Levene p"] = table["Levene p"].map(_format_p)
    table["Normal"] = table["Normal"].map(lambda x: "Yes" if bool(x) else "No")
    table["Equal Variance"] = table["Equal Variance"].map(lambda x: "Yes" if bool(x) else "No")

    tex = _latex_table(
        table,
        column_format="lcccc",
        caption=(
            "Assumption checks for parametric testing. Shapiro-Wilk p-values summarize "
            "the minimum normality evidence across perturbation groups for each measure, "
            "and Levene's test evaluates homogeneity of variance."
        ),
        label="tab:assumption_tests",
    )

    return table, tex


def build_omnibus_table(omnibus):

    table = omnibus.copy()
    table["Statistic"] = table["Statistic"].map(lambda x: _format_num(x, 3))
    table["p-value"] = table["p-value"].map(_format_p)
    table["Effect Size (Eta²)"] = table["Effect Size (Eta²)"].map(lambda x: _format_num(x, 3))
    table["Significant"] = table["Significant"].map(lambda x: "Yes" if bool(x) else "No")

    tex = _latex_table(
        table,
        column_format="l l c c c c",
        caption=(
            "Omnibus inferential tests across perturbation conditions. "
            "Kruskal-Wallis tests were used when parametric assumptions were violated; "
            "the maximum-speed measure was invariant across all conditions."
        ),
        label="tab:omnibus_tests",
    )

    return table, tex


def build_pairwise_posthoc(raw_data, omnibus):

    significant_variables = []
    for key, label in VARIABLES.items():
        row = omnibus[omnibus["Variable"] == label]
        if row.empty:
            continue
        if bool(row.iloc[0]["Significant"]):
            significant_variables.append((key, label))

    rows = []

    for variable_key, variable_label in significant_variables:
        tests = []

        for index_a, condition_a in enumerate(CONDITIONS):
            for condition_b in CONDITIONS[index_a + 1:]:
                values_a = raw_data[raw_data["condition"] == condition_a][variable_key].values
                values_b = raw_data[raw_data["condition"] == condition_b][variable_key].values

                if np.all(values_a == values_a[0]) and np.all(values_b == values_b[0]) and values_a[0] == values_b[0]:
                    statistic = np.nan
                    p_value = 1.0
                    rbc = 0.0
                else:
                    statistic, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
                    rbc = (2.0 * statistic / (len(values_a) * len(values_b))) - 1.0

                tests.append(
                    {
                        "Variable": variable_label,
                        "Comparison": f"{condition_a} vs {condition_b}",
                        "U": statistic,
                        "p_raw": p_value,
                        "Rank-biserial r": rbc,
                    }
                )

        if tests:
            corrected = multipletests(
                [item["p_raw"] for item in tests],
                method="holm",
            )

            for item, p_adj, reject in zip(tests, corrected[1], corrected[0]):
                item["p_adj"] = p_adj
                item["Significant"] = bool(reject)
                rows.append(item)

    table = pd.DataFrame(rows)

    if not table.empty:
        table["U"] = table["U"].map(lambda x: _format_num(x, 1))
        table["p_raw"] = table["p_raw"].map(_format_p)
        table["p_adj"] = table["p_adj"].map(_format_p)
        table["Rank-biserial r"] = table["Rank-biserial r"].map(lambda x: _format_num(x, 3))
        table["Significant"] = table["Significant"].map(lambda x: "Yes" if bool(x) else "No")

    tex = _latex_table(
        table,
        column_format="l l c c c c c",
        caption=(
            "Pairwise post hoc comparisons for outcome measures with significant omnibus effects. "
            "Two-sided Mann-Whitney U tests are Holm-corrected within each behavioural variable, "
            "and rank-biserial correlation reports effect direction and magnitude."
        ),
        label="tab:pairwise_posthoc",
    )

    return table, tex


def write_outputs(outputs):

    manifest_rows = []

    for file_name, table_df, tex_text in outputs:
        tex_path = TABLE_DIR / f"{file_name}.tex"
        csv_path = TABLE_DIR / f"{file_name}.csv"

        tex_path.write_text(tex_text, encoding="utf-8")
        table_df.to_csv(csv_path, index=False)

        manifest_rows.append({"file": str(tex_path), "type": "tex"})
        manifest_rows.append({"file": str(csv_path), "type": "csv"})

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = TABLE_DIR / "table_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return manifest_path


def main():

    print("=" * 70)
    print("Generating manuscript statistical tables")
    print("=" * 70)

    raw_data = load_raw_data()
    stats = load_statistics_outputs()

    outputs = [
        ("table1_descriptive_statistics",) + build_descriptive_table(stats["descriptive"]),
        ("table2_assumption_tests",) + build_assumptions_table(stats["assumptions"]),
        ("table3_omnibus_tests",) + build_omnibus_table(stats["omnibus"]),
        ("table4_pairwise_posthoc",) + build_pairwise_posthoc(raw_data, stats["omnibus"]),
    ]

    manifest_path = write_outputs(outputs)

    print()
    print("=" * 70)
    print("MANUSCRIPT TABLES COMPLETE")
    print("=" * 70)
    for file_name, _, _ in outputs:
        print(TABLE_DIR / f"{file_name}.tex")
        print(TABLE_DIR / f"{file_name}.csv")
    print(manifest_path)


if __name__ == "__main__":
    main()
