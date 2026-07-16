"""
==========================================================
Manuscript Behavioural Figures
Experiment 2

Description
-----------
Builds manuscript-ready behavioural figures to fill the
remaining main-text figure slots in the manuscript.

Outputs
-------
paper/figures/
  - figure2_behavioural_trajectories.png/.pdf
  - figure3_behavioural_performance.png/.pdf
  - figure4_behavioural_adaptation.png/.pdf
  - behavioural_figure_manifest.csv

==========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_ROOT = Path("experiments/version_2_0/results")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P0", "L1", "L2", "L3", "R1", "R2", "R3"]
CONDITION_LABELS = {
    "P0": "Control",
    "L1": "Small Left",
    "L2": "Medium Left",
    "L3": "Large Left",
    "R1": "Small Right",
    "R2": "Medium Right",
    "R3": "Large Right",
}
PALETTE = {
    "P0": "#264653",
    "L1": "#2A9D8F",
    "L2": "#52B788",
    "L3": "#95D5B2",
    "R1": "#E9C46A",
    "R2": "#F4A261",
    "R3": "#E76F51",
}
OBSTACLE = (5.0, 3.8)
START = (5.0, 1.0)
GOAL = (5.0, 6.6)


def set_style():

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 600,
            "figure.dpi": 300,
        }
    )


def _panel_label(ax, label):

    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def load_summary_data():

    frames = []

    for condition in CONDITIONS:
        summary_file = RESULTS_ROOT / f"evaluation_{condition}" / "summary.csv"
        if not summary_file.exists():
            continue
        df = pd.read_csv(summary_file)
        df["condition"] = condition
        df["condition_label"] = CONDITION_LABELS[condition]
        frames.append(df)

    if not frames:
        raise RuntimeError("No behavioural summaries found.")

    return pd.concat(frames, ignore_index=True)


def load_trajectories():

    trajectories = {}

    for condition in CONDITIONS:
        condition_dir = RESULTS_ROOT / f"evaluation_{condition}"
        summary_file = condition_dir / "summary.csv"
        if not summary_file.exists():
            continue

        summary = pd.read_csv(summary_file).set_index("episode")
        condition_trajectories = []

        for path in sorted(condition_dir.glob("trajectory_*.csv")):
            episode = int(path.stem.split("_")[1])
            df = pd.read_csv(path)
            if df.empty or episode not in summary.index:
                continue

            row = summary.loc[episode]
            condition_trajectories.append(
                {
                    "episode": episode,
                    "success": bool(row["success"]),
                    "collision": bool(row["collision"]),
                    "xy": df[["x", "y"]].to_numpy(),
                }
            )

        trajectories[condition] = condition_trajectories

    return trajectories


def _interpolate_path(path, n_points=200):

    if len(path) < 2:
        return np.repeat(path, n_points, axis=0)

    src_t = np.linspace(0.0, 1.0, len(path))
    dst_t = np.linspace(0.0, 1.0, n_points)
    x = np.interp(dst_t, src_t, path[:, 0])
    y = np.interp(dst_t, src_t, path[:, 1])
    return np.column_stack([x, y])


def _mean_path(trajectories, success_only=True):

    paths = []

    for item in trajectories:
        if success_only and not item["success"]:
            continue
        paths.append(_interpolate_path(item["xy"]))

    if not paths:
        return None

    stack = np.stack(paths, axis=0)
    return stack.mean(axis=0)


def _draw_workspace(ax):

    obstacle = plt.Circle(OBSTACLE, 0.45, facecolor="#D8DEE9", edgecolor="#222222", linewidth=1.0)
    goal = plt.Circle(GOAL, 0.15, facecolor="white", edgecolor="#2A9D8F", linewidth=1.5)

    ax.add_patch(obstacle)
    ax.add_patch(goal)
    ax.scatter([START[0]], [START[1]], color="#222222", s=18, zorder=5)

    ax.text(START[0], START[1] - 0.28, "Start", ha="center", fontsize=8)
    ax.text(GOAL[0], GOAL[1] + 0.22, "Goal", ha="center", fontsize=8)
    ax.text(OBSTACLE[0], OBSTACLE[1] - 0.70, "Obstacle", ha="center", fontsize=8)

    ax.set_xlim(2.6, 7.4)
    ax.set_ylim(0.7, 6.95)
    ax.set_aspect("equal")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")


def figure2_trajectories(summary_df, trajectories):

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), constrained_layout=True)

    left_conditions = ["P0", "L1", "L2", "L3"]
    right_conditions = ["P0", "R1", "R2", "R3"]

    for ax, conds, title, label in [
        (axes[0], left_conditions, "Leftward perturbations", "A"),
        (axes[1], right_conditions, "Rightward perturbations", "B"),
    ]:
        _draw_workspace(ax)
        for condition in conds:
            mean_path = _mean_path(trajectories.get(condition, []), success_only=True)
            if mean_path is None:
                continue
            ax.plot(
                mean_path[:, 0],
                mean_path[:, 1],
                color=PALETTE[condition],
                linewidth=2.0,
                label=condition,
            )
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper left")
        _panel_label(ax, label)

    ax = axes[2]
    _draw_workspace(ax)
    ax.set_title("Large-right representative trials")
    r3_trials = trajectories.get("R3", [])

    successes = [item for item in r3_trials if item["success"]][:3]
    failures = [item for item in r3_trials if not item["success"]][:3]

    for item in successes:
        path = item["xy"]
        ax.plot(path[:, 0], path[:, 1], color=PALETTE["R3"], linewidth=1.6, alpha=0.85)

    for item in failures:
        path = item["xy"]
        ax.plot(path[:, 0], path[:, 1], color="#6D597A", linewidth=1.6, linestyle="--", alpha=0.9)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=PALETTE["R3"], lw=2, label="Success"),
            plt.Line2D([0], [0], color="#6D597A", lw=2, linestyle="--", label="Failure"),
        ],
        frameon=False,
        loc="upper left",
    )
    _panel_label(ax, "C")

    png = OUT / "figure2_behavioural_trajectories.png"
    pdf = OUT / "figure2_behavioural_trajectories.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return [png, pdf]


def figure3_performance(summary_df):

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    plot_specs = [
        ("reward", "Episode reward", "A"),
        ("steps", "Episode duration", "B"),
        ("path_length", "Path length", "C"),
        ("final_lateral_error", "Final lateral error", "D"),
    ]

    order = CONDITIONS
    palette = [PALETTE[c] for c in order]

    for ax, (metric, title, label) in zip(axes.flatten(), plot_specs):
        sns.boxplot(
            data=summary_df,
            x="condition",
            y=metric,
            hue="condition",
            order=order,
            palette=palette,
            linewidth=1.2,
            fliersize=0,
            dodge=False,
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        sns.stripplot(
            data=summary_df,
            x="condition",
            y=metric,
            order=order,
            color="black",
            size=2.5,
            alpha=0.45,
            jitter=0.18,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=0)
        _panel_label(ax, label)

    png = OUT / "figure3_behavioural_performance.png"
    pdf = OUT / "figure3_behavioural_performance.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return [png, pdf]


def figure4_adaptation(summary_df):

    condition_stats = (
        summary_df.groupby("condition", as_index=False)
        .agg(
            success_rate=("success", "mean"),
            collision_rate=("collision", "mean"),
            peak_lateral_velocity=("peak_lateral_velocity", "mean"),
            max_heading_deviation=("max_heading_deviation", "mean"),
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)

    panels = [
        ("success_rate", "Success rate", "A", "#2A9D8F"),
        ("collision_rate", "Collision rate", "B", "#C8553D"),
        ("peak_lateral_velocity", "Peak lateral velocity", "C", "#457B9D"),
        ("max_heading_deviation", "Maximum heading deviation", "D", "#E07A2F"),
    ]

    for ax, (metric, title, label, color) in zip(axes.flatten(), panels):
        sns.barplot(
            data=condition_stats,
            x="condition",
            y=metric,
            order=CONDITIONS,
            color=color,
            ax=ax,
        )
        if metric in {"success_rate", "collision_rate"}:
            ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        _panel_label(ax, label)

    png = OUT / "figure4_behavioural_adaptation.png"
    pdf = OUT / "figure4_behavioural_adaptation.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return [png, pdf]


def save_manifest(paths):

    manifest = pd.DataFrame({"file": [str(p) for p in paths], "exists": [p.exists() for p in paths]})
    manifest_path = OUT / "behavioural_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def main():

    print("=" * 70)
    print("Generating manuscript behavioural figures")
    print("=" * 70)

    set_style()
    summary_df = load_summary_data()
    trajectories = load_trajectories()

    outputs = []
    outputs += figure2_trajectories(summary_df, trajectories)
    outputs += figure3_performance(summary_df)
    outputs += figure4_adaptation(summary_df)

    manifest = save_manifest(outputs)

    print()
    print("=" * 70)
    print("MANUSCRIPT BEHAVIOURAL FIGURES COMPLETE")
    print("=" * 70)
    for path in outputs:
        print(path)
    print(manifest)


if __name__ == "__main__":
    main()
