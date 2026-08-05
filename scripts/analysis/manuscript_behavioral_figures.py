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


def _export_figure(fig, stem):

    outputs = []

    for suffix in [".png", ".pdf", ".svg"]:
        path = OUT / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)

    return outputs


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
                    "condition": condition,
                    "episode": episode,
                    "success": bool(row["success"]),
                    "collision": bool(row["collision"]),
                    "desired_route": row.get("desired_route", "either"),
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


def _pick_trials(trajectories, desired_route=None, success=None, n=3):

    selected = []

    for item in trajectories:
        if desired_route is not None and item.get("desired_route") != desired_route:
            continue
        if success is not None and bool(item["success"]) != bool(success):
            continue
        selected.append(item)
        if len(selected) >= n:
            break

    return selected


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
    ax.set_title("Rightward successes and failed right-perturbation trials")

    rightward_successes = (
        _pick_trials(trajectories.get("P0", []), desired_route="right", success=True, n=2)
        + _pick_trials(trajectories.get("R1", []), desired_route="right", success=True, n=2)
    )

    failed_right_perturbations = (
        _pick_trials(trajectories.get("R2", []), success=False, n=2)
        + _pick_trials(trajectories.get("R3", []), success=False, n=2)
    )

    for item in rightward_successes:
        path = item["xy"]
        color = PALETTE[item["condition"]]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.7, alpha=0.90)

    for item in failed_right_perturbations:
        path = item["xy"]
        failure_color = PALETTE[item["condition"]]
        ax.plot(path[:, 0], path[:, 1], color=failure_color, linewidth=1.7, linestyle="--", alpha=0.95)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=PALETTE["P0"], lw=2, label="P0 right-cue success"),
            plt.Line2D([0], [0], color=PALETTE["R1"], lw=2, label="R1 right-cue success"),
            plt.Line2D([0], [0], color=PALETTE["R2"], lw=2, linestyle="--", label="R2 failure"),
            plt.Line2D([0], [0], color=PALETTE["R3"], lw=2, linestyle="--", label="R3 failure"),
        ],
        frameon=False,
        loc="upper left",
    )
    _panel_label(ax, "C")

    outputs = _export_figure(fig, "figure2_behavioural_trajectories")
    plt.close(fig)

    return outputs


def figure3_performance(summary_df):

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    plot_specs = [
        ("reward", "Episode reward", "A"),
        ("steps", "Episode duration", "B"),
        ("path_length", "Path length", "C"),
        ("final_lateral_error", "Final lateral error", "D"),
    ]

    order = CONDITIONS

    condition_index = {condition: i for i, condition in enumerate(order)}
    summary_df = summary_df.copy()
    summary_df["condition_index"] = summary_df["condition"].map(condition_index)

    mean_df = (
        summary_df.groupby("condition", as_index=False)
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            reward_n=("reward", "size"),
            steps_mean=("steps", "mean"),
            steps_std=("steps", "std"),
            steps_n=("steps", "size"),
            path_length_mean=("path_length", "mean"),
            path_length_std=("path_length", "std"),
            path_length_n=("path_length", "size"),
            final_lateral_error_mean=("final_lateral_error", "mean"),
            final_lateral_error_std=("final_lateral_error", "std"),
            final_lateral_error_n=("final_lateral_error", "size"),
        )
    )
    mean_df["condition_index"] = mean_df["condition"].map(condition_index)

    for ax, (metric, title, label) in zip(axes.flatten(), plot_specs):
        x = mean_df["condition_index"].to_numpy()
        y = mean_df[f"{metric}_mean"].to_numpy()
        std = mean_df[f"{metric}_std"].to_numpy()
        n = mean_df[f"{metric}_n"].to_numpy()
        ci = 1.96 * (std / np.sqrt(n))

        for condition in order:
            cdf = summary_df[summary_df["condition"] == condition]
            ax.scatter(
                cdf["condition_index"],
                cdf[metric],
                s=10,
                alpha=0.35,
                color=PALETTE[condition],
                linewidths=0,
            )

        ax.plot(
            x,
            y,
            color="#1D3557",
            linewidth=2.2,
            marker="o",
            markersize=4,
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            color="#1D3557",
            alpha=0.16,
            linewidth=0,
        )

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=0)
        _panel_label(ax, label)

    outputs = _export_figure(fig, "figure3_behavioural_performance")
    plt.close(fig)

    return outputs


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

    outputs = _export_figure(fig, "figure4_behavioural_adaptation")
    plt.close(fig)

    return outputs


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
