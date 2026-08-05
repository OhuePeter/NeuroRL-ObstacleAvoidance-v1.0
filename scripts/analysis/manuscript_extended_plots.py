"""
==========================================================
Manuscript Extended Plots
Experiment 2

Description
-----------
Builds publication-ready extended plots emphasizing:
- 7-panel condition trajectories (success/failure)
- velocity change profiles with perturbation direction
- PCA/tuning/correlation panels
- learning curve and success prediction

All figures are exported as PNG/PDF/SVG for Inkscape.
==========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

RESULTS_ROOT = Path("experiments/version_2_0/results")
FIG_OUT = Path("paper/figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

NEURAL_ROOT = RESULTS_ROOT / "neural_analysis"
EVAL_NPZ = Path("experiments/version_1_0/evaluation/evaluations.npz")

CONDITIONS = ["P0", "L1", "L2", "L3", "R1", "R2", "R3"]
LABELS = {
    "P0": "No perturbation",
    "L1": "Small left",
    "L2": "Medium left",
    "L3": "Large left",
    "R1": "Small right",
    "R2": "Medium right",
    "R3": "Large right",
}
PALETTE = {
    "P0": "#355070",
    "L1": "#6D597A",
    "L2": "#B56576",
    "L3": "#E56B6F",
    "R1": "#2A9D8F",
    "R2": "#1D7874",
    "R3": "#0B4F4A",
}

START = (5.0, 1.0)
GOAL = (5.0, 9.0)
OBSTACLES = [(4.25, 5.0), (5.75, 5.0)]
GOAL_RADIUS = 0.35
OBSTACLE_RADIUS = 0.50


def set_style():
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 600,
            "figure.dpi": 300,
        }
    )


def export_figure(fig, stem):
    paths = []
    for ext in [".png", ".pdf", ".svg"]:
        path = FIG_OUT / f"{stem}{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def load_condition_summary(condition):
    path = RESULTS_ROOT / f"evaluation_{condition}" / "summary.csv"
    return pd.read_csv(path)


def draw_workspace(ax):
    goal = plt.Circle(GOAL, GOAL_RADIUS, color="#3A86FF", alpha=0.25)
    ax.add_patch(goal)

    for obstacle in OBSTACLES:
        circle = plt.Circle(obstacle, OBSTACLE_RADIUS, color="black", alpha=0.25)
        ax.add_patch(circle)

    ax.scatter([START[0]], [START[1]], c="black", s=28, zorder=10)
    ax.text(START[0], START[1] - 0.30, "Start", ha="center", fontsize=7)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)


def perturbation_arrow(ax, condition):
    if condition.startswith("L"):
        ax.annotate(
            "Left push",
            xy=(5.7, 5.8),
            xytext=(7.3, 5.8),
            arrowprops=dict(arrowstyle="->", lw=1.3, color="#B56576"),
            color="#B56576",
            fontsize=7,
            va="center",
        )
    elif condition.startswith("R"):
        ax.annotate(
            "Right push",
            xy=(4.3, 5.8),
            xytext=(2.7, 5.8),
            arrowprops=dict(arrowstyle="->", lw=1.3, color="#1D7874"),
            color="#1D7874",
            fontsize=7,
            va="center",
        )


def figure5_trajectories_7panel():
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    flat_axes = axes.flatten()

    for i, condition in enumerate(CONDITIONS):
        ax = flat_axes[i]
        draw_workspace(ax)
        perturbation_arrow(ax, condition)

        condition_dir = RESULTS_ROOT / f"evaluation_{condition}"
        summary = load_condition_summary(condition).set_index("episode")

        successes = 0
        failures = 0

        for trajectory_file in sorted(condition_dir.glob("trajectory_*.csv")):
            episode = int(trajectory_file.stem.split("_")[1])
            if episode not in summary.index:
                continue

            trajectory = pd.read_csv(trajectory_file)
            success = bool(summary.loc[episode, "success"])

            if success:
                color = "#2A9D8F"
                alpha = 0.28
                lw = 1.2
                successes += 1
            else:
                color = "#C1121F"
                alpha = 0.85
                lw = 2.0
                failures += 1

            ax.plot(
                trajectory["x"],
                trajectory["y"],
                color=color,
                alpha=alpha,
                linewidth=lw,
            )

        ax.set_title(f"{condition} ({LABELS[condition]})\nS={successes}  F={failures}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    flat_axes[-1].axis("off")
    handles = [
        plt.Line2D([0], [0], color="#2A9D8F", lw=2, label="Successful"),
        plt.Line2D([0], [0], color="#C1121F", lw=2, label="Unsuccessful"),
    ]
    fig.legend(handles=handles, loc="lower right", frameon=False)

    fig.suptitle(
        "Trajectory outcomes across all perturbation conditions",
        fontsize=12,
        y=1.01,
    )

    return export_figure(fig, "figure5_trajectories_7panel")


def load_kinematics(condition):
    condition_dir = RESULTS_ROOT / f"evaluation_{condition}"
    frames = []
    for path in sorted(condition_dir.glob("kinematics_*.csv")):
        ep = int(path.stem.split("_")[1])
        df = pd.read_csv(path)
        df["episode"] = ep
        df["condition"] = condition
        frames.append(df)
    return frames


def padded_mean(traces):
    max_len = max(len(t) for t in traces)
    arr = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        arr[i, : len(t)] = t
    return np.nanmean(arr, axis=0)


def figure6_velocity_change():
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), constrained_layout=True)
    ax_speed, ax_vx = axes

    for condition in CONDITIONS:
        frames = load_kinematics(condition)
        if not frames:
            continue

        speed_traces = [f["speed"].to_numpy() for f in frames]
        vx_traces = [f["vx"].to_numpy() for f in frames]

        mean_speed = padded_mean(speed_traces)
        mean_vx = padded_mean(vx_traces)

        ax_speed.plot(mean_speed, color=PALETTE[condition], lw=1.8, label=condition)
        ax_vx.plot(mean_vx, color=PALETTE[condition], lw=1.8, label=condition)

    for ax in axes:
        ax.axvspan(25, 39, color="#E9ECEF", alpha=0.6, zorder=0)
        ax.set_xlim(left=0)

    ax_speed.set_title("Velocity profile changes across perturbation conditions")
    ax_speed.set_ylabel("Speed")

    ax_vx.set_title("Lateral velocity (vx) reveals perturbation direction")
    ax_vx.set_ylabel("vx")
    ax_vx.set_xlabel("Step")

    ax_vx.annotate(
        "Leftward perturbation drives negative vx",
        xy=(32, -0.25),
        xytext=(55, -0.55),
        arrowprops=dict(arrowstyle="->", color="#B56576", lw=1.2),
        color="#B56576",
        fontsize=8,
    )
    ax_vx.annotate(
        "Rightward perturbation drives positive vx",
        xy=(32, 0.25),
        xytext=(55, 0.52),
        arrowprops=dict(arrowstyle="->", color="#1D7874", lw=1.2),
        color="#1D7874",
        fontsize=8,
    )

    handles, labels = ax_speed.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=1)

    return export_figure(fig, "figure6_velocity_change")


def figure7_pca_tuning_correlation():
    policy_path = NEURAL_ROOT / "policy_pca_scores.csv"
    if not policy_path.exists():
        return []

    pca_df = pd.read_csv(policy_path)
    summary_frames = []
    for condition in CONDITIONS:
        df = load_condition_summary(condition)
        df["condition"] = condition
        summary_frames.append(df)
    summary_df = pd.concat(summary_frames, ignore_index=True)

    tuning_rows = []
    for condition in CONDITIONS:
        subset = summary_df[summary_df["condition"] == condition]
        tuning_rows.append(
            {
                "condition": condition,
                "success_rate": subset["success"].mean() * 100.0,
                "peak_lateral_velocity": subset["peak_lateral_velocity"].mean(),
                "final_lateral_error": subset["final_lateral_error"].mean(),
            }
        )
    tuning_df = pd.DataFrame(tuning_rows)

    metric_df = summary_df[
        [
            "reward",
            "steps",
            "path_length",
            "mean_speed",
            "peak_lateral_velocity",
            "max_heading_deviation",
            "final_lateral_error",
        ]
    ]
    corr = metric_df.corr(numeric_only=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    sample = pca_df.sample(min(8000, len(pca_df)), random_state=42)
    sns.scatterplot(
        data=sample,
        x="PC1",
        y="PC2",
        hue="condition",
        palette=PALETTE,
        alpha=0.2,
        s=8,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("Policy latent PCA (PC1 vs PC2)")
    axes[0].legend(frameon=False, fontsize=7)

    level_map = {"P0": 0, "L1": 1, "L2": 2, "L3": 3, "R1": 1, "R2": 2, "R3": 3}
    side_map = {"P0": "Control", "L1": "Left", "L2": "Left", "L3": "Left", "R1": "Right", "R2": "Right", "R3": "Right"}
    tuning_df["level"] = tuning_df["condition"].map(level_map)
    tuning_df["side"] = tuning_df["condition"].map(side_map)

    for side, color in [("Left", "#B56576"), ("Right", "#1D7874")]:
        sdf = tuning_df[tuning_df["side"] == side].sort_values("level")
        axes[1].plot(
            sdf["level"],
            sdf["success_rate"],
            marker="o",
            lw=2,
            color=color,
            label=side,
        )
    axes[1].scatter([0], tuning_df[tuning_df["condition"] == "P0"]["success_rate"], color="#355070", s=40, label="Control")
    axes[1].set_xticks([0, 1, 2, 3])
    axes[1].set_xticklabels(["P0", "1", "2", "3"])
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_title("Perturbation tuning curve")
    axes[1].legend(frameon=False)

    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar=False,
        square=True,
        linewidths=0.4,
        linecolor="white",
        ax=axes[2],
    )
    axes[2].set_title("Behavioural correlation matrix")

    return export_figure(fig, "figure7_pca_tuning_correlation")


def figure8_learning_and_prediction():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    ax_curve, ax_pred = axes

    if EVAL_NPZ.exists():
        npz = np.load(EVAL_NPZ)
        timesteps = npz["timesteps"]
        results = npz["results"]
        mean_reward = results.mean(axis=1)
        std_reward = results.std(axis=1)

        ax_curve.plot(timesteps, mean_reward, color="#1D3557", lw=2.2)
        ax_curve.fill_between(
            timesteps,
            mean_reward - std_reward,
            mean_reward + std_reward,
            color="#1D3557",
            alpha=0.18,
        )
        ax_curve.set_title("Learning curve (evaluation reward)")
        ax_curve.set_xlabel("Timesteps")
        ax_curve.set_ylabel("Reward")
    else:
        ax_curve.text(0.5, 0.5, "No evaluations.npz found", ha="center", va="center")
        ax_curve.set_axis_off()

    summary_frames = []
    for condition in CONDITIONS:
        df = load_condition_summary(condition)
        df["condition"] = condition
        summary_frames.append(df)
    summary_df = pd.concat(summary_frames, ignore_index=True)

    x = summary_df[
        [
            "steps",
            "path_length",
            "mean_speed",
            "peak_lateral_velocity",
            "max_heading_deviation",
            "final_lateral_error",
            "route_signal",
        ]
    ]
    y = summary_df["success"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)

    prob = clf.predict_proba(x_test)[:, 1]
    pred = clf.predict(x_test)

    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_pred, colorbar=False)
    ax_pred.set_title("Success prediction confusion matrix")

    ax_pred.text(
        0.02,
        -0.22,
        f"ROC AUC = {roc_auc:.3f}",
        transform=ax_pred.transAxes,
        fontsize=8,
    )

    inset = ax_pred.inset_axes([0.58, 0.08, 0.38, 0.38])
    inset.plot(fpr, tpr, color="#2A9D8F", lw=2)
    inset.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    inset.set_title("ROC", fontsize=8)
    inset.tick_params(labelsize=7)

    return export_figure(fig, "figure8_learning_prediction")


def save_caption_draft():
    text = """# Extended Figure Captions (Draft)\n\n## Figure 5 (7-panel trajectories)\nEach panel shows one perturbation condition (P0, L1-L3, R1-R3). Green traces are successful reaches and red traces are unsuccessful attempts. The starting position is fixed and explicitly marked in every panel. Direction arrows indicate the perturbation side so the behavioural adaptation can be read directly from trajectory geometry.\n\n## Figure 6 (velocity change)\nTop panel shows condition-wise mean speed profiles across movement time. Bottom panel shows lateral velocity (vx), where sign and amplitude track the perturbation direction and compensation dynamics. The shaded epoch marks the perturbation window.\n\n## Figure 7 (PCA, tuning, correlation)\nLeft: latent PCA organization by perturbation condition. Middle: perturbation tuning curve showing how success changes with perturbation level by side. Right: correlation matrix of behavioural metrics to reveal coupled changes in kinematics, error, and reward.\n\n## Figure 8 (learning + prediction)\nLeft: learning curve from evaluation checkpoints across training timesteps. Right: success-prediction performance from behavioural features, shown as a confusion matrix with ROC inset. This panel quantifies how well trial outcome can be predicted from measured behavioural state variables.\n"""
    path = Path("paper/extended_figure_captions.md")
    path.write_text(text, encoding="utf-8")
    return path


def main():
    print("=" * 70)
    print("Generating extended manuscript plot bundle")
    print("=" * 70)

    set_style()

    outputs = []
    outputs += figure5_trajectories_7panel()
    outputs += figure6_velocity_change()
    outputs += figure7_pca_tuning_correlation()
    outputs += figure8_learning_and_prediction()

    caption_path = save_caption_draft()

    manifest = pd.DataFrame({"file": [str(p) for p in outputs], "exists": [p.exists() for p in outputs]})
    manifest_path = FIG_OUT / "extended_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nGenerated files:")
    for p in outputs:
        print(p)
    print(caption_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
