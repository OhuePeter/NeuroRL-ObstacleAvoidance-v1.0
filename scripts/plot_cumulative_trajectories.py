"""
==========================================================
Experiment 2

Cumulative Trajectory Plot

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Plots cumulative trajectories for all perturbation
conditions in publication style.

Green = Success
Red = Failure

SVG export: text kept as editable text (not paths) so the
file opens cleanly in Inkscape for final publication polish.

==========================================================
"""

from pathlib import Path
import json

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"   # editable text in Inkscape
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["xtick.major.width"] = 0.8
matplotlib.rcParams["ytick.major.width"] = 0.8
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["axes.titlesize"] = 9
matplotlib.rcParams["legend.fontsize"] = 9
matplotlib.rcParams["figure.dpi"] = 150

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/version_2_0/results")
OUTPUT_DIR = Path("paper/figures")

CONDITIONS = [
    "P0",
    "L1",
    "L2",
    "L3",
    "R1",
    "R2",
    "R3",
]

LABELS = {
    "P0": "No Perturbation (P0)",
    "L1": "Left — Small (L1)",
    "L2": "Left — Medium (L2)",
    "L3": "Left — Large (L3)",
    "R1": "Right — Small (R1)",
    "R2": "Right — Medium (R2)",
    "R3": "Right — Large (R3)",
}

# Condition colours: neutral → increasing saturation per direction
CONDITION_COLOURS = {
    "P0": "#555555",
    "L1": "#f4a9c4",
    "L2": "#e05c9a",
    "L3": "#a8004e",
    "R1": "#a0d8c8",
    "R2": "#3aab8c",
    "R3": "#005f47",
}

WORLD_WIDTH = 10
WORLD_HEIGHT = 10

START = (5.0, 1.0)
GOAL = (5.0, 9.0)

OBSTACLES = [
    (4.25, 5.0),
    (5.75, 5.0),
]

GOAL_RADIUS = 0.35
OBSTACLE_RADIUS = 0.50
PERTURB_ZONE_Y = (1.0, 4.5)   # approximate region where perturbation is active


def draw_world(ax):
    # perturbation zone shading
    ax.axhspan(
        PERTURB_ZONE_Y[0],
        PERTURB_ZONE_Y[1],
        color="#d4edda",
        alpha=0.45,
        zorder=0,
        label="Perturbation zone",
    )

    ax.scatter(
        START[0],
        START[1],
        s=45,
        c="black",
        marker="o",
        label="Start",
        zorder=10,
        linewidths=0,
    )

    goal = plt.Circle(
        GOAL,
        GOAL_RADIUS,
        color="royalblue",
        alpha=0.35,
        zorder=8,
    )
    ax.add_patch(goal)

    for obstacle in OBSTACLES:
        circle = plt.Circle(
            obstacle,
            OBSTACLE_RADIUS,
            color="#333333",
            alpha=0.22,
            zorder=9,
        )
        ax.add_patch(circle)

    ax.set_xlim(0, WORLD_WIDTH)
    ax.set_ylim(0, WORLD_HEIGHT)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)


def plot_condition(ax, condition):

    folder = ROOT / f"evaluation_{condition}"
    colour = CONDITION_COLOURS[condition]

    success_count = 0
    failure_count = 0

    for trajectory_file in sorted(folder.glob("trajectory_*.csv")):

        episode = trajectory_file.stem.split("_")[1]

        metadata_file = folder / f"metadata_{episode}.json"

        if not metadata_file.exists():
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        trajectory = pd.read_csv(trajectory_file)

        if metadata["success"]:
            alpha = 0.30
            lw = 1.2
            linestyle = "-"
            success_count += 1
        else:
            alpha = 0.90
            lw = 1.8
            linestyle = "--"
            failure_count += 1

        ax.plot(
            trajectory["x"],
            trajectory["y"],
            color=colour if metadata["success"] else "#cc0000",
            alpha=alpha,
            linewidth=lw,
            linestyle=linestyle,
            solid_capstyle="round",
        )

    draw_world(ax)

    total = success_count + failure_count
    rate = int(round(100 * success_count / total)) if total > 0 else 0

    ax.set_title(
        f"{LABELS[condition]}\n"
        f"Success: {success_count}/{total} ({rate}%)",
        fontsize=8.5,
        fontweight="bold",
        pad=4,
    )

    ax.set_xlabel("X position (a.u.)", labelpad=3)
    ax.set_ylabel("Y position (a.u.)", labelpad=3)
    ax.tick_params(length=3)


def main():

    # 174 mm wide × ~110 mm tall — fits a two-column journal figure
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(6.85, 4.33),   # inches: 174 mm × 110 mm
    )

    axes = axes.flatten()

    for ax, condition in zip(axes, CONDITIONS):
        plot_condition(ax, condition)

    # legend panel
    leg_ax = axes[-1]
    leg_ax.axis("off")
    leg_ax.plot([], [], color="#555555", lw=1.5, linestyle="-",  label="Success")
    leg_ax.plot([], [], color="#cc0000", lw=1.5, linestyle="--", label="Failure")
    leg_ax.plot([], [], color="#3aab8c", lw=1.5, linestyle="-",  label="Rightward pert.")
    leg_ax.plot([], [], color="#e05c9a", lw=1.5, linestyle="-",  label="Leftward pert.")
    leg_ax.fill_between([], [], color="#d4edda", alpha=0.6, label="Perturbation zone")
    leg_ax.legend(loc="center", frameon=False, fontsize=8)

    fig.suptitle(
        "Adaptive route geometry under graded lateral perturbations",
        fontsize=10,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    svg_path = OUTPUT_DIR / "figure5_cumulative_perturbation_trajectories.svg"
    png_path = OUTPUT_DIR / "figure5_cumulative_perturbation_trajectories.png"
    pdf_path = OUTPUT_DIR / "figure5_cumulative_perturbation_trajectories.pdf"

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.show()

    print()
    print("=" * 60)
    print(f"SVG  →  {svg_path}")
    print(f"PNG  →  {png_path}")
    print(f"PDF  →  {pdf_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()