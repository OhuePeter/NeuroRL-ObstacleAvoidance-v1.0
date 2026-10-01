"""
==========================================================
Build Manuscript Figure 2: Cumulative Reach Trajectories

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Single-panel overlay of all evaluation trajectories across
the seven perturbation conditions (P0, L1-L3, R1-R3),
reproducing paper/fig2_trajectories.png/.svg from the raw
evaluation data instead of a hand-edited Inkscape file.

Fonts are set to Arial 10pt throughout (COSYNE poster/figure
convention). This figure has no sub-panels, so no panel
label (I/II/III) is drawn.

SVG export: text kept as editable text (not paths) so the
file still opens cleanly in Inkscape for final polish.

Usage
-----
python scripts/build_fig2_trajectories.py
==========================================================
"""

from pathlib import Path

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"   # editable text in Inkscape
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["figure.dpi"] = 150

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("experiments/version_1_0/results")
OUTPUT_PNG = Path("paper/fig2_trajectories.png")
OUTPUT_SVG = Path("paper/figures/fig2_trajectories.svg")

CONDITIONS = ["P0", "L1", "L2", "L3", "R1", "R2", "R3"]

CONDITION_COLOURS = {
    "P0": "#555555",
    "L1": "#f4a9c4",
    "L2": "#e05c9a",
    "L3": "#a8004e",
    "R1": "#a0d8c8",
    "R2": "#3aab8c",
    "R3": "#005f47",
}

FAILURE_COLOUR = "#cc0000"

WORLD_WIDTH = 10
WORLD_HEIGHT = 10

START = (5.0, 1.0)
GOAL = (5.0, 9.0)
GOAL_RADIUS = 0.35

OBSTACLES = [
    (4.25, 5.0),
    (5.75, 5.0),
]
OBSTACLE_RADIUS = 0.50

PERTURB_ZONE_Y = (1.0, 4.5)   # approximate region where perturbation is active


def plot_condition(ax, condition):

    folder = ROOT / f"evaluation_{condition}"
    summary = pd.read_csv(folder / "summary.csv")
    colour = CONDITION_COLOURS[condition]

    for trajectory_file in sorted(folder.glob("trajectory_*.csv")):

        episode = int(trajectory_file.stem.split("_")[1])
        success = bool(summary.iloc[episode]["success"])
        trajectory = pd.read_csv(trajectory_file)

        if success:
            ax.plot(
                trajectory["x"], trajectory["y"],
                color=colour, alpha=0.55, linewidth=1.1,
                linestyle="-", solid_capstyle="round", zorder=5,
            )
        else:
            ax.plot(
                trajectory["x"], trajectory["y"],
                color=FAILURE_COLOUR, alpha=0.90, linewidth=1.6,
                linestyle="--", zorder=6,
            )


def draw_world(ax):

    ax.axhspan(
        PERTURB_ZONE_Y[0], PERTURB_ZONE_Y[1],
        color="#d4edda", alpha=0.45, zorder=0,
    )

    mid_y = sum(PERTURB_ZONE_Y) / 2
    ax.text(0.2, mid_y + 0.35, "Perturbation", ha="left", va="bottom", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(2.1, mid_y), xytext=(0.2, mid_y), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(9.8, mid_y + 0.35, "Perturbation", ha="right", va="bottom", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(7.9, mid_y), xytext=(9.8, mid_y), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    for i, obstacle in enumerate(OBSTACLES, start=1):
        circle = plt.Circle(obstacle, OBSTACLE_RADIUS, color="#333333", alpha=0.22, zorder=9)
        ax.add_patch(circle)
        ax.text(
            obstacle[0], obstacle[1] + OBSTACLE_RADIUS + 0.35, f"Obstacle {i}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2b2b7a",
        )

    goal_patch = plt.Circle(GOAL, GOAL_RADIUS, color="royalblue", alpha=0.35, zorder=8)
    ax.add_patch(goal_patch)
    ax.text(GOAL[0], GOAL[1], "Goal", ha="center", va="center", fontsize=9, fontweight="bold", color="#1b1b3a")

    ax.scatter(START[0], START[1], s=45, c="black", marker="o", zorder=10, linewidths=0)
    ax.text(START[0], START[1] - 0.45, "Start", ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_xlim(0, WORLD_WIDTH)
    ax.set_ylim(0, WORLD_HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")


def build_legend(ax):

    left_handles = [
        plt.Line2D([], [], color=CONDITION_COLOURS[c], lw=2, label=c)
        for c in ["P0", "L1", "L2", "L3"]
    ]
    left_handles.append(
        plt.Line2D([], [], color=FAILURE_COLOUR, lw=1.6, linestyle="--", label="Failure trajectory")
    )
    right_handles = [
        plt.Line2D([], [], color=CONDITION_COLOURS[c], lw=2, label=c)
        for c in ["R1", "R2", "R3"]
    ]

    leg1 = ax.legend(handles=left_handles, loc="upper left", frameon=False, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=right_handles, loc="upper right", frameon=False, fontsize=9)


def main():

    fig, ax = plt.subplots(figsize=(6.0, 7.5))

    for condition in CONDITIONS:
        plot_condition(ax, condition)

    draw_world(ax)
    build_legend(ax)

    ax.set_title(
        "Adaptive route geometry under graded lateral perturbations",
        fontsize=11, fontweight="bold", pad=10,
    )

    fig.tight_layout()

    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"Figure 2 saved to:\n{OUTPUT_PNG.resolve()}\n{OUTPUT_SVG.resolve()}")


if __name__ == "__main__":
    main()
