"""
==========================================================
Experiment 2

Overlay of All Trajectories

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Overlay all trajectories.

Colours
--------
Gray  : No perturbation (P0)
Blue  : Left perturbations (L1-L3)
Green : Right perturbations (R1-R3)
Red   : Failed trajectories

==========================================================
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ==========================================================
# Paths
# ==========================================================

ROOT = Path("experiments/version_2_0/results")

CONDITIONS = [
    "P0",
    "L1",
    "L2",
    "L3",
    "R1",
    "R2",
    "R3",
]

# ==========================================================
# World
# ==========================================================

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


# ==========================================================
# Main
# ==========================================================

def main():

    fig, ax = plt.subplots(figsize=(8, 8))

    p0_count = 0
    left_count = 0
    right_count = 0
    failure_count = 0

    # ------------------------------------------------------
    # Plot all trajectories
    # ------------------------------------------------------

    for condition in CONDITIONS:

        folder = ROOT / f"evaluation_{condition}"

        summary = pd.read_csv(folder / "summary.csv")

        trajectory_files = sorted(folder.glob("trajectory_*.csv"))

        for episode, trajectory_file in enumerate(trajectory_files):

            trajectory = pd.read_csv(trajectory_file)

            success = bool(summary.iloc[episode]["success"])

            # ----------------------------------------------
            # Choose colour
            # ----------------------------------------------

            if not success:

                color = "crimson"
                alpha = 0.90
                lw = 2.5

                failure_count += 1

            elif condition == "P0":

                color = "gray"
                alpha = 0.35
                lw = 1.2

                p0_count += 1

            elif condition.startswith("L"):

                color = "royalblue"
                alpha = 0.35
                lw = 1.4

                left_count += 1

            else:

                color = "forestgreen"
                alpha = 0.35
                lw = 1.4

                right_count += 1

            ax.plot(
                trajectory["x"],
                trajectory["y"],
                color=color,
                alpha=alpha,
                linewidth=lw,
            )

    # ------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------

    for obstacle in OBSTACLES:

        ax.add_patch(

            Circle(
                obstacle,
                OBSTACLE_RADIUS,
                color="black",
                alpha=0.45,
            )

        )

    # ------------------------------------------------------
    # Goal
    # ------------------------------------------------------

    ax.add_patch(

        Circle(
            GOAL,
            GOAL_RADIUS,
            color="royalblue",
            alpha=0.40,
        )

    )

    # ------------------------------------------------------
    # Start
    # ------------------------------------------------------

    ax.scatter(
        START[0],
        START[1],
        c="black",
        s=80,
        zorder=20,
    )

    # ------------------------------------------------------
    # Figure formatting
    # ------------------------------------------------------

    ax.set_xlim(0, WORLD_WIDTH)
    ax.set_ylim(0, WORLD_HEIGHT)

    ax.set_aspect("equal")

    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)

    ax.grid(alpha=0.30)

    ax.set_title(
        "Experiment 2\n"
        "Overlay of Navigation Trajectories Under External Perturbations",
        fontsize=16,
        fontweight="bold",
    )

    # ------------------------------------------------------
    # Legend
    # ------------------------------------------------------

    import matplotlib.lines as mlines

    legend = [

        mlines.Line2D(
            [],
            [],
            color="gray",
            linewidth=3,
            label=f"P0 ({p0_count})",
        ),

        mlines.Line2D(
            [],
            [],
            color="royalblue",
            linewidth=3,
            label=f"Left Perturbations ({left_count})",
        ),

        mlines.Line2D(
            [],
            [],
            color="forestgreen",
            linewidth=3,
            label=f"Right Perturbations ({right_count})",
        ),

        mlines.Line2D(
            [],
            [],
            color="crimson",
            linewidth=3,
            label=f"Failures ({failure_count})",
        ),

    ]

    ax.legend(
        handles=legend,
        loc="upper left",
        fontsize=11,
    )

    plt.tight_layout()

    output = ROOT / "trajectory_overlay.png"

    plt.savefig(
        output,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print("=" * 60)
    print("Trajectory overlay saved to:")
    print(output)
    print("=" * 60)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()