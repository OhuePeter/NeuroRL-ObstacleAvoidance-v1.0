"""
==========================================================
Plot Trajectories

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Plots all evaluation trajectories for the
baseline (P0) condition.

Version:
1.0
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

from src.utils.config import ConfigLoader


def main():

    config = ConfigLoader.load_environment()

    env = config["environment"]
    goal = config["goal"]
    obstacles = config["obstacles"]

    width = env["width"]
    height = env["height"]

    results_dir = Path(
        "experiments/version_1_0/results/evaluation_P0"
    )

    figures_dir = Path(
        "experiments/version_1_0/figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # ----------------------------------------------------
    # World Boundary
    # ----------------------------------------------------

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")

    # ----------------------------------------------------
    # Start Position
    # ----------------------------------------------------

    ax.scatter(
        width / 2,
        1.0,
        s=100,
        color="green",
        label="Start",
        zorder=5,
    )

    # ----------------------------------------------------
    # Goal
    # ----------------------------------------------------

    goal_circle = Circle(
        (width / 2, height - 1),
        goal["radius"],
        color="red",
        alpha=0.4,
        label="Goal",
    )

    ax.add_patch(goal_circle)

    # ----------------------------------------------------
    # Obstacles
    # ----------------------------------------------------

    for i, position in enumerate(obstacles["positions"]):

        obstacle = Circle(
            position,
            obstacles["radius"],
            color="black",
            alpha=0.6,
        )

        ax.add_patch(obstacle)

    # ----------------------------------------------------
    # Plot trajectories
    # ----------------------------------------------------

    files = sorted(results_dir.glob("trajectory_*.csv"))

    for file in files:

        trajectory = pd.read_csv(file)

        ax.plot(
            trajectory["x"],
            trajectory["y"],
            linewidth=1.5,
            alpha=0.7,
            color="royalblue",
        )

    ax.set_title(
        "Baseline (P0) Trajectories",
        fontsize=16,
    )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    ax.grid(True)

    ax.legend()

    output = figures_dir / "trajectory_P0.png"

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 60)
    print("TRAJECTORY FIGURE CREATED")
    print("=" * 60)
    print(f"Saved to:\n{output}")


if __name__ == "__main__":
    main()