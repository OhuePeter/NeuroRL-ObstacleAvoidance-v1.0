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

==========================================================
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


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

LABELS = {
    "P0": "No Perturbation",
    "L1": "Small Left",
    "L2": "Medium Left",
    "L3": "Large Left",
    "R1": "Small Right",
    "R2": "Medium Right",
    "R3": "Large Right",
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


def draw_world(ax):

    ax.scatter(
        START[0],
        START[1],
        s=70,
        c="black",
        label="Start",
        zorder=10,
    )

    goal = plt.Circle(
        GOAL,
        GOAL_RADIUS,
        color="royalblue",
        alpha=0.35,
    )

    ax.add_patch(goal)

    for obstacle in OBSTACLES:

        circle = plt.Circle(
            obstacle,
            OBSTACLE_RADIUS,
            color="black",
            alpha=0.30,
        )

        ax.add_patch(circle)

    ax.set_xlim(0, WORLD_WIDTH)
    ax.set_ylim(0, WORLD_HEIGHT)
    ax.set_aspect("equal")


def plot_condition(ax, condition):

    folder = ROOT / f"evaluation_{condition}"

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

            color = "forestgreen"
            alpha = 0.25
            lw = 1.3
            success_count += 1

        else:

            color = "crimson"
            alpha = 0.90
            lw = 2.0
            failure_count += 1

        ax.plot(
            trajectory["x"],
            trajectory["y"],
            color=color,
            alpha=alpha,
            linewidth=lw,
        )

    draw_world(ax)

    ax.set_title(
        f"{LABELS[condition]}\n"
        f"S={success_count}  F={failure_count}",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")


def main():

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18,9),
    )

    axes = axes.flatten()

    for ax, condition in zip(axes, CONDITIONS):

        plot_condition(ax, condition)

    axes[-1].axis("off")

    axes[-1].plot([], [], color="forestgreen",
                  linewidth=2,
                  label="Success")

    axes[-1].plot([], [], color="crimson",
                  linewidth=2,
                  label="Failure")

    axes[-1].legend(
        loc="center",
        fontsize=12,
    )

    plt.suptitle(
        "Experiment 2\n"
        "Cumulative Trajectories Under External Perturbations",
        fontsize=18,
        fontweight="bold",
    )

    plt.tight_layout()

    output = ROOT / "cumulative_trajectories.png"

    plt.savefig(
        output,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print("="*60)
    print(output)
    print("="*60)


if __name__ == "__main__":
    main()