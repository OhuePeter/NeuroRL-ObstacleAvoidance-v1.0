"""
==========================================================
Experiment 2

Nashed-style Cumulative Trajectories

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Plots cumulative trajectories grouped by the
route taken around the obstacle pair.

Green = Success
Red = Failure

==========================================================
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

START = (5.0, 1.0)
GOAL = (5.0, 9.0)

OBSTACLES = [
    (4.25, 5.0),
    (5.75, 5.0),
]

GOAL_RADIUS = 0.35
OBSTACLE_RADIUS = 0.50


def classify_route(df):
    """
    Classify a trajectory based on its x-position
    when crossing the obstacle row (y≈5).
    """

    idx = (df["y"] - 5.0).abs().idxmin()

    x = df.loc[idx, "x"]

    if x < 4.6:
        return "Left"

    elif x > 5.4:
        return "Right"

    else:
        return "Middle"


def draw_world(ax):

    ax.scatter(
        START[0],
        START[1],
        s=80,
        c="black",
        zorder=20,
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
            alpha=0.35,
        )

        ax.add_patch(circle)

    ax.set_xlim(0,10)
    ax.set_ylim(0,10)

    ax.set_aspect("equal")

    ax.grid(alpha=0.2)


def main():

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18,6),
    )

    panels = {
        "Left": axes[0],
        "Middle": axes[1],
        "Right": axes[2],
    }

    counts = {
        "Left":[0,0],
        "Middle":[0,0],
        "Right":[0,0],
    }

    for condition in CONDITIONS:

        folder = ROOT / f"evaluation_{condition}"

        print(f"\n{condition}")
        print(folder)

        files = sorted(folder.glob("trajectory_*.csv"))

        print(f"Found {len(files)} trajectory files")

        for trajectory_file in files:

            episode = trajectory_file.stem.split("_")[1]

            metadata_file = folder / f"metadata_{episode}.json"

            if not metadata_file.exists():
                continue

            with open(metadata_file) as f:
                metadata = json.load(f)

            trajectory = pd.read_csv(trajectory_file)

            print(
                trajectory_file.name,
                trajectory.shape,
                trajectory.head()
            )

            route = classify_route(trajectory)

            ax = panels[route]

            if metadata["success"]:

                color = "forestgreen"
                alpha = 0.20
                lw = 1.2

                counts[route][0] += 1

            else:

                color = "crimson"
                alpha = 0.90
                lw = 2.5

                counts[route][1] += 1

            ax.plot(
                trajectory["x"],
                trajectory["y"],
                color=color,
                alpha=alpha,
                linewidth=lw,
            )

    for route, ax in panels.items():

        draw_world(ax)

        s,f = counts[route]

        ax.set_title(
            f"{route} Route\n"
            f"Success={s}   Failure={f}",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    # legend

    green = plt.Line2D(
        [0],
        [0],
        color="forestgreen",
        lw=3,
        label="Successful",
    )

    red = plt.Line2D(
        [0],
        [0],
        color="crimson",
        lw=3,
        label="Failed",
    )

    fig.legend(
        handles=[green, red],
        loc="upper center",
        ncol=2,
        fontsize=12,
    )

    plt.suptitle(
        "Experiment 2\n"
        "Cumulative Trajectories Grouped by Route Selection",
        fontsize=18,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0,0,1,0.93])

    output = ROOT / "nashed_style_trajectories.png"

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