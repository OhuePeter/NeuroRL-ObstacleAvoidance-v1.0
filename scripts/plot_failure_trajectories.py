"""
==========================================================
Plot Success and Failure Trajectories

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Plots all trajectories from one perturbation condition.

Green = Success
Red = Failure

Requires:
- trajectory_XXX.csv
- metadata_XXX.json

==========================================================
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


RESULTS = Path("experiments/version_2_0/results")


def plot_condition(condition):

    folder = RESULTS / f"evaluation_{condition}"

    trajectory_files = sorted(
        folder.glob("trajectory_*.csv")
    )

    plt.figure(figsize=(6, 8))

    success_count = 0
    failure_count = 0

    for trajectory_file in trajectory_files:

        episode = trajectory_file.stem.split("_")[1]

        metadata_file = (
            folder /
            f"metadata_{episode}.json"
        )

        if not metadata_file.exists():
            print(f"Missing {metadata_file}")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        trajectory = pd.read_csv(trajectory_file)

        if metadata["success"]:

            color = "forestgreen"
            alpha = 0.5
            success_count += 1

        else:

            color = "crimson"
            alpha = 0.9
            failure_count += 1

        plt.plot(
            trajectory["x"],
            trajectory["y"],
            color=color,
            alpha=alpha,
            linewidth=2,
        )

        plt.scatter(
            trajectory["x"].iloc[0],
            trajectory["y"].iloc[0],
            c="black",
            s=10,
        )

        plt.scatter(
            trajectory["x"].iloc[-1],
            trajectory["y"].iloc[-1],
            c=color,
            s=20,
        )

    # Obstacles

    obstacle_x = [4.25, 5.75]
    obstacle_y = [5.0, 5.0]
    obstacle_radius = 0.5

    ax = plt.gca()

    for x, y in zip(obstacle_x, obstacle_y):

        obstacle = plt.Circle(
            (x, y),
            obstacle_radius,
            color="black",
            alpha=0.25,
        )

        ax.add_patch(obstacle)

    # Goal

    goal = plt.Circle(
        (5.0, 9.0),
        0.35,
        color="blue",
        alpha=0.3,
    )

    ax.add_patch(goal)

    plt.xlim(0, 10)
    plt.ylim(0, 10)

    plt.gca().set_aspect("equal")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    plt.title(
        f"{condition}\n"
        f"Success: {success_count}   "
        f"Failure: {failure_count}"
    )

    plt.tight_layout()

    output = (
        folder /
        f"failure_trajectory_comparison_{condition}.png"
    )

    plt.savefig(
        output,
        dpi=300,
    )

    plt.close()

    print(output)


def main():

    conditions = [
        "P0",
        "L1",
        "L2",
        "L3",
        "R1",
        "R2",
        "R3",
    ]

    for condition in conditions:

        plot_condition(condition)


if __name__ == "__main__":
    main()