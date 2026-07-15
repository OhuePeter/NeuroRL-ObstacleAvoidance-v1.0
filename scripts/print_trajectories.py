"""
==========================================================
Print Individual Trajectories

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Plots every trajectory individually to verify that
trajectory files are being loaded correctly.
==========================================================
"""

from pathlib import Path

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


def main():

    for condition in CONDITIONS:

        folder = ROOT / f"evaluation_{condition}"

        print("=" * 60)
        print(condition)
        print(folder)
        print("=" * 60)

        files = sorted(folder.glob("trajectory_*.csv"))

        print(f"Found {len(files)} trajectory files")

        for file in files:

            print(f"Loading {file.name}")

            df = pd.read_csv(file)

            print(df.head())

            plt.figure(figsize=(6, 6))

            plt.plot(
                df["x"],
                df["y"],
                color="forestgreen",
                linewidth=2,
            )

            # Start
            plt.scatter(
                df["x"].iloc[0],
                df["y"].iloc[0],
                c="black",
                s=80,
                label="Start",
            )

            # End
            plt.scatter(
                df["x"].iloc[-1],
                df["y"].iloc[-1],
                c="red",
                s=80,
                label="End",
            )

            # Obstacles
            obstacle1 = plt.Circle(
                (4.25, 5.0),
                0.5,
                color="black",
                alpha=0.4,
            )

            obstacle2 = plt.Circle(
                (5.75, 5.0),
                0.5,
                color="black",
                alpha=0.4,
            )

            plt.gca().add_patch(obstacle1)
            plt.gca().add_patch(obstacle2)

            # Goal
            goal = plt.Circle(
                (5.0, 9.0),
                0.35,
                color="royalblue",
                alpha=0.5,
            )

            plt.gca().add_patch(goal)

            plt.xlim(0, 10)
            plt.ylim(0, 10)

            plt.gca().set_aspect("equal")

            plt.grid(True)

            plt.title(
                f"{condition} : {file.stem}",
                fontsize=14,
            )

            plt.xlabel("X Position")
            plt.ylabel("Y Position")

            plt.legend()

            plt.show()


if __name__ == "__main__":
    main()