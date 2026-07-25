from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

condition = "R2"
episode = 0

trajectory = Path(
    f"experiments/version_2_0/results/evaluation_{condition}/trajectory_{episode:03d}.csv"
)

df = pd.read_csv(trajectory)

plt.figure(figsize=(6, 6))

plt.plot(df["x"], df["y"], linewidth=2)

# Start
plt.scatter(
    df["x"].iloc[0],
    df["y"].iloc[0],
    s=80,
    label="Start"
)

# End
plt.scatter(
    df["x"].iloc[-1],
    df["y"].iloc[-1],
    s=80,
    label="End"
)

# Obstacles
plt.scatter(
    [4.25, 5.75],
    [5.0, 5.0],
    s=300,
    marker="o",
    label="Obstacles"
)

# Goal
plt.scatter(
    [5.0],
    [9.0],
    s=150,
    marker="*",
    label="Goal"
)

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gca().set_aspect("equal")

plt.grid(True)
plt.legend()

plt.show()