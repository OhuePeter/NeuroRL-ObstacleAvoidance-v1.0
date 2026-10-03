"""Quick validation plot: overlay the recovered fig2 trajectories on the
known world geometry, to visually confirm the SVG recovery is sound.
Not a manuscript figure -- diagnostic only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("experiments/version_1_0/results/recovered_fig2_svg_trajectories")
OUT = Path("experiments/version_1_0/results/recovered_fig2_svg_trajectories/validation_plot.png")

CONDITION_COLOURS = {
    "P0": "#234551", "L1": "#784c7f", "L2": "#b34c7d", "L3": "#cf4759",
    "R1": "#289c8f", "R2": "#1e788a", "R3": "#0a3a47",
}

OBSTACLES = [(4.25, 5.0), (5.75, 5.0)]
GOAL = (5.0, 9.0)
START = (5.0, 1.0)

fig, ax = plt.subplots(figsize=(6, 7))

for condition, colour in CONDITION_COLOURS.items():
    for f in sorted((ROOT / condition).glob("episode_*.csv")):
        df = pd.read_csv(f)
        ax.plot(df["x"], df["y"], color=colour, alpha=0.5, linewidth=1)

for obs in OBSTACLES:
    ax.add_patch(plt.Circle(obs, 0.5, color="black", alpha=0.2))
ax.add_patch(plt.Circle(GOAL, 0.35, color="royalblue", alpha=0.3))
ax.scatter(*START, color="black", s=40, zorder=5)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_title("Recovered Figure 2 trajectories (validation)")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"Saved to {OUT.resolve()}")
