"""
==========================================================
Build COSYNE Composite Figure: Trajectories, Manifold,
and Closed-Loop Attractor Mechanism

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Combines the three main-text figures that together tell the
mechanistic story of the manuscript into a single poster/
abstract-ready composite:
  Panel I   - Figure 2 (cumulative reach trajectories)
  Panel II  - Figure 4 (hidden-layer PCA manifold + clustering)
  Panel III - Figure 7 (closed-loop fixed-point / attractor analysis)

Each source figure already contains its own internal panel
labels (I/II for Figure 4, I/II/III for Figure 7); this script
only adds the outer I/II/III labels that identify which
main-text figure each block reproduces, matching the existing
COSYNE abstract caption.

Fonts are set to Arial 10pt throughout (COSYNE convention); text
is kept editable in the SVG output.

Usage
-----
python scripts/build_cosyne_composite_figure.py
==========================================================
"""

from pathlib import Path

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"   # editable text in Inkscape
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 10

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

PAPER = Path("paper")
OUTPUT_PNG = PAPER / "figures" / "fig_composite_obstacle_avoidance_mechanism.png"
OUTPUT_SVG = PAPER / "figures" / "fig_composite_obstacle_avoidance_mechanism.svg"

SOURCES = {
    "I": PAPER / "fig2_trajectories.png",
    "II": PAPER / "fig4_pca.png",
    "III": PAPER / "fig7_attractor.png",
}

CAPTION = (
    "Figure. Obstacle avoidance and its mechanism. (I) Trajectories bend "
    "with perturbation strength; dashed lines mark failures, which occur "
    "at the strongest rightward impulses. (II) Hidden-layer activity "
    "forms a compact PC1-PC2 manifold with four phase-aligned clusters. "
    "(III) Closed-loop analysis identifies one stable fixed point per "
    "detour route. The fixed-point curves track the empirical paths and "
    "occupy the same manifold region as the evaluation activity. Reduced "
    "obstacle clearance on the left route is consistent with the higher "
    "failure rate under rightward perturbations."
)


def panel_label(ax, label):
    ax.text(
        0.0, 1.02, label,
        transform=ax.transAxes,
        fontsize=14, fontweight="bold",
        va="bottom", ha="left",
    )


def main():

    fig = plt.figure(figsize=(12, 13))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.14, wspace=0.06)

    ax_i = fig.add_subplot(grid[0, 0])
    ax_ii = fig.add_subplot(grid[0, 1])
    ax_iii = fig.add_subplot(grid[1, :])

    for ax, key in ((ax_i, "I"), (ax_ii, "II"), (ax_iii, "III")):
        image = mpimg.imread(SOURCES[key])
        ax.imshow(image)
        ax.axis("off")
        panel_label(ax, key)

    fig.text(0.02, 0.012, CAPTION, ha="left", va="bottom", fontsize=9, wrap=True)

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"Composite figure saved to:\n{OUTPUT_PNG.resolve()}\n{OUTPUT_SVG.resolve()}")


if __name__ == "__main__":
    main()
