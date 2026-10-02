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

Panel sizes are computed directly from each source image's
pixel aspect ratio (not a fixed grid ratio), so the two top
panels exactly fill the row with no dead space, and the
bottom panel's height matches its own aspect ratio. The
caption text is intentionally not baked into the image: the
abstract already carries the caption as separate, editable
text, and duplicating it here just wastes vertical space.

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

TOTAL_WIDTH = 11.0        # inches
COLUMN_GAP = 0.15         # inches, between panel I and panel II
ROW_GAP = 0.35            # inches, between the top row and panel III
LABEL_STRIP = 0.3         # inches reserved above each row for the I/II/III label
MARGIN = 0.15             # inches, outer margin on all sides


def panel_label(ax, label):
    ax.text(
        0.0, 1.02, label,
        transform=ax.transAxes,
        fontsize=14, fontweight="bold",
        va="bottom", ha="left",
    )


def main():

    images = {key: mpimg.imread(path) for key, path in SOURCES.items()}
    # aspect = height / width, read directly from each PNG's pixel shape
    aspect = {key: img.shape[0] / img.shape[1] for key, img in images.items()}

    # Choose column widths w1, w2 (for panels I, II) so that, at those
    # widths, both images need exactly the same row height: this is what
    # removes the dead space that a fixed grid ratio otherwise leaves.
    usable_width = TOTAL_WIDTH - 2 * MARGIN - COLUMN_GAP
    row0_height = usable_width / (1 / aspect["I"] + 1 / aspect["II"])
    col_width_i = row0_height / aspect["I"]
    col_width_ii = row0_height / aspect["II"]

    row1_width = TOTAL_WIDTH - 2 * MARGIN
    row1_height = row1_width * aspect["III"]

    fig_width = TOTAL_WIDTH
    fig_height = (
        MARGIN + LABEL_STRIP + row0_height + ROW_GAP
        + LABEL_STRIP + row1_height + MARGIN
    )

    fig = plt.figure(figsize=(fig_width, fig_height))

    def add_panel(x0, y0_from_top, width, height, image, label):
        # y0_from_top is measured from the top of the figure, in inches.
        rect = [
            x0 / fig_width,
            1 - (y0_from_top + height) / fig_height,
            width / fig_width,
            height / fig_height,
        ]
        ax = fig.add_axes(rect)
        ax.imshow(image)
        ax.axis("off")
        panel_label(ax, label)

    top_y = MARGIN + LABEL_STRIP
    add_panel(MARGIN, top_y, col_width_i, row0_height, images["I"], "I")
    add_panel(MARGIN + col_width_i + COLUMN_GAP, top_y, col_width_ii, row0_height, images["II"], "II")

    bottom_y = top_y + row0_height + ROW_GAP + LABEL_STRIP
    add_panel(MARGIN, bottom_y, row1_width, row1_height, images["III"], "III")

    fig.savefig(OUTPUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUTPUT_SVG, facecolor="white")

    plt.close(fig)

    print(f"Composite figure saved to:\n{OUTPUT_PNG.resolve()}\n{OUTPUT_SVG.resolve()}")


if __name__ == "__main__":
    main()
