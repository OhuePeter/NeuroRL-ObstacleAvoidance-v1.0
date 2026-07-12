"""
==========================================================
Matplotlib Style

Author:
Peter Ohue
==========================================================
"""

import matplotlib.pyplot as plt


def set_publication_style():
    """
    Configure Matplotlib for publication-quality figures.
    """

    plt.style.use("default")

    plt.rcParams.update({

        "font.family": "DejaVu Sans",

        "font.size": 11,

        "axes.labelsize": 12,

        "axes.titlesize": 13,

        "axes.linewidth": 1.2,

        "xtick.labelsize": 10,

        "ytick.labelsize": 10,

        "legend.fontsize": 10,

        "figure.dpi": 600,

        "savefig.dpi": 600,

        "savefig.bbox": "tight",

        "axes.spines.top": False,

        "axes.spines.right": False,

        "lines.linewidth": 2.0

    })