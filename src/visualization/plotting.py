"""
==========================================================
Plotting Utilities

Author:
Peter Ohue
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt

from .styles import set_publication_style


def save_figure(fig, filename):
    """
    Save figure as PNG, PDF and SVG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure

    filename : str
        Figure filename without extension.
    """

    root = Path("figures/manuscript")

    root.mkdir(parents=True, exist_ok=True)

    fig.savefig(root / f"{filename}.png")

    fig.savefig(root / f"{filename}.pdf")

    fig.savefig(root / f"{filename}.svg")


def create_figure(width=7, height=5):
    """
    Create a publication-quality figure.
    """

    set_publication_style()

    fig, ax = plt.subplots(figsize=(width, height))

    return fig, ax