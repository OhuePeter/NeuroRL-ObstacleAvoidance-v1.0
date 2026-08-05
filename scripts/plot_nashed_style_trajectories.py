"""
==========================================================
Experiment 2

Nashed-style Route Replication Figure

Description
-----------
Generates the improved route-family replication panel
used in the extended manuscript bundle.

Outputs:
- experiments/version_2_0/results/nashed_style_trajectories.png
- experiments/version_2_0/results/nashed_style_trajectories.pdf
- experiments/version_2_0/results/nashed_style_trajectories.svg
==========================================================
"""

from pathlib import Path

from scripts.analysis.manuscript_extended_plots import (
    set_style,
    figure9_nashed_replication_clean,
)


def main():
    set_style()
    figure9_nashed_replication_clean()

    root = Path("experiments/version_2_0/results")

    print("\n" + "=" * 60)
    print(root / "nashed_style_trajectories.png")
    print(root / "nashed_style_trajectories.pdf")
    print(root / "nashed_style_trajectories.svg")
    print("=" * 60)


if __name__ == "__main__":
    main()
