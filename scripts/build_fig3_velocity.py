"""
==========================================================
Build Manuscript Figure 3: Velocity Profiles

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Two-panel figure built directly from per-episode kinematics
CSVs, replacing the hand-edited paper/fig3_velocity.svg:
  Panel I  - mean lateral velocity (vx) +/-1 SD over time,
             aligned to movement onset, per condition.
  Panel II - mean total speed vs. normalised movement phase
             (0 = onset, 100 = goal arrival), per condition.

Fonts are set to Arial 10pt throughout (COSYNE convention).
Panel labels use Roman numerals (I, II) rather than letters.

Usage
-----
python scripts/build_fig3_velocity.py
==========================================================
"""

from pathlib import Path

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"   # editable text in Inkscape
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["figure.dpi"] = 150

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("experiments/version_1_0/results")
OUTPUT_PNG = Path("paper/fig3_velocity.png")
OUTPUT_SVG = Path("paper/figures/fig3_velocity.svg")

CONDITIONS = ["P0", "L1", "L2", "L3", "R1", "R2", "R3"]

CONDITION_COLOURS = {
    "P0": "#555555",
    "L1": "#f4a9c4",
    "L2": "#e05c9a",
    "L3": "#a8004e",
    "R1": "#a0d8c8",
    "R2": "#3aab8c",
    "R3": "#005f47",
}

PERTURB_STEPS = (50, 55)
N_PHASE_BINS = 100


def panel_label(ax, label):
    ax.text(
        0.0, 1.03, label,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        va="bottom", ha="left",
    )


def load_condition_kinematics(condition):

    folder = ROOT / f"evaluation_{condition}"
    episodes = [pd.read_csv(f) for f in sorted(folder.glob("kinematics_*.csv"))]
    return episodes


def plot_lateral_velocity(ax, kinematics_by_condition):

    # Raw vx is dominated by route geometry: every saved episode takes the
    # same left detour, so even the unperturbed (P0) trajectory has
    # substantial negative vx during this window. The perturbation-induced
    # correction only becomes visible once that shared route baseline is
    # subtracted out, so this panel plots the deviation from the P0 mean
    # rather than raw vx.
    min_len_all = min(len(e) for episodes in kinematics_by_condition.values() for e in episodes)
    p0_episodes = kinematics_by_condition["P0"]
    p0_vx = np.stack([e["vx"].to_numpy()[:min_len_all] for e in p0_episodes])
    p0_mean_vx = p0_vx.mean(axis=0)

    for condition in CONDITIONS:

        episodes = kinematics_by_condition[condition]
        vx = np.stack([e["vx"].to_numpy()[:min_len_all] for e in episodes])

        deviation = vx - p0_mean_vx
        mean_dev = deviation.mean(axis=0)
        sd_dev = deviation.std(axis=0)
        steps = np.arange(min_len_all)

        colour = CONDITION_COLOURS[condition]
        ax.plot(steps, mean_dev, color=colour, lw=1.4, label=condition)
        ax.fill_between(steps, mean_dev - sd_dev, mean_dev + sd_dev, color=colour, alpha=0.12, linewidth=0)

    ax.axvspan(*PERTURB_STEPS, color="#d4edda", alpha=0.5, zorder=0)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Lateral velocity deviation from P0, $\\Delta v_x$ (a.u./s)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)


def plot_speed_phase(ax, kinematics_by_condition):

    phase = np.linspace(0, 100, N_PHASE_BINS)

    for condition in CONDITIONS:

        episodes = kinematics_by_condition[condition]
        resampled = []

        for episode in episodes:
            n = len(episode)
            episode_phase = np.linspace(0, 100, n)
            resampled.append(np.interp(phase, episode_phase, episode["speed"].to_numpy()))

        resampled = np.stack(resampled)
        mean_speed = resampled.mean(axis=0)

        colour = CONDITION_COLOURS[condition]
        ax.plot(phase, mean_speed, color=colour, lw=1.4, label=condition)

    ax.set_xlabel("Normalised movement phase (%)")
    ax.set_ylabel("Speed (a.u./s)")
    ax.spines[["top", "right"]].set_visible(False)


def main():

    kinematics_by_condition = {c: load_condition_kinematics(c) for c in CONDITIONS}

    fig, (ax_i, ax_ii) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    plot_lateral_velocity(ax_i, kinematics_by_condition)
    panel_label(ax_i, "I")

    plot_speed_phase(ax_ii, kinematics_by_condition)
    panel_label(ax_ii, "II")

    fig.tight_layout()

    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"Figure 3 saved to:\n{OUTPUT_PNG.resolve()}\n{OUTPUT_SVG.resolve()}")


if __name__ == "__main__":
    main()
