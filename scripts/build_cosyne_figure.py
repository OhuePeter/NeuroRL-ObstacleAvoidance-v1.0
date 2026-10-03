"""
==========================================================
Build COSYNE Figure: Obstacle Avoidance and its Mechanism

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Single three-panel figure drawn natively (vector, live text),
replacing the previous composite of embedded raster images.

  I   - reach trajectories for all seven perturbation
        conditions on both detour routes, with the y = 5 m
        slice analysed in III and the closed-loop fixed
        points marked on it;
  II  - PC1-PC2 of hidden-layer activity coloured by time
        within the movement, with the fixed-point curves of
        both routes projected into the same space;
  III - commanded lateral acceleration at y = 5 m and vx = 0
        for each route. The zero crossing is the fixed point,
        and the negative slope there is what makes it stable.

Every panel comes from one checkpoint (best_model) and one
environment (scripts/cosyne_figure_data.py), so nothing mixes
runs. Text is Arial 10 pt and stays editable in the SVG
(svg.fonttype = none). Panel groups and annotations carry
readable ids so the file ungroups cleanly in Inkscape. Only
the dense PC1-PC2 point cloud is rasterised inside the SVG.

Usage
-----
python -m scripts.build_cosyne_figure
==========================================================
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.cosyne_figure_data import (
    CONDITIONS, Y_PROBE, analyse, load_rollouts,
)


def apply_style():
    """Set after every import: the environment/policy modules reset rcParams,
    which silently turned live SVG text into outlined glyphs."""

    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    for key in ("font.size", "axes.labelsize", "axes.titlesize",
                "xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
        matplotlib.rcParams[key] = 10
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["svg.hashsalt"] = "cosyne"

OUT_DIR = Path("paper/figures")
STEM = "fig_composite_obstacle_avoidance_mechanism"

COND_COLOUR = {
    "P0": "#4d4d4d",
    "L1": "#9ecae1", "L2": "#4292c6", "L3": "#08519c",
    "R1": "#fdae6b", "R2": "#f16913", "R3": "#a63603",
}
LEFT_STYLE, RIGHT_STYLE = "-", (0, (5, 2.5))
STYLE = {"left": LEFT_STYLE, "right": RIGHT_STYLE}

OBSTACLES = [(4.25, 5.0), (5.75, 5.0)]
OBSTACLE_R = 0.5
GOAL, GOAL_R, START = (5.0, 9.0), 0.35, (5.0, 1.0)

W = 6.5          # inches = the docx text width, so 10 pt text prints at 10 pt
L_MARGIN, R_MARGIN = 0.62, 0.15


def episodes(d, route=None):
    """Yield (condition, route, step, x, y, success, collision)."""

    outcome = {
        (c, e): (s, k)
        for c, e, s, k in zip(
            d["o_condition"], d["o_episode"], d["o_success"], d["o_collision"]
        )
    }

    for c in CONDITIONS:
        m_c = d["condition"] == c
        for e in np.unique(d["episode"][m_c]):
            m = m_c & (d["episode"] == e)
            r = d["route"][m][0]
            if route is not None and r != route:
                continue
            s, k = outcome[(c, e)]
            yield (c, r, d["step"][m], d["x"][m], d["y"][m], bool(s), bool(k))


def panel_label(fig, x_in, y_in, text, fig_h):
    fig.text(x_in / W, 1 - y_in / fig_h, text, fontsize=10,
             fontweight="bold", ha="left", va="top", gid=f"label_{text}")


def build(d, res):

    top1, w1 = 0.30, 2.60
    h1 = w1 * 9.2 / 7.0          # equal-aspect axes: 7 m wide, 9.2 m tall
    gap_h = 0.85
    h2 = 2.0
    fig_h = top1 + h1 + gap_h + h2 + 1.0
    fig = plt.figure(figsize=(W, fig_h))

    x1 = L_MARGIN
    x2 = x1 + w1 + 0.80
    w2 = W - R_MARGIN - x2
    top2 = top1 + h1 + gap_h

    def axes_in(x, top, w, h, gid):
        ax = fig.add_axes([x / W, 1 - (top + h) / fig_h, w / W, h / fig_h])
        ax.set_gid(gid)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return ax

    analysis = res["analysis"]
    vy = res["vy_plateau"]
    x_star = {r: res[r]["x_star"] for r in ("left", "right")}
    edge = {"left": OBSTACLES[0][0] - OBSTACLE_R,
            "right": OBSTACLES[1][0] + OBSTACLE_R}
    clearance = {r: abs(x_star[r] - edge[r]) for r in ("left", "right")}

    # ------------------------------------------------------ panel I
    ax1 = axes_in(x1, top1, w1, h1, "panel_I_trajectories")
    ax1.set_xlim(1.5, 8.5)
    ax1.set_ylim(0.4, 9.6)
    ax1.set_aspect("equal")

    y_lo = float(np.median(d["y"][(d["condition"] == "P0") & (d["step"] == 40)]))
    y_hi = float(np.median(d["y"][(d["condition"] == "P0") & (d["step"] == 53)]))
    ax1.axhspan(y_lo, y_hi, color="#e6e6e6", zorder=0, lw=0,
                gid="perturbation_window")
    ax1.text(1.6, (y_lo + y_hi) / 2, "Perturbation", ha="left", va="center",
             color="#404040", gid="label_perturbation")

    for i, (ox, oy) in enumerate(OBSTACLES):
        ax1.add_patch(plt.Circle((ox, oy), OBSTACLE_R, facecolor="#bdbdbd",
                                 edgecolor="#737373", lw=0.8, zorder=2,
                                 gid=f"obstacle_{i + 1}"))
    ax1.add_patch(plt.Circle(GOAL, GOAL_R, facecolor="#c6dbef",
                             edgecolor="#2171b5", lw=0.8, zorder=2, gid="goal"))
    ax1.plot(*START, "o", color="black", ms=4, zorder=6, gid="start")

    for c, r, step, x, y, ok, hit in episodes(d):
        ax1.plot(x, y, color=COND_COLOUR[c], lw=0.7, alpha=0.5, zorder=3,
                 ls="-" if ok else (0, (3, 1.5)))
        if hit:
            ax1.plot(x[-1], y[-1], "x", color="black", ms=4, mew=0.9, zorder=7)

    ax1.axhline(Y_PROBE, color="black", lw=0.6, ls=":", zorder=1,
                gid="slice_y5")
    ax1.text(8.45, Y_PROBE + 0.12, "y = 5 m", ha="right", va="bottom",
             gid="label_slice")
    for r in ("left", "right"):
        ax1.plot(x_star[r], Y_PROBE, "*", color="black", mec="white", mew=0.6,
                 ms=11, zorder=8, gid=f"fixed_point_{r}")

    ax1.text(GOAL[0] + 0.5, GOAL[1], "Goal", va="center", ha="left",
             gid="label_goal")
    ax1.text(START[0], START[1] - 0.22, "Start", va="top", ha="center",
             gid="label_start")
    ax1.set_xlabel("Lateral position, x (m)")
    ax1.set_ylabel("Forward position, y (m)")
    ax1.set_xticks([2, 4, 6, 8])
    ax1.set_yticks([1, 3, 5, 7, 9])

    keys = [Line2D([0], [0], color=COND_COLOUR[c], lw=2, label=c)
            for c in CONDITIONS]
    keys.append(Line2D([0], [0], color="black", marker="x", ls="", ms=5,
                       label="Collision"))
    leg1 = ax1.legend(handles=keys, loc="upper left", ncol=1, frameon=False,
                      handlelength=1.3, handletextpad=0.5,
                      borderaxespad=0.1, labelspacing=0.3)
    leg1.set_gid("key_conditions")

    # ----------------------------------------------------- panel II
    ax2 = axes_in(x2, top1, w2, h1, "panel_II_manifold")
    order = np.argsort(d["step"])
    sc = ax2.scatter(res["scores"][order, 0], res["scores"][order, 1],
                     c=d["step"][order], cmap="viridis", s=3, alpha=0.4,
                     linewidths=0, zorder=2, rasterized=True)
    for r in ("left", "right"):
        cs = res[r]["curve_scores"]
        ax2.plot(cs[:, 0], cs[:, 1], color="white", lw=3.4, zorder=4,
                 solid_capstyle="round")
        ax2.plot(cs[:, 0], cs[:, 1], color="black", lw=1.6, zorder=5,
                 ls=STYLE[r], gid=f"fixed_point_curve_{r}")
    ax2.set_xlabel(f"PC1 ({100 * res['var'][0]:.0f}%)")
    ax2.set_ylabel(f"PC2 ({100 * res['var'][1]:.0f}%)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xticks([-4, 0, 4])
    ax2.set_yticks([-8, -4, 0, 4, 8])

    curve_keys = [
        Line2D([0], [0], color="black", lw=1.6, ls=LEFT_STYLE,
               label="Fixed points, left route"),
        Line2D([0], [0], color="black", lw=1.6, ls=RIGHT_STYLE,
               label="Fixed points, right route"),
    ]
    leg2 = ax2.legend(handles=curve_keys, loc="upper left", frameon=False,
                      handlelength=2.2, borderaxespad=0.1)
    leg2.set_gid("key_fixed_points")

    cax = ax2.inset_axes([0.06, 0.16, 0.34, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Step", labelpad=1)
    cb.set_ticks([0, 80, 160])
    cb.solids.set_rasterized(True)
    cax.set_gid("colourbar_step")

    # ---------------------------------------------------- panel III
    ax3 = axes_in(x1, top2, W - L_MARGIN - R_MARGIN, h2,
                  "panel_III_restoring_profile")
    xg = np.linspace(1.5, 8.5, 141)
    prof = {}
    for r in ("left", "right"):
        prof[r] = np.array([analysis.lateral_acceleration(x, Y_PROBE, 0.0, vy,
                                                         route=r) for x in xg])

    ax3.axvspan(OBSTACLES[0][0] - OBSTACLE_R, OBSTACLES[1][0] + OBSTACLE_R,
                color="#d9d9d9", lw=0, zorder=0, gid="obstacle_extent")
    ax3.text(5.0, 0.99, "Obstacles", transform=ax3.get_xaxis_transform(),
             ha="center", va="top", color="#404040", gid="label_obstacles")
    ax3.axhline(0.0, color="black", lw=0.5, zorder=1)

    for r in ("left", "right"):
        ax3.plot(xg, prof[r], color="black", lw=1.5, ls=STYLE[r], zorder=3,
                 gid=f"profile_{r}")
        ax3.plot(x_star[r], 0.0, "*", color="black", mec="white", mew=0.6,
                 ms=11, zorder=6, gid=f"fixed_point_{r}_III")

    lo = min(prof["left"].min(), prof["right"].min())
    hi = max(prof["left"].max(), prof["right"].max())
    ax3.set_ylim(lo - 0.32 * (hi - lo), hi + 0.12 * (hi - lo))
    ax3.set_xlim(1.5, 8.5)

    base = ax3.get_ylim()[0]
    for r in ("left", "right"):
        xs_obs = [np.interp(Y_PROBE, np.maximum.accumulate(y), x)
                  for c, rr, s, x, y, ok, hit in episodes(d, r) if ok]
        ax3.plot(xs_obs, np.full(len(xs_obs), base + 0.045 * (hi - lo)), "|",
                 color="black", alpha=0.35, ms=7, mew=0.8, zorder=2,
                 gid=f"observed_crossings_{r}")

    ax3.annotate(f"x* = {x_star['left']:.2f} m\n\u03c4 = {res['left']['tau']:.2f} s\n"
                 f"{clearance['left']:.2f} m clearance",
                 (x_star["left"], 0.0), xytext=(1.6, -0.62), ha="left",
                 va="center", gid="annotation_left",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))
    ax3.annotate(f"x* = {x_star['right']:.2f} m\n\u03c4 = {res['right']['tau']:.2f} s\n"
                 f"{clearance['right']:.2f} m clearance",
                 (x_star["right"], 0.0), xytext=(8.4, 0.62), ha="right",
                 va="center", gid="annotation_right",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))

    ax3.set_xlabel("Lateral position, x (m)")
    ax3.set_ylabel("Commanded $a_x$ (m/s$^2$)")
    ax3.set_xticks([2, 3, 4, 5, 6, 7, 8])

    keys3 = [
        Line2D([0], [0], color="black", lw=1.5, ls=LEFT_STYLE,
               label="Left route"),
        Line2D([0], [0], color="black", lw=1.5, ls=RIGHT_STYLE,
               label="Right route"),
        Line2D([0], [0], color="black", marker="|", ls="", ms=7, alpha=0.5,
               label="Observed crossings at y = 5 m"),
    ]
    leg3 = ax3.legend(handles=keys3, loc="upper center", ncol=3, frameon=False,
                      bbox_to_anchor=(0.5, -0.30), handlelength=2.0,
                      columnspacing=1.6)
    leg3.set_gid("key_profile")

    panel_label(fig, 0.05, 0.05, "I", fig_h)
    panel_label(fig, x2 - 0.55, 0.05, "II", fig_h)
    panel_label(fig, 0.05, top2 - 0.42, "III", fig_h)

    return fig


def main():

    d = load_rollouts()
    res = analyse(d)
    apply_style()
    fig = build(d, res)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{STEM}.png"
    svg = OUT_DIR / f"{STEM}.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)

    print(f"Saved:\n{png.resolve()}\n{svg.resolve()}")


if __name__ == "__main__":
    main()
