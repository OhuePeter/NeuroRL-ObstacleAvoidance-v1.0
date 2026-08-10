"""
==========================================================
Conference Poster Generator
Neural Population Dynamics — Adaptive Obstacle Avoidance in RL

Authors: Peter Ohue, Emily Oby, Gunnar Blohm
Queen's University, Centre for Neuroscience Studies

Produces:
  presentation/poster/output/poster.pdf   (vector, print-ready)
  presentation/poster/output/poster.png   (600 dpi, digital use)

Run from repo root:
  python presentation/poster/create_poster.py
==========================================================
"""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent.parent
FIGS     = ROOT / "paper"
QR_PATH  = ROOT / "presentation" / "qr_codes" / "github_repo_qr.png"
OUT_DIR  = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# Colour palette  (Queen's University blues + accents)
# ----------------------------------------------------------
QB      = "#00355f"   # Queen's deep blue
QB_LITE = "#cce0f0"   # header background tint
TEAL    = "#2a9d8f"   # accent (success / right perturbation)
CORAL   = "#e76f51"   # accent (failure)
SAND    = "#f4f1eb"   # panel background
WHITE   = "#ffffff"
DARK    = "#1a1a2e"
MIDGRAY = "#5a5a6e"
GOLD    = "#f0c040"   # highlight accent

# ----------------------------------------------------------
# Typography
# ----------------------------------------------------------
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype":         "none",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     False,
    "axes.spines.bottom":   False,
})

# ----------------------------------------------------------
# Poster canvas  (A0 portrait: 33.1 × 46.8 inches)
# ----------------------------------------------------------
FIG_W, FIG_H = 33.1, 46.8
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=SAND)

# ----------------------------------------------------------
# Master grid
# ----------------------------------------------------------
#   Row 0: header        (~8% height)
#   Row 1: body          (~82% height)
#   Row 2: footer        (~10% height)
outer = gridspec.GridSpec(
    3, 1,
    figure=fig,
    left=0.01, right=0.99,
    bottom=0.01, top=0.99,
    hspace=0.01,
    height_ratios=[0.08, 0.82, 0.10],
)


# ===========================================================
# HEADER
# ===========================================================
ax_hdr = fig.add_subplot(outer[0])
ax_hdr.set_facecolor(QB)
ax_hdr.set_xlim(0, 1)
ax_hdr.set_ylim(0, 1)
ax_hdr.axis("off")

# Title
ax_hdr.text(
    0.50, 0.72,
    "Neural Population Dynamics Reveal Internal Representations\n"
    "Underlying Adaptive Obstacle Avoidance in Reinforcement Learning",
    ha="center", va="center",
    fontsize=36, fontweight="bold", color=WHITE,
    transform=ax_hdr.transAxes,
)
# Authors
ax_hdr.text(
    0.50, 0.30,
    "Peter Ohue   •   Emily Oby   •   Gunnar Blohm",
    ha="center", va="center",
    fontsize=24, color=QB_LITE, style="italic",
    transform=ax_hdr.transAxes,
)
# Institution
ax_hdr.text(
    0.50, 0.10,
    "Centre for Neuroscience Studies  |  Queen's University, Kingston, ON, Canada",
    ha="center", va="center",
    fontsize=18, color=GOLD,
    transform=ax_hdr.transAxes,
)


# ===========================================================
# BODY  — 3 columns
# ===========================================================
body = gridspec.GridSpecFromSubplotSpec(
    1, 3,
    subplot_spec=outer[1],
    wspace=0.025,
    width_ratios=[1, 1, 1],
)


# -----------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------

def panel_bg(ax, color=WHITE, radius=0.012):
    """White rounded background for a data panel."""
    ax.set_facecolor(color)
    for sp in ax.spines.values():
        sp.set_visible(False)


def section_heading(ax_parent, fig, text, color=QB):
    """Draw a filled heading bar spanning an axes bounding box."""
    bb = ax_parent.get_position()
    rect_ax = fig.add_axes([bb.x0, bb.y1 - 0.002, bb.width, 0.020],
                           facecolor=color)
    rect_ax.axis("off")
    rect_ax.text(0.015, 0.5, text, va="center", ha="left",
                 fontsize=20, fontweight="bold", color=WHITE,
                 transform=rect_ax.transAxes)
    return rect_ax


def load_img(name):
    p = FIGS / name
    if p.exists():
        return np.asarray(Image.open(str(p)).convert("RGB"))
    return None


def show_fig(ax, name, title="", title_size=16):
    img = load_img(name)
    if img is not None:
        ax.imshow(img, aspect="auto")
    else:
        ax.text(0.5, 0.5, f"[{name}]", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color=MIDGRAY)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold",
                     color=DARK, pad=5)


def wrapped_text(ax, x, y, text, width=55, fontsize=16, color=DARK,
                 ha="left", va="top", bold=False, linespacing=1.4):
    lines = textwrap.wrap(text, width)
    joined = "\n".join(lines)
    ax.text(x, y, joined, ha=ha, va=va, fontsize=fontsize, color=color,
            fontweight="bold" if bold else "normal",
            linespacing=linespacing,
            transform=ax.transAxes)


def bullet(ax, y, text, indent=0.03, fontsize=15, color=DARK, width=52):
    lines = textwrap.wrap(text, width)
    ax.text(indent, y, "-  " + lines[0], ha="left", va="top",
            fontsize=fontsize, color=color,
            transform=ax.transAxes)
    for i, line in enumerate(lines[1:], 1):
        ax.text(indent + 0.055, y - i * 0.028, line, ha="left", va="top",
                fontsize=fontsize, color=color,
                transform=ax.transAxes)
    return y - (len(lines) * 0.028 + 0.022)


def key_result_box(ax, x, y, w, h, title, body_text, color=TEAL):
    """Coloured callout box for a key result."""
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((x, y - h), w, h,
                         boxstyle="round,pad=0.01",
                         linewidth=2, edgecolor=color,
                         facecolor=color + "22",
                         transform=ax.transAxes, zorder=3)
    ax.add_patch(box)
    ax.text(x + 0.01, y - 0.008, title, ha="left", va="top",
            fontsize=15, fontweight="bold", color=color,
            transform=ax.transAxes, zorder=4)
    ax.text(x + 0.01, y - 0.036, textwrap.fill(body_text, 28),
            ha="left", va="top", fontsize=13, color=DARK,
            transform=ax.transAxes, zorder=4, linespacing=1.3)


# ===========================================================
# COLUMN 1 — Background · Methods · Key Findings
# ===========================================================
col1 = gridspec.GridSpecFromSubplotSpec(
    6, 1,
    subplot_spec=body[0],
    hspace=0.03,
    height_ratios=[0.13, 0.015, 0.22, 0.015, 0.35, 0.27],
)

# ── Background / Motivation ──────────────────────────────
ax_bg = fig.add_subplot(col1[0])
panel_bg(ax_bg)
ax_bg.set_xlim(0, 1); ax_bg.set_ylim(0, 1); ax_bg.axis("off")
wrapped_text(ax_bg, 0.02, 0.97,
    "BACKGROUND & MOTIVATION", fontsize=19, color=QB, bold=True)
y = 0.80
y = bullet(ax_bg, y,
    "Motor recovery after stroke is highly heterogeneous. Existing therapies "
    "lack mechanistic precision to tailor training to the individual patient.",
    fontsize=14)
y = bullet(ax_bg, y,
    "Reinforcement learning (RL) controllers trained by reward discover adaptive "
    "motor strategies analogously to biological motor learning.",
    fontsize=14)
y = bullet(ax_bg, y,
    "Motor cortex population activity evolves on compact low-dimensional "
    "manifolds. Whether RL controllers develop the same organisation is unknown.",
    fontsize=14)
wrapped_text(ax_bg, 0.02, y - 0.01,
    "We ask: does reward-driven learning produce structured neural population "
    "representations parallel to biological motor cortex?",
    fontsize=14, color=QB, bold=True, width=54)

# ── separator ────────────────────────────────────────────
ax_sep1 = fig.add_subplot(col1[1]); ax_sep1.axis("off")
ax_sep1.plot([0, 1], [0.5, 0.5], color=QB, linewidth=2,
             transform=ax_sep1.transAxes)

# ── Methods ──────────────────────────────────────────────
ax_meth = fig.add_subplot(col1[2])
panel_bg(ax_meth)
ax_meth.set_xlim(0, 1); ax_meth.set_ylim(0, 1); ax_meth.axis("off")
wrapped_text(ax_meth, 0.02, 0.97,
    "METHODS", fontsize=19, color=QB, bold=True)
y = 0.84
y = bullet(ax_meth, y,
    "2-D point-mass agent navigates obstacle corridor to a fixed goal under "
    "Newtonian dynamics (Gymnasium environment).",
    fontsize=13, width=50)
y = bullet(ax_meth, y,
    "PPO actor-critic controller trained for 3 × 10⁶ steps; "
    "two hidden layers × 256 units (feedforward MLP).",
    fontsize=13, width=50)
y = bullet(ax_meth, y,
    "7 perturbation conditions: control (P0), leftward (L1–L3) "
    "and rightward (R1–R3) lateral force impulses.",
    fontsize=13, width=50)
y = bullet(ax_meth, y,
    "Population analysis: PCA · K-means · linear decoding applied to "
    "hidden-layer activations during evaluation.",
    fontsize=13, width=50)

# ── separator ────────────────────────────────────────────
ax_sep2 = fig.add_subplot(col1[3]); ax_sep2.axis("off")
ax_sep2.plot([0, 1], [0.5, 0.5], color=QB, linewidth=2,
             transform=ax_sep2.transAxes)

# ── Schematic (Fig 1) ─────────────────────────────────────
ax_sch = fig.add_subplot(col1[4])
panel_bg(ax_sch)
show_fig(ax_sch, "fig1_schematic.png",
         title="Fig 1  ·  Task & Control Architecture", title_size=15)

# ── Key Findings callout boxes ────────────────────────────
ax_kf = fig.add_subplot(col1[5])
panel_bg(ax_kf)
ax_kf.set_xlim(0, 1); ax_kf.set_ylim(0, 1); ax_kf.axis("off")
wrapped_text(ax_kf, 0.02, 0.97,
    "KEY FINDINGS", fontsize=19, color=QB, bold=True)
key_result_box(ax_kf, 0.02, 0.87, 0.46, 0.22,
    "74.8 % variance in 2 PCs",
    "Hidden-layer activity compresses into a compact low-dimensional manifold — "
    "mirroring motor cortex.",
    color=QB)
key_result_box(ax_kf, 0.52, 0.87, 0.46, 0.22,
    "4 phase-aligned clusters",
    "K-means reveals movement-phase structure (approach · perturbation · "
    "correction · arrival).",
    color=TEAL)
key_result_box(ax_kf, 0.02, 0.58, 0.46, 0.22,
    "Direction asymmetry",
    "Rightward perturbations are more disruptive due to learned left-biased "
    "corridor trajectory.",
    color=CORAL)
key_result_box(ax_kf, 0.52, 0.58, 0.46, 0.22,
    "Linear decodability",
    "Extreme conditions linearly separable; mild conditions geometrically "
    "indistinct — matching biological findings.",
    color="#8338ec")


# ===========================================================
# COLUMN 2 — Results: Trajectories + Velocity
# ===========================================================
col2 = gridspec.GridSpecFromSubplotSpec(
    5, 1,
    subplot_spec=body[1],
    hspace=0.04,
    height_ratios=[0.025, 0.34, 0.015, 0.57, 0.015],
)

ax_res_hdr = fig.add_subplot(col2[0])
ax_res_hdr.set_facecolor(TEAL); ax_res_hdr.axis("off")
ax_res_hdr.text(0.5, 0.5, "RESULTS", ha="center", va="center",
                fontsize=22, fontweight="bold", color=WHITE,
                transform=ax_res_hdr.transAxes)

# Trajectory figure
ax_traj = fig.add_subplot(col2[1])
panel_bg(ax_traj)
show_fig(ax_traj, "fig2_trajectories.png",
         title="Fig 2  ·  Adaptive Route Geometry Under Lateral Perturbation",
         title_size=15)

ax_traj_cap = fig.add_subplot(col2[2])
ax_traj_cap.set_facecolor(QB_LITE); ax_traj_cap.axis("off")
ax_traj_cap.text(0.01, 0.82,
    "Solid = success  ·  Dashed = failure  ·  Shaded band = perturbation zone  ·  "
    "Rightward conditions (R2, R3) show highest failure rate",
    ha="left", va="center", fontsize=13, color=DARK, style="italic",
    transform=ax_traj_cap.transAxes, wrap=True)

# Velocity figure
ax_vel = fig.add_subplot(col2[3])
panel_bg(ax_vel)
show_fig(ax_vel, "fig3_velocity.png",
         title="Fig 3  ·  Velocity Profiles: Proportional Correction, Preserved Speed",
         title_size=15)

ax_vel_cap = fig.add_subplot(col2[4])
ax_vel_cap.set_facecolor(QB_LITE); ax_vel_cap.axis("off")
ax_vel_cap.text(0.01, 0.82,
    "(A) Lateral velocity diverges by direction during perturbation window (t = 50–55)  ·  "
    "(B) Total speed is invariant — adaptation is purely trajectory-level",
    ha="left", va="center", fontsize=13, color=DARK, style="italic",
    transform=ax_vel_cap.transAxes)


# ===========================================================
# COLUMN 3 — Results: PCA · Decoding · Learning  +  Discussion
# ===========================================================
col3 = gridspec.GridSpecFromSubplotSpec(
    8, 1,
    subplot_spec=body[2],
    hspace=0.03,
    height_ratios=[0.025, 0.22, 0.005, 0.18, 0.005, 0.15, 0.005, 0.39],
)

ax_res_hdr3 = fig.add_subplot(col3[0])
ax_res_hdr3.set_facecolor(TEAL); ax_res_hdr3.axis("off")
ax_res_hdr3.text(0.5, 0.5, "RESULTS (cont.)", ha="center", va="center",
                 fontsize=22, fontweight="bold", color=WHITE,
                 transform=ax_res_hdr3.transAxes)

# PCA
ax_pca = fig.add_subplot(col3[1])
panel_bg(ax_pca)
show_fig(ax_pca, "fig4_pca.png",
         title="Fig 4  ·  Low-Dimensional Manifold (PC1 = 46.8 %, PC2 = 28.0 %)",
         title_size=14)

ax_pca_cap = fig.add_subplot(col3[2])
ax_pca_cap.set_facecolor(QB_LITE); ax_pca_cap.axis("off")
ax_pca_cap.text(0.01, 0.8,
    "(A) Coloured by condition  ·  (B) K-means K=4 reveals phase-aligned clusters",
    ha="left", va="center", fontsize=12, color=DARK, style="italic",
    transform=ax_pca_cap.transAxes)

# Decoding
ax_dec = fig.add_subplot(col3[3])
panel_bg(ax_dec)
show_fig(ax_dec, "fig5_decoding.png",
         title="Fig 5  ·  Linear Decoder Confusion Matrix",
         title_size=14)

ax_dec_cap = fig.add_subplot(col3[4])
ax_dec_cap.set_facecolor(QB_LITE); ax_dec_cap.axis("off")
ax_dec_cap.text(0.01, 0.8,
    "L3 decoded at 70 %, R3 at 62 %  ·  Mild conditions overlap geometrically",
    ha="left", va="center", fontsize=12, color=DARK, style="italic",
    transform=ax_dec_cap.transAxes)

# Learning
ax_lrn = fig.add_subplot(col3[5])
panel_bg(ax_lrn)
show_fig(ax_lrn, "fig6_learning.png",
         title="Fig 6  ·  Training Convergence & Evaluation Stability",
         title_size=14)

ax_lrn_cap = fig.add_subplot(col3[6])
ax_lrn_cap.set_facecolor(QB_LITE); ax_lrn_cap.axis("off")
ax_lrn_cap.text(0.01, 0.8,
    "Rapid convergence in first 1 × 10⁶ steps  ·  Stable plateau validates analyses",
    ha="left", va="center", fontsize=12, color=DARK, style="italic",
    transform=ax_lrn_cap.transAxes)

# Discussion + Future Work
ax_disc = fig.add_subplot(col3[7])
panel_bg(ax_disc)
ax_disc.set_xlim(0, 1); ax_disc.set_ylim(0, 1); ax_disc.axis("off")
wrapped_text(ax_disc, 0.02, 0.97,
    "DISCUSSION & FUTURE WORK", fontsize=18, color=QB, bold=True)
y = 0.88
y = bullet(ax_disc, y,
    "Adaptation via trajectory correction with preserved speed mirrors optimal "
    "feedback control principles in biological reaching.",
    fontsize=13, width=48)
y = bullet(ax_disc, y,
    "74.8 % variance in 2 PCs matches motor cortex manifold compression "
    "documented in non-human primates and humans.",
    fontsize=13, width=48)
y = bullet(ax_disc, y,
    "Phase-aligned cluster structure parallels sequential neural population "
    "states identified during biological reaching.",
    fontsize=13, width=48)
y = bullet(ax_disc, y,
    "Direction-dependent failure asymmetry reflects learned geometry — "
    "not an inherent algorithmic limitation.",
    fontsize=13, width=48)

wrapped_text(ax_disc, 0.02, y - 0.01,
    "FUTURE DIRECTIONS", fontsize=16, color=TEAL, bold=True)
y2 = y - 0.055
y2 = bullet(ax_disc, y2,
    "Multi-seed validation to confirm representational stability.",
    fontsize=13, color=MIDGRAY, width=48)
y2 = bullet(ax_disc, y2,
    "Recurrent architectures (LSTM) for temporal perturbation history.",
    fontsize=13, color=MIDGRAY, width=48)
y2 = bullet(ax_disc, y2,
    "Human EMG manifold comparison in stroke rehabilitation protocol.",
    fontsize=13, color=MIDGRAY, width=48)
y2 = bullet(ax_disc, y2,
    "Closed-loop VR platform using latent geometry as adaptive-difficulty driver.",
    fontsize=13, color=MIDGRAY, width=48)


# ===========================================================
# FOOTER
# ===========================================================
ftr = gridspec.GridSpecFromSubplotSpec(
    1, 3,
    subplot_spec=outer[2],
    wspace=0.02,
    width_ratios=[0.50, 0.34, 0.16],
)

# ── References ───────────────────────────────────────────
ax_ref = fig.add_subplot(ftr[0])
ax_ref.set_facecolor(QB); ax_ref.axis("off")
refs = (
    "Selected references:  "
    "[1] Churchland et al., Nature 2012  ·  "
    "[2] Gallego et al., Neuron 2017  ·  "
    "[3] Sadtler et al., Nature 2014  ·  "
    "[4] Schulman et al., arXiv 2017  ·  "
    "[5] Pruszynski & Scott, Exp Brain Res 2012  ·  "
    "[6] Oby et al., PNAS 2019  ·  "
    "[7] Todorov & Jordan, Nat Neurosci 2002"
)
ax_ref.text(0.01, 0.6, refs,
            ha="left", va="center", fontsize=11, color=QB_LITE,
            transform=ax_ref.transAxes, wrap=True)

# ── Acknowledgments ───────────────────────────────────────
ax_ack = fig.add_subplot(ftr[1])
ax_ack.set_facecolor(QB); ax_ack.axis("off")
ax_ack.text(0.5, 0.7,
    "Supported by the Connected Minds Program,\n"
    "Canada First Research Excellence Fund (Grant CFREF-2022-00010)",
    ha="center", va="center", fontsize=14, color=GOLD,
    fontweight="bold", transform=ax_ack.transAxes)
ax_ack.text(0.5, 0.25,
    "Code & data: github.com/OhuePeter/NeuroRL-ObstacleAvoidance-v1.0",
    ha="center", va="center", fontsize=13, color=QB_LITE,
    transform=ax_ack.transAxes)

# ── QR code ───────────────────────────────────────────────
ax_qr = fig.add_subplot(ftr[2])
ax_qr.set_facecolor(QB); ax_qr.axis("off")
if QR_PATH.exists():
    qr_img = np.asarray(Image.open(str(QR_PATH)).convert("RGB"))
    ax_qr.imshow(qr_img, aspect="auto",
                 extent=[0.05, 0.85, 0.05, 0.90])
ax_qr.text(0.50, 0.96, "Scan for code & data",
           ha="center", va="top", fontsize=14, color=QB_LITE,
           transform=ax_qr.transAxes)


# ===========================================================
# Thin column dividers
# ===========================================================
for xpos in [0.3433, 0.6766]:
    line = Line2D([xpos, xpos], [0.09, 0.91],
                  transform=fig.transFigure,
                  color=QB, linewidth=1.5, alpha=0.25)
    fig.add_artist(line)


# ===========================================================
# Save
# ===========================================================
pdf_out = OUT_DIR / "poster.pdf"
png_out = OUT_DIR / "poster.png"

fig.savefig(str(pdf_out), format="pdf", dpi=150, bbox_inches="tight",
            facecolor=SAND)
fig.savefig(str(png_out), format="png", dpi=150, bbox_inches="tight",
            facecolor=SAND)

print(f"Poster saved:")
print(f"  PDF → {pdf_out}")
print(f"  PNG → {png_out}")
