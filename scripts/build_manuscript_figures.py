"""
==========================================================
Build Manuscript Figures 2-6 and Their Statistics

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Redraws manuscript Figures 2-6 as native vector graphics (live
Arial 10 pt text in the SVG) from the SAME rollouts as the
COSYNE figure (scripts/cosyne_figure_data.py), so the manuscript
no longer mixes runs:

  Fig 2  reach trajectories, all conditions, by detour route
  Fig 3  lateral-velocity deviation from P0, and speed
  Fig 4  PC1-PC2 of hidden activity by condition and by step
  Fig 5  decoder confusion matrix (held-out episodes)
  Fig 6  learning curves (evaluation reward by checkpoint and the
         training-reward curve of the run that produced them)

It also prints every number the Results text quotes. The
decoder is trained and tested on different EPISODES (grouped
cross-validation); timesteps from one episode never appear on
both sides.

Usage
-----
python -m scripts.build_manuscript_figures
==========================================================
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.build_cosyne_figure import (
    COND_COLOUR, GOAL, GOAL_R, OBSTACLES, OBSTACLE_R, START, apply_style,
    episodes,
)
from scripts.cosyne_figure_data import CONDITIONS, analyse, load_rollouts

PAPER = Path("paper")
SVG_DIR = PAPER / "figures"
LOG_DIR = Path("experiments/version_1_0/logs/PPO_9")
EVAL_NPZ = Path("experiments/version_1_0/evaluation/evaluations.npz")

W = 6.5
ONSET = 40
WINDOW = (54, 100)       # steps after the impulse has ended


def save(fig, stem):

    SVG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(SVG_DIR / f"{stem}.svg", facecolor="white")
    plt.close(fig)
    print(f"  saved {stem}")


def label(fig, x_in, y_in, text, fig_h):
    fig.text(x_in / W, 1 - y_in / fig_h, text, fontsize=10, fontweight="bold",
             ha="left", va="top", gid=f"label_{text}")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def cond_keys(extra=()):
    keys = [Line2D([0], [0], color=COND_COLOUR[c], lw=2, label=c)
            for c in CONDITIONS]
    return keys + list(extra)


# ------------------------------------------------------------------ Fig 2
def fig2(d):

    w, h = 2.70, 2.70 * 9.2 / 7.0
    fig_h = 0.30 + h + 0.95
    fig = plt.figure(figsize=(W, fig_h))
    y_lo = float(np.median(d["y"][(d["condition"] == "P0") & (d["step"] == ONSET)]))
    y_hi = float(np.median(d["y"][(d["condition"] == "P0") & (d["step"] == ONSET + 13)]))

    for i, route in enumerate(("left", "right")):

        x0 = 0.62 + i * (w + 0.42)
        ax = fig.add_axes([x0 / W, 1 - (0.30 + h) / fig_h, w / W, h / fig_h])
        ax.set_gid(f"panel_{'I' if i == 0 else 'II'}_{route}_detour")
        clean(ax)
        ax.set_xlim(1.5, 8.5)
        ax.set_ylim(0.4, 9.6)
        ax.set_aspect("equal")
        ax.axhspan(y_lo, y_hi, color="#e6e6e6", lw=0, zorder=0)
        for ox, oy in OBSTACLES:
            ax.add_patch(plt.Circle((ox, oy), OBSTACLE_R, fc="#bdbdbd",
                                    ec="#737373", lw=0.8, zorder=2))
        ax.add_patch(plt.Circle(GOAL, GOAL_R, fc="#c6dbef", ec="#2171b5",
                                lw=0.8, zorder=2))
        ax.plot(*START, "o", color="black", ms=4, zorder=6)
        for c, r, step, x, y, ok, hit in episodes(d, route):
            ax.plot(x, y, color=COND_COLOUR[c], lw=0.7, alpha=0.5, zorder=3,
                    ls="-" if ok else (0, (3, 1.5)))
            if hit:
                ax.plot(x[-1], y[-1], "x", color="black", ms=4, mew=0.9, zorder=7)
        ax.set_title(f"{route.capitalize()} detour", pad=4)
        ax.set_xlabel("Lateral position, x (m)")
        ax.set_xticks([2, 4, 6, 8])
        ax.set_yticks([1, 3, 5, 7, 9])
        if i == 0:
            ax.set_ylabel("Forward position, y (m)")
        else:
            ax.set_yticklabels([])
        label(fig, x0 - 0.55, 0.05, "I" if i == 0 else "II", fig_h)

    keys = cond_keys([Line2D([0], [0], color="black", marker="x", ls="", ms=5,
                             label="Collision")])
    fig.legend(handles=keys, loc="lower center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.2, handletextpad=0.4,
               columnspacing=0.9)
    save(fig, "fig2_trajectories")


# ------------------------------------------------------------------ Fig 3
def delta_vx(d):
    """Mean +- SD of vx minus the same-route P0 mean, over successful episodes."""

    steps = np.arange(1, 171)
    base = {}
    for r in ("left", "right"):
        rows = [np.interp(steps, s, vx) for c, rr, s, x, y, vx, ok, hit in
                episodes_vx(d, r) if c == "P0"]
        base[r] = np.mean(rows, axis=0)

    out = {}
    for c in CONDITIONS:
        rows = []
        for cc, r, s, x, y, vx, ok, hit in episodes_vx(d):
            if cc == c and ok:
                rows.append(np.interp(steps, s, vx) - base[r])
        out[c] = (np.mean(rows, axis=0), np.std(rows, axis=0), len(rows))

    return steps, out


def episodes_vx(d, route=None):

    outcome = {(c, e): (s, k) for c, e, s, k in zip(
        d["o_condition"], d["o_episode"], d["o_success"], d["o_collision"])}
    for c in CONDITIONS:
        m_c = d["condition"] == c
        for e in np.unique(d["episode"][m_c]):
            m = m_c & (d["episode"] == e)
            r = d["route"][m][0]
            if route is not None and r != route:
                continue
            s, k = outcome[(c, e)]
            yield c, r, d["step"][m], d["x"][m], d["y"][m], d["vx"][m], bool(s), bool(k)


def fig3(d):

    steps, dv = delta_vx(d)
    fig_h = 2.75
    fig = plt.figure(figsize=(W, fig_h))
    ax1 = fig.add_axes([0.70 / W, 0.62 / fig_h, 3.55 / W, 1.82 / fig_h])
    ax2 = fig.add_axes([4.95 / W, 0.62 / fig_h, 1.40 / W, 1.82 / fig_h])
    ax1.set_gid("panel_I_delta_vx")
    ax2.set_gid("panel_II_speed")
    for ax in (ax1, ax2):
        clean(ax)

    ax1.axvspan(ONSET, ONSET + 13, color="#e6e6e6", lw=0, zorder=0)
    for c in CONDITIONS:
        m, s, n = dv[c]
        ax1.plot(steps, m, color=COND_COLOUR[c], lw=1.2, label=c, zorder=3)
        ax1.fill_between(steps, m - s, m + s, color=COND_COLOUR[c], alpha=0.15,
                         lw=0, zorder=2)
    ax1.axhline(0, color="black", lw=0.5, zorder=1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("$\\Delta v_x$ from P0 (m/s)")
    ax1.set_xlim(0, 170)

    for c in CONDITIONS:
        rows = [np.interp(np.arange(1, 171), s, np.hypot(vx, vy)) for c2, r, s, vx, vy in
                speed_rows(d) if c2 == c]
        ax2.plot(np.arange(1, 171), np.mean(rows, axis=0), color=COND_COLOUR[c], lw=1.2)
    ax2.axhline(1.0, color="black", lw=0.5, ls=":")
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(0, 170)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_xticks([0, 80, 160])

    fig.legend(handles=cond_keys(), loc="upper center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.2, handletextpad=0.4,
               columnspacing=0.9)
    label(fig, 0.05, 0.38, "I", fig_h)
    label(fig, 4.40, 0.38, "II", fig_h)
    save(fig, "fig3_velocity")

    return steps, dv


def speed_rows(d):
    for c in CONDITIONS:
        m_c = d["condition"] == c
        for e in np.unique(d["episode"][m_c]):
            m = m_c & (d["episode"] == e)
            yield c, d["route"][m][0], d["step"][m], d["vx"][m], d["vy"][m]


# ------------------------------------------------------------------ Fig 4
def fig4(d, res):

    fig_h = 3.35
    fig = plt.figure(figsize=(W, fig_h))
    s = res["scores"]
    order = np.argsort(d["step"])
    axes = []
    for i in range(2):
        ax = fig.add_axes([(0.62 + i * 3.05) / W, 0.62 / fig_h, 2.35 / W, 2.30 / fig_h])
        ax.set_gid(f"panel_{'I' if i == 0 else 'II'}_pca")
        clean(ax)
        axes.append(ax)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(s))
    for c in CONDITIONS:
        m = d["condition"][perm] == c
        axes[0].scatter(s[perm][m, 0], s[perm][m, 1], s=2, color=COND_COLOUR[c],
                        alpha=0.35, linewidths=0, rasterized=True)
    sc = axes[1].scatter(s[order, 0], s[order, 1], c=d["step"][order], cmap="viridis",
                         s=2, alpha=0.4, linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.06, pad=0.03)
    cb.set_label("Step")
    cb.set_ticks([0, 80, 160])
    for ax in axes:
        ax.set_xlabel(f"PC1 ({100 * res['var'][0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({100 * res['var'][1]:.0f}%)")
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([-8, -4, 0, 4, 8])
        ax.set_aspect("equal", adjustable="datalim")
    label(fig, 0.05, 0.38, "I", fig_h)
    label(fig, 3.30, 0.38, "II", fig_h)
    fig.legend(handles=cond_keys(), loc="upper center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.2, handletextpad=0.4,
               columnspacing=0.9)
    save(fig, "fig4_pca")


# ------------------------------------------------------------------ Fig 5
def decoding(d):
    """Held-out-episode decoding of condition from hidden activity."""

    cond_idx = np.array([CONDITIONS.index(c) for c in d["condition"]])
    group = cond_idx * 1000 + d["episode"]
    step = d["step"]
    X = d["latent"]

    results = {}
    for name, mask in (
        ("post-impulse window", (step >= WINDOW[0]) & (step <= WINDOW[1])),
        ("all steps (every 3rd)", (step % 3 == 0)),
    ):
        Xm, ym, gm = X[mask], cond_idx[mask], group[mask]
        pred = np.zeros_like(ym)
        for tr, te in GroupKFold(n_splits=5).split(Xm, ym, gm):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=5000))
            clf.fit(Xm[tr], ym[tr])
            pred[te] = clf.predict(Xm[te])
        conf = np.zeros((7, 7))
        for t, p in zip(ym, pred):
            conf[t, p] += 1
        results[name] = (conf / conf.sum(1, keepdims=True), float((pred == ym).mean()),
                         int(mask.sum()))
        results[name + "_raw"] = (pred, ym, step[mask])

    # route decodability, same protocol
    route = (d["route"] == "right").astype(int)
    mask = (step % 3 == 0)
    pred = np.zeros(mask.sum(), dtype=int)
    Xm, ym, gm = X[mask], route[mask], group[mask]
    for tr, te in GroupKFold(n_splits=5).split(Xm, ym, gm):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(Xm[tr], ym[tr])
        pred[te] = clf.predict(Xm[te])
    results["route"] = float((pred == ym).mean())

    return results


def fig5(conf):

    fig_h = 3.55
    fig = plt.figure(figsize=(4.3, fig_h))
    ax = fig.add_axes([0.62 / 4.3, 0.62 / fig_h, 3.05 / 4.3, 2.75 / fig_h])
    ax.set_gid("panel_confusion")
    im = ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if conf[i, j] > 0.55 else "black")
    ax.set_xticks(range(7))
    ax.set_xticklabels(CONDITIONS)
    ax.set_yticks(range(7))
    ax.set_yticklabels(CONDITIONS)
    ax.set_xlabel("Predicted condition")
    ax.set_ylabel("True condition")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Proportion of timesteps")
    fig.savefig(PAPER / "fig5_decoding.png", dpi=300, facecolor="white")
    fig.savefig(SVG_DIR / "fig5_decoding.svg", facecolor="white")
    plt.close(fig)
    print("  saved fig5_decoding")


# ------------------------------------------------------------------ Fig 6
def learning():

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(LOG_DIR), size_guidance={"scalars": 0})
    acc.Reload()
    ev = acc.Scalars("rollout/ep_rew_mean")
    step = np.array([e.step for e in ev])
    rew = np.array([e.value for e in ev])
    smooth = pd.Series(rew).ewm(span=15, adjust=False).mean().to_numpy()

    z = np.load(EVAL_NPZ)
    return step, rew, smooth, z["timesteps"], z["results"]


def fig6(lr):

    step, rew, smooth, ts, res = lr
    fig_h = 2.75
    fig = plt.figure(figsize=(W, fig_h))
    ax1 = fig.add_axes([0.80 / W, 0.62 / fig_h, 2.45 / W, 1.95 / fig_h])
    ax2 = fig.add_axes([4.05 / W, 0.62 / fig_h, 2.30 / W, 1.95 / fig_h])
    ax1.set_gid("panel_I_eval")
    ax2.set_gid("panel_II_training")
    for ax in (ax1, ax2):
        clean(ax)

    m, s = res.mean(1), res.std(1)
    ax1.plot(ts / 1e6, m, color="#08306b", lw=1.4, marker="o", ms=3)
    ax1.fill_between(ts / 1e6, m - s, m + s, color="#08306b", alpha=0.18, lw=0)
    best = ts[np.argmax(m)]
    ax1.axvline(best / 1e6, color="black", lw=0.6, ls=":")
    ax1.text(best / 1e6 + 0.06, 280.15, "best checkpoint\n(800k steps)", ha="left",
             va="bottom")
    ax1.set_xlabel("Training steps (millions)")
    ax1.set_ylabel("Evaluation reward")
    ax1.set_ylim(280.0, 282.4)

    ax2.plot(step / 1e6, rew, color="#9ecae1", lw=0.8, label="Rollout mean")
    ax2.plot(step / 1e6, smooth, color="#cb181d", lw=1.4, label="Smoothed")
    ax2.axvline(best / 1e6, color="black", lw=0.6, ls=":")
    ax2.set_xlabel("Training steps (millions)")
    ax2.set_ylabel("Training reward")
    ax2.legend(frameon=False, loc="lower right", handlelength=1.4)
    label(fig, 0.05, 0.38, "I", fig_h)
    label(fig, 3.55, 0.38, "II", fig_h)
    save(fig, "fig6_learning")


# ------------------------------------------------------------------ stats
def summarise(d, res, dv, dec, lr):

    steps, deltas = dv
    print("\n=== SUMMARY NUMBERS ===")

    print("\nspeed: first step where mean speed >= 0.99 (all conditions):")
    sp = []
    for c in CONDITIONS:
        rows = [np.interp(np.arange(1, 171), s, np.hypot(vx, vy))
                for c2, r, s, vx, vy in speed_rows(d) if c2 == c]
        mean = np.mean(rows, axis=0)
        sp.append(int(np.argmax(mean >= 0.99)) + 1)
    print("  ", dict(zip(CONDITIONS, sp)))

    print("\ndelta vx (pooled routes, successful episodes; n per condition):")
    for c in CONDITIONS:
        m, s, n = deltas[c]
        w = slice(ONSET - 1, 100)
        i_pk = int(np.argmax(np.abs(m[w]))) + ONSET
        pk = m[i_pk - 1]
        after = m[i_pk:130]
        opp = after.min() if pk > 0 else after.max()
        settle = np.where(np.abs(m[i_pk:]) < 0.02)[0]
        print(f"  {c}: n={n} peak {pk:+.3f} at step {i_pk}; later opposite-sign extreme {opp:+.3f}; "
              f"within 0.02 of P0 from step {i_pk + int(settle[0]) if len(settle) else 'never'}")

    print("\nPCA variance:", (100 * res["var"]).round(1), "total", round(100 * res["var"].sum(), 1))

    for name in ("post-impulse window", "all steps (every 3rd)"):
        conf, acc, n = dec[name]
        print(f"\ndecoder ({name}), n={n}, held-out episodes, chance 0.143: accuracy {acc:.3f}")
        print("  diagonal:", {c: round(conf[i, i], 2) for i, c in enumerate(CONDITIONS)})
        for i, c in enumerate(CONDITIONS):
            j = int(np.argmax(np.where(np.arange(7) == i, -1, conf[i])))
            print(f"  {c}: most common error -> {CONDITIONS[j]} ({conf[i, j]:.2f})")
    print(f"\nroute decoding accuracy (held-out episodes, chance 0.5): {dec['route']:.3f}")

    pred, ym, st = dec["all steps (every 3rd)_raw"]
    for lo, hi, tag in ((0, ONSET, "before the impulse (steps < 40)"),
                        (ONSET, WINDOW[0], "during the impulse (40-53)"),
                        (WINDOW[0], 400, "after the impulse (>= 54)")):
        m = (st >= lo) & (st < hi)
        print(f"  condition accuracy {tag}: {(pred[m] == ym[m]).mean():.3f} (n={m.sum()})")

    side = {0: "P0", 1: "L", 2: "L", 3: "L", 4: "R", 5: "R", 6: "R"}
    pw, yw, _ = dec["post-impulse window_raw"]
    same_side = np.mean([side[p] == side[t] for p, t in zip(pw, yw)])
    print(f"  post-impulse window: predicted on the correct side (L / P0 / R): {same_side:.3f}")

    print("\nper-route K=4 clustering of PC1-PC2 (clusters ordered by mean step):")
    from sklearn.cluster import KMeans
    for r in ("left", "right"):
        m = d["route"] == r
        km = KMeans(n_clusters=4, random_state=0, n_init=10).fit(res["scores"][m])
        lab = km.labels_
        stp = d["step"][m]
        order = np.argsort([stp[lab == k].mean() for k in range(4)])
        for rank, k in enumerate(order):
            sel = lab == k
            shares = [(d["condition"][m][sel] == c).mean() for c in CONDITIONS]
            print(f"  {r} cluster {rank + 1}: steps {np.percentile(stp[sel], 25):.0f}-"
                  f"{np.percentile(stp[sel], 75):.0f} (mean {stp[sel].mean():.0f}); "
                  f"condition share min {min(shares):.2f} max {max(shares):.2f}")

    step, rew, smooth, ts, ev = lr
    print("\nlearning: eval means by checkpoint:", dict(zip(ts.tolist(), ev.mean(1).round(2).tolist())))
    print("  eval std by checkpoint:", dict(zip(ts.tolist(), ev.std(1).round(2).tolist())))
    plateau = float(np.mean(rew[-100:]))
    print(f"  training reward: first {rew[0]:.0f}; mean of last 100 points {plateau:.1f}; max {rew.max():.1f}")
    for frac in (0.0, 0.5, 0.9):
        thr = (1 - frac) * rew[0] + frac * plateau if frac else 0.0
        idx = np.where(smooth >= thr)[0]
        print(f"  smoothed reward first >= {thr:.0f} at step {int(step[idx[0]]) if len(idx) else None}")
    idx = np.where(smooth >= 0.9 * plateau)[0]
    print("  step where smoothed reward first reaches 90% of plateau:", int(step[idx[0]]) if len(idx) else None)
    print("  std of smoothed reward after 1M steps:", round(float(np.std(smooth[step >= 1e6])), 2))


def main():

    d = load_rollouts()
    res = analyse(d)
    apply_style()

    print("Drawing figures ...")
    fig2(d)
    dv = fig3(d)
    fig4(d, res)
    dec = decoding(d)
    fig5(dec["post-impulse window"][0])
    lr = learning()
    fig6(lr)

    summarise(d, res, dv, dec, lr)


if __name__ == "__main__":
    main()
