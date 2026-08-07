"""
==========================================================
High-Impact Manuscript Extended Plots
Experiment 2

Description
-----------
Generates publication-ready figures emphasizing:
- cumulative perturbation trajectories with clear obstacle avoidance context
- perturbation-phase velocity effects with robust summaries
- clean PCA with explicit variance contribution labels
- latent-state clustering analysis with silhouette validation
- meaningful learning diagnostics from checkpoint and training logs
- interpretable success prediction and perturbation tuning

Exports PNG/PDF/SVG for Inkscape workflows.
==========================================================
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None

RESULTS_ROOT = Path("experiments/version_2_0/results")
FIG_OUT = Path("paper/figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

NEURAL_ROOT = RESULTS_ROOT / "neural_analysis"
EVAL_NPZ = Path("experiments/version_1_0/evaluation/evaluations.npz")
TB_ROOT = Path("experiments/version_1_0/logs")

CONDITIONS = ["P0", "L1", "L2", "L3", "R1", "R2", "R3"]
LABELS = {
    "P0": "P0 control",
    "L1": "L1 small left",
    "L2": "L2 medium left",
    "L3": "L3 large left",
    "R1": "R1 small right",
    "R2": "R2 medium right",
    "R3": "R3 large right",
}
PALETTE = {
    "P0": "#264653",
    "L1": "#7A4E80",
    "L2": "#B44D7E",
    "L3": "#D1495B",
    "R1": "#2A9D8F",
    "R2": "#1F7A8C",
    "R3": "#0B3C49",
}

START = (5.0, 1.0)
GOAL = (5.0, 9.0)
OBSTACLES = [(4.25, 5.0), (5.75, 5.0)]
GOAL_RADIUS = 0.35
OBSTACLE_RADIUS = 0.50
PERTURBATION_STEP_MIN = 35
PERTURBATION_STEP_MAX = 46


def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "svg.fonttype": "none",           # editable text in Inkscape
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 600,
            "figure.dpi": 300,
            "axes.linewidth": 0.8,
        }
    )


def export_figure(fig, stem: str):
    outputs = []
    for ext in [".png", ".pdf", ".svg"]:
        path = FIG_OUT / f"{stem}{ext}"
        fmt = ext.lstrip(".")
        fig.savefig(path, format=fmt, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def load_condition_summary(condition: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_ROOT / f"evaluation_{condition}" / "summary.csv")


def load_trajectory(condition: str, episode: int) -> pd.DataFrame:
    return pd.read_csv(RESULTS_ROOT / f"evaluation_{condition}" / f"trajectory_{episode:03d}.csv")


def load_kinematics(condition: str, episode: int) -> pd.DataFrame:
    return pd.read_csv(RESULTS_ROOT / f"evaluation_{condition}" / f"kinematics_{episode:03d}.csv")


def draw_workspace(ax):
    goal = plt.Circle(GOAL, GOAL_RADIUS, color="#3A86FF", alpha=0.20, ec="#3A86FF", lw=1)
    ax.add_patch(goal)

    for obstacle in OBSTACLES:
        circle = plt.Circle(obstacle, OBSTACLE_RADIUS, color="#202020", alpha=0.22)
        ax.add_patch(circle)

    ax.scatter([START[0]], [START[1]], c="#111111", s=35, zorder=10)
    ax.text(START[0], START[1] - 0.33, "Start", ha="center", fontsize=7)

    ax.set_xlim(2.0, 8.0)
    ax.set_ylim(0.7, 9.4)
    ax.set_aspect("equal")


def perturbation_y_band() -> Tuple[float, float]:
    ys = []
    for condition in CONDITIONS:
        summary = load_condition_summary(condition)
        for ep in summary["episode"].astype(int).tolist():
            kdf = load_kinematics(condition, ep)
            i0 = min(PERTURBATION_STEP_MIN, len(kdf) - 1)
            i1 = min(PERTURBATION_STEP_MAX, len(kdf) - 1)
            if i1 <= i0:
                continue
            ys.extend(kdf.loc[i0:i1, "y"].to_list())
    if not ys:
        return (2.0, 3.0)
    return (float(np.percentile(ys, 5)), float(np.percentile(ys, 95)))


def sample_episode_ids(df: pd.DataFrame, n_success: int = 10, n_failure: int = 6) -> Tuple[List[int], List[int]]:
    sdf = df[df["success"] == True]["episode"].astype(int).tolist()  # noqa: E712
    fdf = df[df["success"] == False]["episode"].astype(int).tolist()  # noqa: E712

    rs = np.random.RandomState(42)
    if len(sdf) > n_success:
        sdf = rs.choice(sdf, size=n_success, replace=False).tolist()
    if len(fdf) > n_failure:
        fdf = rs.choice(fdf, size=n_failure, replace=False).tolist()
    return sdf, fdf


def figure5_cumulative_perturbation_trajectories():
    fig, ax = plt.subplots(1, 1, figsize=(10.2, 8.0), constrained_layout=True)
    draw_workspace(ax)

    y0, y1 = perturbation_y_band()
    ax.axhspan(y0, y1, color="#F1FAEE", alpha=0.75, zorder=0)
    ax.text(
        7.82,
        0.5 * (y0 + y1),
        "Perturbation zone\n(start to obstacle corridor)",
        ha="right",
        va="center",
        fontsize=8,
        color="#4A4A4A",
    )

    legend_lines = []
    for condition in CONDITIONS:
        sdf = load_condition_summary(condition)
        success_ids, failure_ids = sample_episode_ids(sdf, n_success=12, n_failure=5)

        for ep in success_ids:
            tdf = load_trajectory(condition, ep)
            ax.plot(
                tdf["x"],
                tdf["y"],
                color=PALETTE[condition],
                lw=1.35,
                alpha=0.40,
            )

        for ep in failure_ids:
            tdf = load_trajectory(condition, ep)
            ax.plot(
                tdf["x"],
                tdf["y"],
                color=PALETTE[condition],
                lw=1.65,
                alpha=0.85,
                linestyle="--",
            )

        succ_rate = 100.0 * float(sdf["success"].mean())
        legend_lines.append(
            plt.Line2D(
                [0],
                [0],
                color=PALETTE[condition],
                lw=2.4,
                label=f"{condition} ({succ_rate:.0f}% success)",
            )
        )

    ax.annotate(
        "Left perturbations",
        xy=(4.2, y1 + 0.08),
        xytext=(3.1, y1 + 0.55),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#B44D7E"),
        color="#B44D7E",
        fontsize=8,
    )
    ax.annotate(
        "Right perturbations",
        xy=(5.8, y1 + 0.08),
        xytext=(6.7, y1 + 0.55),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#1F7A8C"),
        color="#1F7A8C",
        fontsize=8,
        ha="right",
    )

    ax.set_title(
        "Adaptive route geometry under graded lateral perturbations\n"
        "Solid lines: representative successful episodes | Dashed lines: failed episodes"
    )
    ax.set_xlabel("Lateral position (x)")
    ax.set_ylabel("Forward position (y)")

    fail_proxy = plt.Line2D([0], [0], color="#555555", lw=1.8, linestyle="--", label="Failure trajectory style")
    ax.legend(handles=legend_lines + [fail_proxy], frameon=False, loc="upper left", ncol=2)

    return export_figure(fig, "figure5_cumulative_perturbation_trajectories")


def normalize_trace(values: np.ndarray, n_points: int = 160) -> np.ndarray:
    if len(values) < 2:
        return np.full(n_points, np.nan)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, values)


def aggregate_kinematics(metric: str, centered: bool = False, n_points: int = 160):
    data = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        sdf = load_condition_summary(condition)
        for ep in sdf["episode"].astype(int).tolist():
            kdf = load_kinematics(condition, ep)
            arr = kdf[metric].to_numpy(dtype=float)
            if centered and len(arr) > 0:
                arr = arr - arr[0]
            data[condition].append(normalize_trace(arr, n_points=n_points))
    return data


def summarize_with_iqr(traces: List[np.ndarray]):
    arr = np.array(traces)
    med = np.nanmedian(arr, axis=0)
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    return med, q1, q3


def figure6_velocity_perturbation_effects():
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.8), constrained_layout=True, sharex=True)
    ax_vx, ax_speed = axes

    vx_data = aggregate_kinematics("vx", centered=False)
    speed_delta_data = aggregate_kinematics("speed", centered=True)
    phase = np.linspace(0.0, 100.0, 160)

    # Estimate perturbation phase from index window after normalization.
    p0 = 100.0 * PERTURBATION_STEP_MIN / 175.0
    p1 = 100.0 * PERTURBATION_STEP_MAX / 175.0

    for ax in axes:
        ax.axvspan(p0, p1, color="#E9ECEF", alpha=0.7, zorder=0)

    for condition in CONDITIONS:
        med, q1, q3 = summarize_with_iqr(vx_data[condition])
        ax_vx.plot(phase, med, color=PALETTE[condition], lw=2.0, label=condition)
        ax_vx.fill_between(phase, q1, q3, color=PALETTE[condition], alpha=0.11)

    for condition in CONDITIONS:
        med, q1, q3 = summarize_with_iqr(speed_delta_data[condition])
        ax_speed.plot(phase, med, color=PALETTE[condition], lw=2.0, label=condition)
        ax_speed.fill_between(phase, q1, q3, color=PALETTE[condition], alpha=0.11)

    ax_vx.axhline(0.0, color="#666666", lw=1.0, linestyle=":")
    ax_speed.axhline(0.0, color="#666666", lw=1.0, linestyle=":")

    ax_vx.set_title(
        "Lateral velocity signatures show clear perturbation direction encoding"
    )
    ax_vx.set_ylabel("vx (units/s)")

    ax_speed.set_title(
        "Speed modulation relative to movement onset reveals compensation intensity"
    )
    ax_speed.set_ylabel("speed(t) - speed(0)")
    ax_speed.set_xlabel("Normalized movement phase (%)")

    ax_vx.annotate(
        "Left conditions remain below right conditions\nduring perturbation",
        xy=((p0 + p1) / 2, -0.2),
        xytext=(58, -0.55),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#B44D7E"),
        fontsize=8,
        color="#B44D7E",
    )

    handles, labels = ax_vx.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    return export_figure(fig, "figure6_velocity_perturbation_effects")


def load_policy_pca() -> pd.DataFrame:
    path = NEURAL_ROOT / "policy_pca_scores.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pca_variance_from_scores(df: pd.DataFrame):
    variances = np.array([
        float(df["PC1"].var(ddof=1)),
        float(df["PC2"].var(ddof=1)),
        float(df["PC3"].var(ddof=1)),
    ])
    ratio = variances / variances.sum()
    return ratio


def figure7_pca_and_clustering():
    pca_df = load_policy_pca()
    var_ratio = pca_variance_from_scores(pca_df)

    sample_n = min(15000, len(pca_df))
    sample = pca_df.sample(sample_n, random_state=42)

    X = sample[["PC1", "PC2", "PC3"]].to_numpy()
    X2 = sample[["PC1", "PC2"]].to_numpy()

    k_candidates = [2, 3, 4, 5, 6, 7]
    silhouette_scores = []
    models = {}

    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        models[k] = (km, labels)

    best_idx = int(np.argmax(silhouette_scores))
    best_k = k_candidates[best_idx]
    best_model, best_labels = models[best_k]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)

    sns.scatterplot(
        data=sample,
        x="PC1",
        y="PC2",
        hue="condition",
        palette=PALETTE,
        alpha=0.18,
        s=8,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title(
        "Policy latent manifold by perturbation condition\n"
        f"PC1 explains {100 * var_ratio[0]:.1f}% | PC2 explains {100 * var_ratio[1]:.1f}%"
    )
    axes[0].set_xlabel(f"PC1 ({100 * var_ratio[0]:.1f}% variance)")
    axes[0].set_ylabel(f"PC2 ({100 * var_ratio[1]:.1f}% variance)")
    axes[0].legend(frameon=False, fontsize=7, loc="best")

    cmap = plt.get_cmap("tab10")
    for c in range(best_k):
        mask = best_labels == c
        axes[1].scatter(
            X2[mask, 0],
            X2[mask, 1],
            s=8,
            alpha=0.23,
            color=cmap(c),
            label=f"Cluster {c + 1}",
        )

    centers = best_model.cluster_centers_[:, :2]
    axes[1].scatter(centers[:, 0], centers[:, 1], s=140, color="black", marker="X", label="Centroids")
    axes[1].set_title(
        "Unsupervised latent-state clustering reveals structured neural regimes\n"
        f"Best silhouette at k={best_k}: {silhouette_scores[best_idx]:.3f}"
    )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)

    outputs = export_figure(fig, "figure7_pca_clean_clustering")

    silhouette_df = pd.DataFrame({"k": k_candidates, "silhouette": silhouette_scores})
    silhouette_df.to_csv(FIG_OUT / "figure7_clustering_silhouette.csv", index=False)

    return outputs


def parse_training_event_series() -> pd.DataFrame:
    if EventAccumulator is None or not TB_ROOT.exists():
        return pd.DataFrame()

    run_dirs = sorted([p for p in TB_ROOT.iterdir() if p.is_dir() and p.name.startswith("PPO_")])
    rows = []
    for run in run_dirs:
        event_files = sorted(run.glob("events.out.tfevents.*"))
        if not event_files:
            continue

        try:
            acc = EventAccumulator(str(run))
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
        except Exception:
            continue

        for tag in ["rollout/ep_rew_mean", "rollout/ep_len_mean", "train/value_loss", "train/policy_gradient_loss"]:
            if tag not in tags:
                continue
            events = acc.Scalars(tag)
            for e in events:
                rows.append({"run": run.name, "tag": tag, "step": int(e.step), "value": float(e.value)})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def smooth_series(values: np.ndarray, win: int = 15):
    if len(values) < win:
        return values
    kernel = np.ones(win) / win
    return np.convolve(values, kernel, mode="same")


def condition_magnitude_series(summary_df: pd.DataFrame):
    mapping = {"P0": 0.0, "L1": -1.0, "L2": -2.0, "L3": -3.0, "R1": 1.0, "R2": 2.0, "R3": 3.0}
    tmp = summary_df.copy()
    tmp["signed_level"] = tmp["condition"].map(mapping)
    grouped = (
        tmp.groupby("condition", as_index=False)
        .agg(success_rate=("success", "mean"), collision_rate=("collision", "mean"), mean_reward=("reward", "mean"))
    )
    grouped["signed_level"] = grouped["condition"].map(mapping)
    grouped = grouped.sort_values("signed_level")
    return grouped


def figure8_learning_tuning_prediction():
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)
    ax_eval, ax_train, ax_tune, ax_pred = axes.flatten()

    # (A) evaluation checkpoint learning curve
    if EVAL_NPZ.exists():
        npz = np.load(EVAL_NPZ)
        timesteps = npz["timesteps"]
        results = npz["results"]
        mean_reward = results.mean(axis=1)
        q25 = np.percentile(results, 25, axis=1)
        q75 = np.percentile(results, 75, axis=1)

        ax_eval.plot(timesteps, mean_reward, color="#1D3557", lw=2.2)
        ax_eval.fill_between(timesteps, q25, q75, color="#1D3557", alpha=0.18)
        ax_eval.set_title("Learning stability across evaluation checkpoints")
        ax_eval.set_xlabel("Timesteps")
        ax_eval.set_ylabel("Evaluation reward")
    else:
        ax_eval.text(0.5, 0.5, "No evaluations.npz available", ha="center", va="center")
        ax_eval.set_axis_off()

    # (B) training event trajectory
    ev = parse_training_event_series()
    if not ev.empty:
        rew = ev[ev["tag"] == "rollout/ep_rew_mean"].copy().sort_values("step")
        if not rew.empty:
            y = rew["value"].to_numpy(dtype=float)
            ys = smooth_series(y, win=17)
            ax_train.plot(rew["step"], y, color="#A8DADC", lw=1.1, alpha=0.7, label="raw")
            ax_train.plot(rew["step"], ys, color="#E63946", lw=2.2, label="smoothed")
            ax_train.set_title("Policy improvement trajectory during optimization")
            ax_train.set_xlabel("Training step")
            ax_train.set_ylabel("rollout/ep_rew_mean")
            ax_train.legend(frameon=False)
        else:
            ax_train.text(0.5, 0.5, "No rollout reward scalar found", ha="center", va="center")
            ax_train.set_axis_off()
    else:
        ax_train.text(0.5, 0.5, "No tensorboard scalar data available", ha="center", va="center")
        ax_train.set_axis_off()

    # (C) tuning curve
    frames = []
    for condition in CONDITIONS:
        df = load_condition_summary(condition)
        df["condition"] = condition
        frames.append(df)
    summary_df = pd.concat(frames, ignore_index=True)

    tune = condition_magnitude_series(summary_df)
    ax_tune.plot(tune["signed_level"], 100 * tune["success_rate"], marker="o", lw=2.2, color="#2A9D8F", label="success %")
    ax_tune.plot(tune["signed_level"], 100 * tune["collision_rate"], marker="s", lw=2.0, color="#E76F51", label="collision %")
    ax_tune.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax_tune.set_xticklabels(["L3", "L2", "L1", "P0", "R1", "R2", "R3"])
    ax_tune.set_ylim(0, 105)
    ax_tune.set_ylabel("Rate (%)")
    ax_tune.set_title("Perturbation tuning: asymmetry and performance collapse at high rightward force")
    ax_tune.legend(frameon=False)

    # (D) success prediction with interpretable coefficients
    features = [
        "steps",
        "path_length",
        "mean_speed",
        "peak_lateral_velocity",
        "max_heading_deviation",
        "final_lateral_error",
        "route_signal",
    ]
    X = summary_df[features].to_numpy(dtype=float)
    y = summary_df["success"].astype(int).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(x_train_s, y_train)

    probs = clf.predict_proba(x_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    coef = pd.Series(clf.coef_[0], index=features).sort_values()
    colors = ["#457B9D" if v > 0 else "#E76F51" for v in coef.values]
    ax_pred.barh(coef.index, coef.values, color=colors, alpha=0.85)
    ax_pred.axvline(0.0, color="#333333", lw=1)
    ax_pred.set_title(f"Outcome prediction drivers (ROC AUC = {roc_auc:.3f})")
    ax_pred.set_xlabel("Standardized logistic coefficient")

    inset = ax_pred.inset_axes([0.55, 0.08, 0.42, 0.42])
    inset.plot(fpr, tpr, color="#1D3557", lw=2)
    inset.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=1)
    inset.set_title("ROC", fontsize=8)
    inset.tick_params(labelsize=7)

    return export_figure(fig, "figure8_learning_tuning_prediction")


def figure9_nashed_replication_clean():
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.3), constrained_layout=True, sharey=True)
    route_axes = {"left": axes[0], "middle": axes[1], "right": axes[2]}

    def route_at_obstacle(trajectory_df: pd.DataFrame):
        idx = (trajectory_df["y"] - 5.0).abs().idxmin()
        x_val = float(trajectory_df.loc[idx, "x"])
        if x_val < 4.6:
            return "left"
        if x_val > 5.4:
            return "right"
        return "middle"

    for condition in CONDITIONS:
        sdf = load_condition_summary(condition)
        for ep in sdf["episode"].astype(int).tolist():
            tdf = load_trajectory(condition, ep)
            r = route_at_obstacle(tdf)
            ax = route_axes[r]
            success = bool(sdf[sdf["episode"] == ep]["success"].iloc[0])
            ax.plot(
                tdf["x"],
                tdf["y"],
                color=PALETTE[condition],
                alpha=0.22 if success else 0.75,
                lw=1.2 if success else 1.8,
                linestyle="-" if success else "--",
            )

    for name, ax in route_axes.items():
        draw_workspace(ax)
        ax.set_title(f"{name.capitalize()} route family")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")

    handles = [
        plt.Line2D([0], [0], color=PALETTE[c], lw=2, label=c) for c in CONDITIONS
    ]
    style_handles = [
        plt.Line2D([0], [0], color="#555555", lw=1.7, linestyle="-", label="success"),
        plt.Line2D([0], [0], color="#555555", lw=1.7, linestyle="--", label="failure"),
    ]
    fig.legend(handles=handles + style_handles, frameon=False, ncol=9, loc="upper center", bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Nashed-style cumulative route decomposition across perturbation conditions", y=1.10)

    outputs = export_figure(fig, "figure9_nashed_replication_clean")

    # Keep legacy output path updated for compatibility.
    for ext in ["png", "pdf", "svg"]:
        src = FIG_OUT / f"figure9_nashed_replication_clean.{ext}"
        dst = RESULTS_ROOT / f"nashed_style_trajectories.{ext}"
        dst.write_bytes(src.read_bytes())

    return outputs


def save_caption_draft():
    text = """# Extended Figure Captions (Revised)\n\n## Figure 5: Adaptive route geometry under graded perturbations\nAll perturbation conditions are shown together to expose the full robustness envelope in one view. The shaded corridor marks the perturbation zone between movement onset and obstacle approach. Condition color identifies force level and direction; dashed paths denote failures. This panel emphasizes that robust obstacle avoidance is preserved in mild-to-moderate perturbations and breaks down selectively in the strongest rightward regime.\n\n## Figure 6: Velocity signatures of perturbation compensation\nTop panel shows median lateral velocity with interquartile bands across normalized movement phase. Bottom panel shows speed modulation relative to movement onset. The perturbation phase is shaded, making direction-specific divergence and compensation dynamics immediately visible.\n\n## Figure 7: Clean latent PCA and unsupervised clustering\nPolicy latent activity is visualized in PC1-PC2 space with explicit variance contributions shown in axis labels and title. Unsupervised clustering identifies discrete latent regimes and reports the best silhouette-supported cluster count, quantifying latent-state structure rather than relying on visual impression alone.\n\n## Figure 8: Learning dynamics, perturbation tuning, and prediction drivers\nThe panel combines checkpoint evaluation trajectory, optimization-time reward evolution, signed perturbation tuning (success and collision rates), and interpretable prediction coefficients with ROC inset. Together, these analyses reveal not only whether the policy works, but where and why it fails under stronger perturbations.\n\n## Figure 9: Nashed-style route-family replication\nTrajectories are grouped by obstacle-crossing route family (left, middle, right), preserving condition color and outcome style. This layout reproduces route-family analysis in a cleaner, condition-aware format that supports direct comparison of adaptation and failure topologies.\n"""
    path = Path("paper/extended_figure_captions.md")
    path.write_text(text, encoding="utf-8")
    return path


def main():
    print("=" * 72)
    print("Generating revised high-impact extended figures")
    print("=" * 72)

    set_style()

    outputs = []
    outputs += figure5_cumulative_perturbation_trajectories()
    outputs += figure6_velocity_perturbation_effects()
    outputs += figure7_pca_and_clustering()
    outputs += figure8_learning_tuning_prediction()
    outputs += figure9_nashed_replication_clean()

    caption_path = save_caption_draft()

    manifest = pd.DataFrame(
        {
            "file": [str(p) for p in outputs],
            "exists": [p.exists() for p in outputs],
        }
    )
    manifest_path = FIG_OUT / "extended_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nGenerated files:")
    for p in outputs:
        print(p)
    print(caption_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
