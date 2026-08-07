"""
==========================================================
Enhanced Neural Manuscript Figures
Experiment 2

Generates claim-safe, manuscript-ready figures:
- Figure 4: PCA manifold + latent trajectory dynamics
- Figure 5: Hidden-unit tuning to movement speed
- Figure 6: Goal-distance decoding + condition confusion matrix
- Figure 7: Route-choice distribution + perturbation distribution
- Figure 8: Hierarchical condition organization in latent space
==========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

RESULTS_ROOT = Path("experiments/version_2_0/results")
NEURAL_ROOT = RESULTS_ROOT / "neural_analysis"
FIG_OUT = Path("paper/figures")
FIG_OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["P0", "L1", "L2", "L3", "R1", "R2", "R3"]
PALETTE = {
    "P0": "#264653",
    "L1": "#7A4E80",
    "L2": "#B44D7E",
    "L3": "#D1495B",
    "R1": "#2A9D8F",
    "R2": "#1F7A8C",
    "R3": "#0B3C49",
}


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
            "axes.linewidth": 0.8,
            "savefig.dpi": 600,
            "figure.dpi": 300,
        }
    )


def export_figure(fig, stem):
    outputs = []
    for ext in [".png", ".pdf", ".svg"]:
        out = FIG_OUT / f"{stem}{ext}"
        fmt = ext.lstrip(".")
        fig.savefig(out, format=fmt, bbox_inches="tight")
        outputs.append(out)
    plt.close(fig)
    return outputs


def load_policy_pca_scores():
    return pd.read_csv(NEURAL_ROOT / "policy_pca_scores.csv")


def pca_variance(scores_df):
    variances = np.array([
        float(scores_df["PC1"].var(ddof=1)),
        float(scores_df["PC2"].var(ddof=1)),
        float(scores_df["PC3"].var(ddof=1)),
    ])
    ratios = variances / variances.sum()
    return ratios


def figure4_pca_and_trajectory():
    df = load_policy_pca_scores()
    ratios = pca_variance(df)

    fig = plt.figure(figsize=(13.2, 5.4), constrained_layout=True)

    ax1 = fig.add_subplot(1, 2, 1)
    sampled = df.sample(min(15000, len(df)), random_state=42)
    sns.scatterplot(
        data=sampled,
        x="PC1",
        y="PC2",
        hue="condition",
        palette=PALETTE,
        alpha=0.20,
        s=8,
        linewidth=0,
        ax=ax1,
    )
    ax1.set_title(
        "Neural population PCA under graded perturbations\n"
        f"PC1={100*ratios[0]:.1f}%  PC2={100*ratios[1]:.1f}%  (PC1+PC2={100*(ratios[0]+ratios[1]):.1f}%)"
    )
    ax1.set_xlabel(f"PC1 ({100*ratios[0]:.1f}% variance)")
    ax1.set_ylabel(f"PC2 ({100*ratios[1]:.1f}% variance)")
    ax1.legend(frameon=False, ncol=2, fontsize=7)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    episode_index = pd.read_csv(NEURAL_ROOT / "episode_index.csv")

    picked = []
    for c in CONDITIONS:
        sub = episode_index[(episode_index["condition"] == c) & (episode_index["success"] == True)].head(1)  # noqa: E712
        if not sub.empty:
            picked.append(sub)
    picked_df = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()

    for _, row in picked_df.iterrows():
        start = int(row["start_index"])
        end = int(row["end_index"])
        seg = df.iloc[start:end]
        if seg.empty:
            continue
        ax2.plot(
            seg["PC1"],
            seg["PC2"],
            seg["PC3"],
            color=PALETTE[row["condition"]],
            lw=2.0,
            alpha=0.9,
            label=row["condition"],
        )
    ax2.set_title("Temporal neural population trajectories in latent space")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")

    handles, labels = ax2.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax2.legend(uniq.values(), uniq.keys(), frameon=False, loc="upper left")

    return export_figure(fig, "figure4_neural_population_dynamics")


def load_latent_speed_table():
    rows = []
    unit_dim = None

    for condition in CONDITIONS:
        summary_path = RESULTS_ROOT / f"evaluation_{condition}" / "summary.csv"
        summary = pd.read_csv(summary_path)

        for episode in summary["episode"].astype(int).tolist():
            policy_path = RESULTS_ROOT / f"evaluation_{condition}" / "neural" / f"policy_{episode:03d}.npy"
            kin_path = RESULTS_ROOT / f"evaluation_{condition}" / f"kinematics_{episode:03d}.csv"
            if not policy_path.exists() or not kin_path.exists():
                continue

            latent = np.load(policy_path)
            kin = pd.read_csv(kin_path)
            n = min(len(latent), len(kin))
            if n == 0:
                continue

            latent = latent[:n]
            speed = kin["speed"].to_numpy(dtype=float)[:n]

            if unit_dim is None:
                unit_dim = latent.shape[1]

            for t in range(n):
                row = {
                    "condition": condition,
                    "episode": episode,
                    "timestep": t,
                    "speed": float(speed[t]),
                }
                for u in range(latent.shape[1]):
                    row[f"u{u}"] = float(latent[t, u])
                rows.append(row)

    df = pd.DataFrame(rows)
    return df, unit_dim


def figure5_hidden_unit_tuning():
    df, unit_dim = load_latent_speed_table()
    if df.empty or unit_dim is None:
        return []

    corr = {}
    for u in range(unit_dim):
        col = f"u{u}"
        corr[col] = float(df[[col, "speed"]].corr().iloc[0, 1])

    top_units = sorted(corr.keys(), key=lambda k: abs(corr[k]), reverse=True)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6), constrained_layout=True)

    sample = df.sample(min(12000, len(df)), random_state=42)

    for ax, unit in zip(axes.flatten(), top_units):
        x = sample["speed"].to_numpy(dtype=float)
        y = sample[unit].to_numpy(dtype=float)

        ax.scatter(x, y, s=8, alpha=0.22, color="#4C78A8", linewidths=0)

        m, b = np.polyfit(x, y, deg=1)
        xs = np.linspace(np.min(x), np.max(x), 100)
        ys = m * xs + b
        ax.plot(xs, ys, color="#D1495B", lw=2)

        ax.set_title(f"{unit}  (r={corr[unit]:.2f})")
        ax.set_xlabel("Speed")
        ax.set_ylabel("Activation")

    fig.suptitle(
        "Hidden-unit tuning during goal-directed obstacle avoidance\n"
        "Representative PPO policy units show diverse speed modulation patterns",
        y=1.02,
    )

    return export_figure(fig, "figure5_hidden_unit_tuning")


def build_goal_distance_decoder_table():
    rows = []
    for condition in CONDITIONS:
        summary = pd.read_csv(RESULTS_ROOT / f"evaluation_{condition}" / "summary.csv")
        for episode in summary["episode"].astype(int).tolist():
            policy_path = RESULTS_ROOT / f"evaluation_{condition}" / "neural" / f"policy_{episode:03d}.npy"
            kin_path = RESULTS_ROOT / f"evaluation_{condition}" / f"kinematics_{episode:03d}.csv"
            if not policy_path.exists() or not kin_path.exists():
                continue

            latent = np.load(policy_path)
            kin = pd.read_csv(kin_path)
            n = min(len(latent), len(kin))
            if n == 0:
                continue

            latent = latent[:n]
            goal_distance = kin["goal_distance"].to_numpy(dtype=float)[:n]

            for t in range(n):
                row = {
                    "condition": condition,
                    "episode": int(episode),
                    "timestep": int(t),
                    "goal_distance": float(goal_distance[t]),
                }
                for u in range(latent.shape[1]):
                    row[f"u{u}"] = float(latent[t, u])
                rows.append(row)

    return pd.DataFrame(rows)


def figure6_goal_decoding_and_confusion():
    gdf = build_goal_distance_decoder_table()
    if gdf.empty:
        return []

    unit_cols = [c for c in gdf.columns if c.startswith("u")]
    X = gdf[unit_cols].to_numpy(dtype=float)
    y = gdf["goal_distance"].to_numpy(dtype=float)
    groups = gdf["episode"].to_numpy(dtype=int) + 1000 * gdf["condition"].astype("category").cat.codes.to_numpy(dtype=int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    cm = pd.read_csv(NEURAL_ROOT / "decoding_confusion_policy.csv", index_col=0)
    cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)

    ax1 = axes[0]
    ax1.hexbin(y_test, y_pred, gridsize=45, cmap="viridis", mincnt=1)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, "--", color="white", lw=1.4)
    ax1.set_title(f"Goal-variable decoding from hidden state\nR2={r2:.3f}, MAE={mae:.3f}")
    ax1.set_xlabel("True goal distance")
    ax1.set_ylabel("Predicted goal distance")

    ax2 = axes[1]
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar=False,
        square=True,
        linewidths=0.4,
        linecolor="white",
        ax=ax2,
    )
    ax2.set_title("Decoder confusion matrix (policy latent, row-normalized)")
    ax2.set_xlabel("Predicted condition")
    ax2.set_ylabel("True condition")

    return export_figure(fig, "figure6_goal_decoding_confusion")


def route_at_obstacle(trajectory_df):
    idx = (trajectory_df["y"] - 5.0).abs().idxmin()
    x = float(trajectory_df.loc[idx, "x"])
    if x < 4.6:
        return "Left"
    if x > 5.4:
        return "Right"
    return "Centre"


def figure7_route_and_perturbation_distribution():
    route_counts = {"Left": 0, "Centre": 0, "Right": 0}
    cond_counts = {c: 0 for c in CONDITIONS}

    for condition in CONDITIONS:
        sdf = pd.read_csv(RESULTS_ROOT / f"evaluation_{condition}" / "summary.csv")
        cond_counts[condition] = len(sdf)
        for episode in sdf["episode"].astype(int).tolist():
            tpath = RESULTS_ROOT / f"evaluation_{condition}" / f"trajectory_{episode:03d}.csv"
            if not tpath.exists():
                continue
            tdf = pd.read_csv(tpath)
            route_counts[route_at_obstacle(tdf)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)

    ax1 = axes[0]
    rc = pd.DataFrame({"route": list(route_counts.keys()), "count": list(route_counts.values())})
    sns.barplot(
        data=rc,
        x="route",
        y="count",
        hue="route",
        palette={"Left": "#B44D7E", "Centre": "#6C757D", "Right": "#1F7A8C"},
        dodge=False,
        legend=False,
        ax=ax1,
    )
    ax1.set_title("Route-choice distribution across evaluation episodes")
    ax1.set_xlabel("Route family")
    ax1.set_ylabel("Episode count")

    ax2 = axes[1]
    cc = pd.DataFrame({"condition": CONDITIONS, "count": [cond_counts[c] for c in CONDITIONS]})
    sns.barplot(
        data=cc,
        x="condition",
        y="count",
        hue="condition",
        palette=PALETTE,
        dodge=False,
        legend=False,
        ax=ax2,
    )
    ax2.set_title("Perturbation distribution across evaluation conditions")
    ax2.set_xlabel("Condition")
    ax2.set_ylabel("Episode count")

    return export_figure(fig, "figure7_route_perturbation_distribution")


def figure8_hierarchical_latent_structure():
    df = load_policy_pca_scores()

    centroids = (
        df.groupby("condition", as_index=False)[["PC1", "PC2", "PC3"]]
        .mean()
        .set_index("condition")
        .loc[CONDITIONS]
    )

    dist = pdist(centroids.values, metric="euclidean")
    Z = linkage(dist, method="ward")
    dist_mat = pd.DataFrame(squareform(dist), index=CONDITIONS, columns=CONDITIONS)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)

    ax1 = axes[0]
    dendrogram(Z, labels=CONDITIONS, ax=ax1, color_threshold=None)
    ax1.set_title("Hierarchical organization of condition centroids in latent space")
    ax1.set_xlabel("Condition")
    ax1.set_ylabel("Ward linkage distance")

    ax2 = axes[1]
    sns.heatmap(
        dist_mat,
        cmap="mako",
        square=True,
        cbar=True,
        linewidths=0.4,
        linecolor="white",
        ax=ax2,
    )
    ax2.set_title("Pairwise latent centroid distances")
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    return export_figure(fig, "figure8_hierarchical_latent_structure")


def main():
    print("=" * 72)
    print("Generating enhanced neural/manuscript figures")
    print("=" * 72)

    set_style()

    outputs = []
    outputs += figure4_pca_and_trajectory()
    outputs += figure5_hidden_unit_tuning()
    outputs += figure6_goal_decoding_and_confusion()
    outputs += figure7_route_and_perturbation_distribution()
    outputs += figure8_hierarchical_latent_structure()

    manifest = pd.DataFrame({"file": [str(p) for p in outputs], "exists": [p.exists() for p in outputs]})
    manifest_path = FIG_OUT / "enhanced_neural_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nGenerated files:")
    for p in outputs:
        print(p)
    print(manifest_path)


if __name__ == "__main__":
    main()
