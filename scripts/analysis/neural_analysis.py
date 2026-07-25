"""
==========================================================
Neural Analysis
Experiment 2

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Loads saved latent neural recordings from Experiment 2
and generates publication-ready analyses:

- 3D PCA of latent states
- Neural trajectory plots in PCA space
- Representational Similarity Analysis (RSA)
- Hidden-unit correlation analysis
- Success vs failure comparisons
- Perturbation-condition neural decoding

Expected input folders
----------------------
experiments/version_2_0/results/evaluation_P0/neural/
experiments/version_2_0/results/evaluation_L1/neural/
...

==========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RESULTS_ROOT = Path("experiments/version_2_0/results")
OUTPUT_DIR = RESULTS_ROOT / "neural_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "P0",
    "L1",
    "L2",
    "L3",
    "R1",
    "R2",
    "R3",
]

MAX_TRAJECTORIES_PER_CONDITION = 3
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------


def _episode_from_stem(filename, prefix):

    stem = Path(filename).stem

    if not stem.startswith(prefix + "_"):
        raise ValueError(f"Unexpected filename format: {filename}")

    return int(stem.split("_")[1])


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------


def load_neural_dataset():

    print("=" * 70)
    print("Loading Neural Recordings")
    print("=" * 70)

    policy_chunks = []
    value_chunks = []
    metadata_rows = []
    episode_rows = []

    total_timesteps = 0
    total_episodes = 0

    for condition in CONDITIONS:

        eval_dir = RESULTS_ROOT / f"evaluation_{condition}"
        neural_dir = eval_dir / "neural"
        summary_file = eval_dir / "summary.csv"

        if not neural_dir.exists():
            print(f"Missing neural directory: {neural_dir}")
            continue

        if not summary_file.exists():
            print(f"Missing summary.csv: {summary_file}")
            continue

        summary_df = pd.read_csv(summary_file)

        if "episode" not in summary_df.columns:
            print(f"Missing 'episode' column in: {summary_file}")
            continue

        summary_df = summary_df.set_index("episode")

        policy_files = sorted(neural_dir.glob("policy_*.npy"))

        for policy_file in policy_files:

            episode = _episode_from_stem(policy_file.name, "policy")

            value_file = neural_dir / f"value_{episode:03d}.npy"

            if not value_file.exists():
                print(f"Missing value file: {value_file}")
                continue

            if episode not in summary_df.index:
                print(f"Episode {episode} missing in summary for {condition}")
                continue

            policy = np.load(policy_file)
            value = np.load(value_file)

            if policy.ndim != 2 or value.ndim != 2:
                print(f"Skipping malformed arrays for {condition} episode {episode:03d}")
                continue

            timesteps = min(policy.shape[0], value.shape[0])

            if timesteps == 0:
                print(f"Skipping empty arrays for {condition} episode {episode:03d}")
                continue

            policy = policy[:timesteps]
            value = value[:timesteps]

            summary = summary_df.loc[episode]
            success = bool(summary.get("success", False))
            collision = bool(summary.get("collision", False))

            start_index = total_timesteps
            end_index = total_timesteps + timesteps

            policy_chunks.append(policy)
            value_chunks.append(value)

            metadata_rows.extend(
                {
                    "condition": condition,
                    "episode": int(episode),
                    "timestep": int(t),
                    "global_index": int(start_index + t),
                    "success": success,
                    "failure": not success,
                    "collision": collision,
                }
                for t in range(timesteps)
            )

            episode_rows.append(
                {
                    "condition": condition,
                    "episode": int(episode),
                    "success": success,
                    "collision": collision,
                    "start_index": int(start_index),
                    "end_index": int(end_index),
                    "timesteps": int(timesteps),
                }
            )

            total_timesteps += timesteps
            total_episodes += 1

            print(f"{condition:<3}  Episode {episode:03d}  {timesteps:>4} timesteps")

    if not policy_chunks:
        raise RuntimeError(
            "No neural recordings were loaded. Check evaluation folders and filenames."
        )

    policy_all = np.vstack(policy_chunks)
    value_all = np.vstack(value_chunks)

    metadata_df = pd.DataFrame(metadata_rows)
    episode_df = pd.DataFrame(episode_rows)

    print()
    print("=" * 70)
    print("Finished Loading")
    print("=" * 70)

    print()
    print("=" * 70)
    print("Neural Dataset Summary")
    print("=" * 70)
    print(f"Conditions : {metadata_df['condition'].nunique()}")
    print(f"Episodes   : {total_episodes}")
    print(f"Timesteps  : {total_timesteps}")
    print()
    print(f"Policy Shape : {policy_all.shape}")
    print(f"Value Shape  : {value_all.shape}")

    metadata_df.to_csv(OUTPUT_DIR / "neural_metadata.csv", index=False)
    episode_df.to_csv(OUTPUT_DIR / "episode_index.csv", index=False)

    return policy_all, value_all, metadata_df, episode_df


# ---------------------------------------------------------
# PCA analysis and plots
# ---------------------------------------------------------


def run_pca(latent, metadata_df, prefix):

    scaler = StandardScaler()
    latent_z = scaler.fit_transform(latent)

    pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
    scores = pca3.fit_transform(latent_z)

    scores_df = metadata_df.copy()
    scores_df["PC1"] = scores[:, 0]
    scores_df["PC2"] = scores[:, 1]
    scores_df["PC3"] = scores[:, 2]

    scores_df.to_csv(OUTPUT_DIR / f"{prefix}_pca_scores.csv", index=False)

    variance = pca3.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    plt.bar([1, 2, 3], variance * 100, color="#2A9D8F", alpha=0.85)
    plt.plot([1, 2, 3], np.cumsum(variance) * 100, marker="o", color="#264653")
    plt.xticks([1, 2, 3], ["PC1", "PC2", "PC3"])
    plt.ylabel("Explained Variance (%)")
    plt.title(f"{prefix.title()} Latent Explained Variance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"explained_variance_{prefix}.png", dpi=300)
    plt.close()

    palette = dict(zip(CONDITIONS, sns.color_palette("tab10", len(CONDITIONS))))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for condition in CONDITIONS:
        subset = scores_df[scores_df["condition"] == condition]
        if subset.empty:
            continue
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            subset["PC3"],
            s=4,
            alpha=0.25,
            color=palette[condition],
            label=condition,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"3D PCA - {prefix.title()} Latent States")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_pca_3d.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=scores_df,
        x="PC1",
        y="PC2",
        hue="condition",
        palette=palette,
        alpha=0.25,
        s=8,
        linewidth=0,
    )
    plt.title(f"2D PCA - {prefix.title()} Latent States")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_pca_2d.png", dpi=300)
    plt.close()

    return latent_z, scores_df, pca3


# ---------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------


def plot_neural_trajectories(scores_df, episode_df, prefix):

    palette = dict(zip(CONDITIONS, sns.color_palette("tab10", len(CONDITIONS))))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    plotted = 0

    for condition in CONDITIONS:

        rows = episode_df[episode_df["condition"] == condition].head(
            MAX_TRAJECTORIES_PER_CONDITION
        )

        for _, row in rows.iterrows():
            start = int(row["start_index"])
            end = int(row["end_index"])

            traj = scores_df.iloc[start:end]

            if traj.empty:
                continue

            ax.plot(
                traj["PC1"].values,
                traj["PC2"].values,
                traj["PC3"].values,
                color=palette[condition],
                alpha=0.8,
                linewidth=1.3,
            )

            plotted += 1

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(
        f"Neural Trajectories in PCA Space ({prefix.title()})\n"
        f"Up to {MAX_TRAJECTORIES_PER_CONDITION} episodes per condition"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_pca_trajectories_3d.png", dpi=300)
    plt.close()

    print(f"Plotted {plotted} trajectories for {prefix} latent space")


# ---------------------------------------------------------
# Representational Similarity Analysis (RSA)
# ---------------------------------------------------------


def rsa_heatmap(latent_z, metadata_df, prefix):

    condition_means = []

    for condition in CONDITIONS:
        idx = metadata_df[metadata_df["condition"] == condition].index
        if len(idx) == 0:
            continue
        condition_means.append(latent_z[idx].mean(axis=0))

    if len(condition_means) < 2:
        print(f"Skipping RSA for {prefix}: not enough condition data")
        return

    condition_means = np.vstack(condition_means)
    similarity = np.corrcoef(condition_means)

    used_conditions = [
        c for c in CONDITIONS if (metadata_df["condition"] == c).any()
    ]

    similarity_df = pd.DataFrame(
        similarity,
        index=used_conditions,
        columns=used_conditions,
    )

    similarity_df.to_csv(OUTPUT_DIR / f"rsa_{prefix}.csv")

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        similarity_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Pearson similarity"},
    )
    plt.title(f"RSA Heatmap ({prefix.title()} Latent)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"rsa_{prefix}.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Hidden-unit correlation analysis
# ---------------------------------------------------------


def hidden_unit_correlations(latent_z, prefix):

    corr = np.corrcoef(latent_z, rowvar=False)

    corr_df = pd.DataFrame(corr)
    corr_df.to_csv(OUTPUT_DIR / f"hidden_unit_correlation_{prefix}.csv", index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Unit correlation"},
    )
    plt.title(f"Hidden-Unit Correlation ({prefix.title()} Latent)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"hidden_unit_correlation_{prefix}.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Success vs failure comparison
# ---------------------------------------------------------


def success_failure_analysis(scores_df, prefix):

    if scores_df["success"].nunique() < 2:
        note = (
            "Only one class (all success or all failure) present. "
            "Skipping success/failure statistical comparison."
        )
        print(f"{prefix}: {note}")
        with open(OUTPUT_DIR / f"success_failure_{prefix}_note.txt", "w", encoding="utf-8") as f:
            f.write(note + "\n")
        return

    success_df = scores_df[scores_df["success"]]
    failure_df = scores_df[~scores_df["success"]]

    rows = []

    for pc in ["PC1", "PC2", "PC3"]:

        x = success_df[pc].values
        y = failure_df[pc].values

        stat, p = ttest_ind(x, y, equal_var=False)

        pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2.0)
        cohen_d = (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else np.nan

        rows.append(
            {
                "Component": pc,
                "Success Mean": float(np.mean(x)),
                "Failure Mean": float(np.mean(y)),
                "t-stat": float(stat),
                "p-value": float(p),
                "Cohen d": float(cohen_d),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / f"success_failure_comparison_{prefix}.csv", index=False)

    long_df = scores_df[["success", "PC1", "PC2", "PC3"]].melt(
        id_vars="success",
        var_name="Component",
        value_name="Score",
    )

    long_df["Outcome"] = np.where(long_df["success"], "Success", "Failure")

    plt.figure(figsize=(10, 7))
    sns.boxplot(
        data=long_df,
        x="Component",
        y="Score",
        hue="Outcome",
        palette={"Success": "#2A9D8F", "Failure": "#E76F51"},
    )
    plt.title(f"Success vs Failure PCA Scores ({prefix.title()} Latent)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"success_failure_{prefix}.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Neural decoding
# ---------------------------------------------------------


def decode_condition(latent, metadata_df, prefix):

    episode_group = metadata_df["condition"] + "_" + metadata_df["episode"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(latent, metadata_df["condition"], groups=episode_group))

    x_train = latent[train_idx]
    x_test = latent[test_idx]

    y_train = metadata_df.iloc[train_idx]["condition"].values
    y_test = metadata_df.iloc[test_idx]["condition"].values

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=20, random_state=RANDOM_STATE)),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred, labels=CONDITIONS)
    cm_df = pd.DataFrame(cm, index=CONDITIONS, columns=CONDITIONS)
    cm_df.to_csv(OUTPUT_DIR / f"decoding_confusion_{prefix}.csv")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(OUTPUT_DIR / f"decoding_report_{prefix}.csv")

    metrics_df = pd.DataFrame(
        [
            {
                "latent": prefix,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        ]
    )
    metrics_df.to_csv(OUTPUT_DIR / f"decoding_metrics_{prefix}.csv", index=False)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Condition Decoding Confusion Matrix ({prefix.title()} Latent)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"decoding_confusion_{prefix}.png", dpi=300)
    plt.close()

    return {
        "latent": prefix,
        "accuracy": acc,
        "balanced_accuracy": bacc,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():

    policy, value, metadata_df, episode_df = load_neural_dataset()

    print()
    print("=" * 70)
    print("Running PCA")
    print("=" * 70)

    policy_z, policy_scores, _ = run_pca(policy, metadata_df, prefix="policy")
    value_z, value_scores, _ = run_pca(value, metadata_df, prefix="value")

    print()
    print("=" * 70)
    print("Plotting Neural Trajectories")
    print("=" * 70)

    plot_neural_trajectories(policy_scores, episode_df, prefix="policy")
    plot_neural_trajectories(value_scores, episode_df, prefix="value")

    print()
    print("=" * 70)
    print("Computing RSA")
    print("=" * 70)

    rsa_heatmap(policy_z, metadata_df, prefix="policy")
    rsa_heatmap(value_z, metadata_df, prefix="value")

    print()
    print("=" * 70)
    print("Computing Hidden-Unit Correlations")
    print("=" * 70)

    hidden_unit_correlations(policy_z, prefix="policy")
    hidden_unit_correlations(value_z, prefix="value")

    print()
    print("=" * 70)
    print("Success vs Failure Analysis")
    print("=" * 70)

    success_failure_analysis(policy_scores, prefix="policy")
    success_failure_analysis(value_scores, prefix="value")

    print()
    print("=" * 70)
    print("Neural Decoding")
    print("=" * 70)

    decoding_policy = decode_condition(policy, metadata_df, prefix="policy")
    decoding_value = decode_condition(value, metadata_df, prefix="value")

    decoding_summary = pd.DataFrame([decoding_policy, decoding_value])
    decoding_summary.to_csv(OUTPUT_DIR / "decoding_summary.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "conditions": metadata_df["condition"].nunique(),
                "episodes": len(episode_df),
                "timesteps": len(metadata_df),
                "policy_units": policy.shape[1],
                "value_units": value.shape[1],
            }
        ]
    )

    summary.to_csv(OUTPUT_DIR / "neural_summary.csv", index=False)

    print()
    print("=" * 70)
    print("NEURAL ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
