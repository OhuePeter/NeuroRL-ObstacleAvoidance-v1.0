"""
==========================================================
COSYNE Figure: Data Stage

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Collects every number the COSYNE composite figure shows from
ONE model and ONE environment, so no panel mixes data from
different runs:

  * live rollouts of a single PPO checkpoint in the
    Experiment 2 environment (7 conditions, alternating
    left/right route cue, fixed perturbation onset);
  * the 256-unit latent activity the policy produced during
    those same rollouts;
  * the closed-loop fixed points and vector field of that same
    checkpoint (src/neural_analysis/attractor_analysis.py).

Rollouts are cached in an .npz (git-ignored under results/);
delete the cache or pass --regenerate to rebuild it.

Usage
-----
python -m scripts.cosyne_figure_data [--regenerate]
==========================================================
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from stable_baselines3 import PPO

from src.environment.environment_v2 import NeuroRLEnvironmentV2
from src.neural_analysis.attractor_analysis import AttractorAnalysis

MODEL_PATH = "experiments/version_1_0/checkpoints/best_model.zip"
CACHE = Path("experiments/version_1_0/results/cosyne_rollouts.npz")

CONDITIONS = ["P0", "L1", "L2", "L3", "R1", "R2", "R3"]
EPISODES_PER_CONDITION = 40      # alternating route cue -> 20 left, 20 right
PERTURB_START = 40               # fixed so conditions differ only in force
SEED = 20260401
N_CLUSTERS = 4
Y_PROBE = 5.0


def _latent(model, obs):

    tensor = torch.as_tensor(
        obs, dtype=torch.float32, device=model.device
    ).unsqueeze(0)

    with torch.no_grad():
        features = model.policy.extract_features(tensor)
        latent_pi, _ = model.policy.mlp_extractor(features)

    return latent_pi.cpu().numpy().squeeze()


def collect_rollouts(model_path=MODEL_PATH):

    model = PPO.load(model_path)

    cols = {k: [] for k in (
        "condition", "episode", "route", "step", "x", "y", "vx", "vy",
        "force", "latent",
    )}
    outcome = []

    for ci, condition in enumerate(CONDITIONS):

        env = NeuroRLEnvironmentV2(
            condition=condition, biological_variability=True
        )
        env.perturbation.start_step = PERTURB_START
        env.variability.rng = np.random.default_rng(SEED + ci)
        np.random.seed(SEED + ci)

        for ep in range(EPISODES_PER_CONDITION):

            obs, _ = env.reset()
            route = env.world.desired_route
            done = False
            info = {}
            t = 0

            while not done:

                latent = _latent(model, obs)
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                t += 1

                agent = env.world.agent
                cols["condition"].append(condition)
                cols["episode"].append(ep)
                cols["route"].append(route)
                cols["step"].append(t)
                cols["x"].append(agent.x)
                cols["y"].append(agent.y)
                cols["vx"].append(agent.vx)
                cols["vy"].append(agent.vy)
                cols["force"].append(float(info.get("perturbation_force", (0, 0))[0])
                                     if isinstance(info.get("perturbation_force"), (tuple, list))
                                     else float(info.get("perturbation_force", 0.0)))
                cols["latent"].append(latent)

                done = terminated or truncated

            outcome.append((
                condition, ep, route,
                bool(info.get("goal_reached")),
                bool(info.get("collision")),
            ))

        print(f"  rollouts done: {condition}")

    out = {k: np.asarray(v) for k, v in cols.items()}
    out["latent"] = out["latent"].astype(np.float32)
    out["o_condition"] = np.array([o[0] for o in outcome])
    out["o_episode"] = np.array([o[1] for o in outcome])
    out["o_route"] = np.array([o[2] for o in outcome])
    out["o_success"] = np.array([o[3] for o in outcome])
    out["o_collision"] = np.array([o[4] for o in outcome])

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)

    return out


def load_rollouts(regenerate=False):

    if CACHE.exists() and not regenerate:
        return dict(np.load(CACHE, allow_pickle=True))

    print("Collecting rollouts (one-off, cached afterwards) ...")
    return collect_rollouts()


def analyse(data):
    """PCA, phase-ordered clusters, fixed points, and curve geometry."""

    result = {}

    latent = data["latent"]
    pca = PCA(n_components=2)
    scores = pca.fit_transform(latent)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
    raw = km.fit_predict(scores)
    order = np.argsort([data["step"][raw == k].mean() for k in range(N_CLUSTERS)])
    relabel = np.zeros(N_CLUSTERS, dtype=int)
    relabel[order] = np.arange(N_CLUSTERS)

    result["pca"] = pca
    result["scores"] = scores
    result["cluster"] = relabel[raw]
    result["var"] = pca.explained_variance_ratio_

    analysis = AttractorAnalysis(MODEL_PATH)
    result["analysis"] = analysis

    p0 = (data["condition"] == "P0") & (data["step"] >= 60) & (data["step"] <= 100)
    vy_plateau = float(np.mean(data["vy"][p0]))
    result["vy_plateau"] = vy_plateau

    y_values = np.linspace(1.5, 8.5, 29)
    result["y_values"] = y_values

    for route in ("left", "right"):

        x_star = analysis.find_lateral_fixed_point(Y_PROBE, vy_plateau, route=route)
        _, eig = analysis.local_jacobian(x_star, Y_PROBE, vy_plateau, route=route)
        tau = max(analysis.time_constant(e, analysis.dt) for e in eig)
        curve = analysis.fixed_point_curve(y_values, vy_plateau, route=route)

        curve_latent = np.array([
            analysis.latent(analysis._observation(x, y, 0.0, vy_plateau, route=route))
            for y, x in curve
        ])

        result[route] = {
            "x_star": x_star, "eig": eig, "tau": tau, "curve": curve,
            "curve_scores": pca.transform(curve_latent),
        }

    return result


def summarise(data, result):

    print("\nOUTCOMES (success / n), by condition x route")
    for c in CONDITIONS:
        parts = []
        for r in ("left", "right"):
            m = (data["o_condition"] == c) & (data["o_route"] == r)
            parts.append(f"{r} {int(data['o_success'][m].sum())}/{int(m.sum())}")
        print(f"  {c}: " + "  ".join(parts))

    print(f"\nPCA variance: PC1 {100*result['var'][0]:.1f}%  PC2 {100*result['var'][1]:.1f}%  "
          f"total {100*result['var'].sum():.1f}%   (n={len(result['scores'])} timesteps)")

    print("\nCLUSTERS (ordered by mean step)")
    for k in range(N_CLUSTERS):
        m = result["cluster"] == k
        fr = " ".join(f"{c}:{(data['condition'][m] == c).mean():.2f}" for c in CONDITIONS)
        rr = " ".join(f"{r}:{(data['route'][m] == r).mean():.2f}" for r in ("left", "right"))
        print(f"  cluster {k+1}: n={m.sum()} step {np.percentile(data['step'][m],25):.0f}-"
              f"{np.percentile(data['step'][m],75):.0f}  {rr}  {fr}")

    print(f"\nvy plateau (P0, steps 60-100): {result['vy_plateau']:.3f}")
    for route in ("left", "right"):
        r = result[route]
        print(f"\n[{route}] x*(y=5) = {r['x_star']:.3f} m  tau = {r['tau']*1000:.0f} ms  eig = {np.round(r['eig'], 4)}")

        m = (data["route"] == route)
        eps = {}
        for c, e, y, x in zip(data["condition"][m], data["episode"][m], data["y"][m], data["x"][m]):
            eps.setdefault((c, e), ([], []))
            eps[(c, e)][0].append(y)
            eps[(c, e)][1].append(x)
        # use only successful P0..R3 episodes that completed the detour
        xs = []
        for (c, e), (yy, xx) in eps.items():
            yy = np.maximum.accumulate(np.array(yy))
            if yy[-1] < 8.4:
                continue
            xs.append(np.interp(r["curve"][:, 0], yy, np.array(xx)))
        xs = np.array(xs)
        diff = r["curve"][:, 1] - xs.mean(0)
        sel = (r["curve"][:, 0] >= 4.5) & (r["curve"][:, 0] <= 5.5)
        print(f"   n episodes for empirical path: {len(xs)}")
        print(f"   x* - empirical mean x(y): y in [4.5,5.5] -> {np.round(diff[sel], 3)}; "
              f"overall median |diff| {np.median(np.abs(diff)):.2f} m, max {np.abs(diff).max():.2f} m")

        sc = result["scores"][::5]
        d = np.sqrt(((r["curve_scores"][:, None, :] - sc[None, :, :]) ** 2).sum(-1)).min(axis=1)
        print(f"   fixed-point curve -> nearest data point in PC space: median {np.median(d):.2f}, max {d.max():.2f}")

    bins = np.linspace(result["scores"][:, 0].min(), result["scores"][:, 0].max(), 25)
    idx = np.digitize(result["scores"][:, 0], bins)
    sp = [result["scores"][idx == i, 1].std() for i in range(1, len(bins)) if (idx == i).sum() > 50]
    print(f"\nmanifold thickness (median PC2 std within PC1 bins): {np.median(sp):.2f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    data = load_rollouts(regenerate=args.regenerate)
    result = analyse(data)
    summarise(data, result)
