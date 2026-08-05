# Overleaf Update Audit (2026-08-05)

## 1) Objective Check: Did we achieve what you asked?

### Achieved
- Clear perturbation performance gradient now exists and is reproducible in current outputs:
  - P0 100%, L1 100%, L2 100%, L3 90%, R1 100%, R2 85%, R3 50%.
- Publication-ready multi-format figure exports (PNG/PDF/SVG) are generated.
- Perturbation effects are now visualized with condition overlays, success/failure styling, and dedicated perturbation-phase velocity plots.
- PCA panel now reports PC1/PC2 variance contributions explicitly.
- Unsupervised clustering was added with silhouette-based model selection.
- Nashed-style route-family decomposition is now generated in a cleaner form.

### Partially achieved
- "High-impact journal ready" is improved substantially, but still needs a final editorial pass on figure typography consistency and panel density.
- Learning analysis is better (checkpoint + optimization trace), but still single-seed unless you retrain across multiple seeds.

### Not fully achieved (yet)
- No multi-seed confidence analysis for robustness/neural metrics.
- No external biological benchmark dataset for direct biological validation.
- No formal uncertainty intervals on all reported summary percentages in manuscript tables.

---

## 2) Methodology Consistency Audit: manuscript text vs code

This section flags concrete mismatches between your draft and the committed code.

### A. Observation/state definition mismatch
- Draft claims 8-D state: relative goal/obstacle + velocity.
- Current code uses 13-D state including absolute position, goal distance, heading, and route cue.
- Source: [src/environment/observation.py](src/environment/observation.py)

Use this corrected equation in Overleaf:

$$
\mathbf{s}_t = [x_t, y_t, v_{x,t}, v_{y,t}, g_x-x_t, g_y-y_t, o_{1x}-x_t, o_{1y}-y_t, o_{2x}-x_t, o_{2y}-y_t, d_{g,t}, \psi_t, r_t]
$$

where $d_{g,t}=\lVert \mathbf{g}-\mathbf{p}_t \rVert$, $\psi_t$ is heading, and $r_t\in\{-1,0,1\}$ is route cue.

### B. Perturbation protocol mismatch
- Draft claims: force at steps 50-55 with magnitudes $\pm0.15, \pm0.30, \pm0.45$.
- Current code: onset jittered between step 35 and 46, duration 14 steps, magnitudes:
  - L1 -0.40, L2 -0.60, L3 -0.85, R1 0.40, R2 0.65, R3 1.80.
- Sources:
  - [src/perturbations/perturbation_v2.py](src/perturbations/perturbation_v2.py)
  - [src/evaluation/biological_variability.py](src/evaluation/biological_variability.py)
  - [configs/environment.yaml](configs/environment.yaml)

Recommended corrected statement:
- "Perturbations were applied as transient lateral forces with trial-to-trial onset jitter ($t_0\in[35,46)$), 14-step duration, and condition-specific magnitudes (L1/L2/L3 = -0.40/-0.60/-0.85; R1/R2/R3 = 0.40/0.65/1.80)."

### C. PPO training setup mismatch
- Draft states 2e6 timesteps and entropy coefficient 0.005.
- Current code uses 3e6 timesteps and entropy coefficient 0.01.
- Sources:
  - [src/training/trainer.py](src/training/trainer.py)
  - [configs/training.yaml](configs/training.yaml)

Also fix typo in draft:
- learning rate should be $3\times10^{-4}$, not $3\times10^{4}$.

### D. VecNormalize claim mismatch
- Draft says observations were normalized using VecNormalize.
- Current training code does not wrap env with VecNormalize.
- Source: [src/training/trainer.py](src/training/trainer.py)

### E. Environment structure mismatch risk
- Draft says Experiment 1 used single obstacle and Experiment 2 used two obstacles.
- Current environment config in main branch defines two obstacles in the active config.
- Source: [configs/environment.yaml](configs/environment.yaml)

Action:
- If you keep the single-obstacle claim for Experiment 1, cite archived branch/tag or old config snapshot as evidence.
- Otherwise, revise text to avoid unsupported historical claim in the current reproducible code path.

### F. Episode count in robustness section
- Current Experiment 2 evaluation is 20 episodes per condition.
- Source: [experiments/version_2_0/evaluation/config.py](experiments/version_2_0/evaluation/config.py)

---

## 3) Are methods consistent with analyses now?

### Mostly yes
- Behavioural summaries, trajectories, velocity effects, PCA, clustering, tuning, and prediction are all drawn from evaluation outputs and neural recordings.
- Perturbation timing and force magnitudes now align between code and updated figure logic.

### Remaining caveats to report explicitly
- Clustering is unsupervised structure discovery, not causal evidence.
- Prediction is in-distribution and single-dataset; avoid overclaiming generalization.
- Learning curves represent one training run unless you add multi-seed replications.

---

## 4) Results currently not provided (or not sufficiently strong)

These are the gaps you should either fill or explicitly acknowledge:

1. Multi-seed robustness confidence intervals.
2. Statistical tests for clustering validity beyond silhouette (for example, stability across resampling).
3. Cross-run reproducibility of PCA manifold geometry.
4. External benchmark against baseline controllers (for example, SAC/TD3 or no-route-cue ablation).
5. Trial-level logged perturbation onset per episode in metadata (currently inferred from config/noise process).

---

## 5) Manual figure replacement map for Overleaf

Use these updated assets:

- Main cumulative perturbation trajectories:
  - [paper/figures/figure5_cumulative_perturbation_trajectories.svg](paper/figures/figure5_cumulative_perturbation_trajectories.svg)
- Perturbation-phase velocity effects:
  - [paper/figures/figure6_velocity_perturbation_effects.svg](paper/figures/figure6_velocity_perturbation_effects.svg)
- Clean PCA + clustering:
  - [paper/figures/figure7_pca_clean_clustering.svg](paper/figures/figure7_pca_clean_clustering.svg)
  - [paper/figures/figure7_clustering_silhouette.csv](paper/figures/figure7_clustering_silhouette.csv)
- Learning + tuning + prediction:
  - [paper/figures/figure8_learning_tuning_prediction.svg](paper/figures/figure8_learning_tuning_prediction.svg)
- Nashed-style replication:
  - [paper/figures/figure9_nashed_replication_clean.svg](paper/figures/figure9_nashed_replication_clean.svg)
  - [experiments/version_2_0/results/nashed_style_trajectories.svg](experiments/version_2_0/results/nashed_style_trajectories.svg)

---

## 6) Overleaf-ready replacement text (Methodology block)

You can paste/adapt the following directly:

"In this study, we trained a PPO controller in a continuous obstacle-avoidance task and then probed policy robustness using graded lateral perturbations. During evaluation, we recorded trial-level behavioural outcomes and hidden-layer latent activity, treating these latents as population data for dimensionality reduction, trajectory analysis, tuning-style summaries, clustering, and decoding. This design links behavioural failure modes to changes in low-dimensional internal dynamics.

The environment is implemented in Gymnasium with continuous actions and physics-constrained updates. The policy is trained with Stable-Baselines3 PPO using a two-layer MLP (256, 256) and evaluated under seven perturbation conditions: P0, L1-L3, and R1-R3. Perturbations are applied as transient horizontal forces with onset jitter in the early movement phase ($t_0\in[35,46)$), duration 14 control steps, and graded force magnitudes (L1/L2/L3: -0.40/-0.60/-0.85; R1/R2/R3: 0.40/0.65/1.80). We additionally include trial-to-trial start, observation, and action noise during evaluation.

The observation vector is 13-dimensional and combines absolute kinematics, relative target/obstacle geometry, goal distance, heading, and a route cue. Behavioural analyses include success/failure, collision rate, reward, duration, path length, and velocity-based adaptation signatures. Neural analyses include PCA with explained-variance reporting, condition-structured latent trajectories, unsupervised clustering with silhouette validation, and outcome prediction from behavioural features."

---

## 7) Precision edits to your current draft

Apply these exact corrections:

1. Replace all references to perturbation timing "50-55" with onset-jitter + 14-step window.
2. Replace perturbation magnitudes $\pm0.15,\pm0.30,\pm0.45$ with calibrated values above.
3. Replace 8-D state equation with 13-D state equation.
4. Replace 2e6 timesteps with 3e6 timesteps.
5. Replace entropy coefficient 0.005 with 0.01 (unless you intentionally retrain with 0.005 and regenerate all results).
6. Remove VecNormalize claim or implement VecNormalize and rerun everything.
7. Fix learning rate notation typo to $3\times10^{-4}$.

---

## 8) Recommended short "limitations" paragraph for manuscript

"All reported results derive from a fixed trained policy and the current perturbation protocol; therefore, between-seed variability and algorithm-level uncertainty were not exhaustively quantified. Clustering and decoding analyses characterize representational structure and predictive association, but do not by themselves establish causal mechanisms. Future work should include multi-seed replication, algorithmic baselines, and controlled ablation experiments to test causal contributions of specific latent dimensions."
