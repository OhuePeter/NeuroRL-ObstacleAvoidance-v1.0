# Manuscript Figure Plan

## Figure budget

Maximum main-text figures: 8

Constraint:
- Figure 1 must be the task schematic.

Current committed or generated main-text candidates:
- Figure 1: task schematic
- Neural summary panel
- Neural 3D PCA
- Neural trajectories
- Neural success vs failure

This leaves 3 remaining main-text figure slots.

## Recommended main-text figure sequence

### Figure 1. Adaptive reaching paradigm and closed-loop controller
Source:
- paper/figures/figure1_schematic.pdf
Script:
- scripts/plot_reaching_schematic.py
Purpose:
- Introduce the task, perturbations, obstacle, and PPO control loop.
Status:
- Ready

### Figure 2. Behavioural trajectory overview across perturbation conditions
Suggested content:
- Mean trajectory per condition
- Overlay trajectories or representative trials
- Clear separation of left and right perturbations
Candidate scripts:
- scripts/analysis/plot_mean_trajectories.py
- scripts/analysis/plot_overlay_trajectories.py
- scripts/plot_overlay_trajectories.py
Purpose:
- Show how perturbations deform reach geometry at the behavioural level.
Status:
- To assemble as manuscript panel

### Figure 3. Behavioural performance summary
Suggested content:
- Compact multi-panel boxplot or violin figure
- Reward
- Duration
- Path length
- Final lateral error
Candidate script:
- scripts/analysis/publication_figures.py
Purpose:
- Present the main behavioural outcome measures in one paper-ready figure.
Status:
- Needs panel selection and manuscript export

### Figure 4. Behavioural strategy or failure analysis
Suggested content:
- Failure trajectories, route selection, or cumulative perturbation effects
- Prefer one message only: robustness breakdown or route adaptation
Candidate scripts:
- scripts/plot_failure_trajectories.py
- scripts/analysis/plot_route_selection.py
- scripts/plot_cumulative_trajectories.py
Purpose:
- Show the behavioural mechanism behind condition-dependent success and failure.
Status:
- Choose one strongest story, move the others to supplement

### Figure 5. Neural population summary
Source:
- experiments/version_2_0/results/neural_analysis/manuscript/figure1_neural_summary.pdf
Script:
- scripts/analysis/manuscript_neural_figures.py
Purpose:
- Summarize condition separation using PCA, RSA, and decoding.
Status:
- Ready

### Figure 6. Neural manifold geometry in 3D PCA space
Source:
- experiments/version_2_0/results/neural_analysis/manuscript/figure2_neural_pca_3d.pdf
Script:
- scripts/analysis/manuscript_neural_figures.py
Purpose:
- Show low-dimensional latent geometry across perturbation conditions.
Status:
- Ready

### Figure 7. Neural trajectories through latent space
Source:
- experiments/version_2_0/results/neural_analysis/manuscript/figure3_neural_trajectories.pdf
Script:
- scripts/analysis/manuscript_neural_figures.py
Purpose:
- Show temporal evolution of neural state trajectories by condition and outcome.
Status:
- Ready

### Figure 8. Success-versus-failure neural comparison
Source:
- experiments/version_2_0/results/neural_analysis/manuscript/figure4_success_failure.pdf
Script:
- scripts/analysis/manuscript_neural_figures.py
Purpose:
- Compare latent neural organization between successful and unsuccessful episodes.
Status:
- Ready

## Recommended supplementary figures

Move these out of the main text unless they are essential to a central claim:
- Hidden-unit correlation heatmaps
- Additional decoding confusion matrices at full resolution
- Full RSA matrices if already summarized in a main panel
- All behavioural metrics not selected for Figure 3
- Additional trajectory overlays for every condition
- Raw kinematic profile plots

## Table strategy

Keep main behavioural tables in the manuscript or appendix as follows:
- Main text or appendix:
  - omnibus tests
  - key descriptive table
- Supplementary:
  - full pairwise post hoc table if journal space is limited
  - full assumption table if not required in the main text

Current generated tables:
- paper/tables/table1_descriptive_statistics.tex
- paper/tables/table2_assumption_tests.tex
- paper/tables/table3_omnibus_tests.tex
- paper/tables/table4_pairwise_posthoc.tex

## Recommended next production steps

1. Convert behavioural figures into manuscript-ready panels for Figure 2 and Figure 3.
2. Decide whether Figure 4 should emphasize failure patterns or route-selection strategy.
3. Keep Figures 5 to 8 as the finalized neural result block unless the manuscript narrative changes.
4. Move excess diagnostics to supplementary material early to avoid figure-count drift.
