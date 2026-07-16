# Manuscript Figure Plan

## Figure budget

Maximum main-text figures: 8

Constraint:
- Figure 1 must be the task schematic.

Current committed or generated main-text candidates:
- Figure 1: task schematic
- Figure 2: behavioural trajectories
- Figure 3: behavioural performance
- Figure 4: behavioural adaptation
- Neural summary panel
- Neural 3D PCA
- Neural trajectories
- Neural success vs failure

This fills the full 8-figure main-text budget.

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
Source:
- paper/figures/figure2_behavioural_trajectories.pdf
Script:
- scripts/analysis/manuscript_behavioral_figures.py
Purpose:
- Show how perturbations deform reach geometry at the behavioural level.
Status:
- Ready

### Figure 3. Behavioural performance summary
Source:
- paper/figures/figure3_behavioural_performance.pdf
Script:
- scripts/analysis/manuscript_behavioral_figures.py
Purpose:
- Present the main behavioural outcome measures in one paper-ready figure.
Status:
- Ready

### Figure 4. Behavioural strategy or failure analysis
Source:
- paper/figures/figure4_behavioural_adaptation.pdf
Script:
- scripts/analysis/manuscript_behavioral_figures.py
Purpose:
- Show robustness breakdown and adaptation across perturbation conditions.
Status:
- Ready

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

1. Keep Figures 1 to 8 fixed unless the manuscript narrative changes substantially.
2. Move excess diagnostics and alternate behavioural visualizations to supplementary material.
3. Revise manuscript prose section by section against the current fixed figure and table set.
