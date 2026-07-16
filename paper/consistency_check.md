# Manuscript Consistency Check

## Scope checked

- manuscript figure order
- figure-file existence for main-text figures
- table-file existence for inserted statistical tables
- consistency with the fixed 8-figure plan

## Current status

### Figure order
The manuscript currently follows the intended sequence:
- Figure 1: schematic
- Figures 2-4: behavioural results
- Figures 5-8: neural results

### Main-text figures
Verified present:
- paper/figures/figure1_schematic.pdf
- paper/figures/figure2_behavioural_trajectories.pdf
- paper/figures/figure3_behavioural_performance.pdf
- paper/figures/figure4_behavioural_adaptation.pdf
- experiments/version_2_0/results/neural_analysis/manuscript/figure1_neural_summary.pdf
- experiments/version_2_0/results/neural_analysis/manuscript/figure2_neural_pca_3d.pdf
- experiments/version_2_0/results/neural_analysis/manuscript/figure3_neural_trajectories.pdf
- experiments/version_2_0/results/neural_analysis/manuscript/figure4_success_failure.pdf

### Inserted statistical tables
Verified present:
- paper/tables/table1_descriptive_statistics.tex
- paper/tables/table2_assumption_tests.tex
- paper/tables/table3_omnibus_tests.tex
- paper/tables/table4_pairwise_posthoc.tex

## Residual issues to keep in mind

### Main-text density
The manuscript already uses the full 8-figure budget. Any new figure should replace an existing main-text figure or move to supplementary material.

### Table placement
If journal space becomes tight, move the assumption table and full pairwise post hoc table to supplementary material first.

### Path convention
Current manuscript figure paths are consistent but come from two locations:
- paper/figures for manuscript-generated schematic and behavioural figures
- experiments/version_2_0/results/neural_analysis/manuscript for neural figures

If desired later, neural figure PDFs can be copied into paper/figures for a single manuscript-asset directory, but this is not currently required for consistency.

## Conclusion

The current manuscript structure is internally consistent with the figure plan, current generated assets, and inserted statistical tables.