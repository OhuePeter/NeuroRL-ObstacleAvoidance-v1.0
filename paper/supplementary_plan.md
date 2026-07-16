# Supplementary Material Plan

## Goal

Keep the main manuscript fixed at 8 figures while moving diagnostics, exhaustive comparisons, and lower-priority visualizations to supplementary material.

## Recommended supplementary figures

### Supplementary Figure S1. Full behavioural metric expansion
Content:
- behavioural metrics not shown in main-text Figure 3
- mean speed
- maximum speed
- peak lateral velocity if not retained in the main text narrative
- maximum heading deviation if the adaptation panel is kept compact
Primary sources:
- paper/tables/table1_descriptive_statistics.csv
- experiments/version_2_0/results/statistics/descriptive_statistics.csv

### Supplementary Figure S2. Full-condition overlay trajectories
Content:
- all-condition trajectory overlays
- denser visualization than main-text Figure 2
Primary sources:
- experiments/version_2_0/results/evaluation_*/trajectory_*.csv

### Supplementary Figure S3. Failure-focused trajectory analysis
Content:
- failed trials only
- representative failure trajectories for the strongest perturbation conditions
Primary sources:
- experiments/version_2_0/results/evaluation_R3/summary.csv
- experiments/version_2_0/results/evaluation_R3/trajectory_*.csv

### Supplementary Figure S4. Neural RSA at full scale
Content:
- enlarged policy and value RSA matrices if the main figure remains compact
Primary sources:
- experiments/version_2_0/results/neural_analysis/rsa_policy.csv
- experiments/version_2_0/results/neural_analysis/rsa_value.csv

### Supplementary Figure S5. Hidden-unit correlation structure
Content:
- policy hidden-unit correlation heatmap
- value hidden-unit correlation heatmap
Primary sources:
- experiments/version_2_0/results/neural_analysis/hidden_unit_correlation_policy.csv
- experiments/version_2_0/results/neural_analysis/hidden_unit_correlation_value.csv

### Supplementary Figure S6. Full decoding confusion matrices
Content:
- policy and value confusion matrices at full resolution
- optional normalized and raw-count variants
Primary sources:
- experiments/version_2_0/results/neural_analysis/decoding_confusion_policy.csv
- experiments/version_2_0/results/neural_analysis/decoding_confusion_value.csv

## Recommended supplementary tables

### Supplementary Table S1. Assumption tests
Keep [paper/tables/table2_assumption_tests.tex](paper/tables/table2_assumption_tests.tex) in supplement if space is limited.

### Supplementary Table S2. Full pairwise post hoc tests
Keep [paper/tables/table4_pairwise_posthoc.tex](paper/tables/table4_pairwise_posthoc.tex) in supplement if journal constraints require a shorter main text.

## Main-text priority

Keep these in the main manuscript unless the story changes:
- Figure 1: schematic
- Figures 2-4: behavioural results
- Figures 5-8: neural results
- omnibus statistical table
- one compact descriptive table

## Editing rule

When adding new visuals, default them to supplementary unless they replace an existing main-text figure.