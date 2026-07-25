# Statistical Analysis

This project reports descriptive and inferential statistics for Experiment 2 perturbation conditions.

## Behavioural outcome groups

- Episode reward
- Episode duration (steps)
- Path length
- Mean and maximum speed
- Peak lateral velocity
- Maximum heading deviation
- Final lateral error
- Success and collision rates

## Condition comparisons

All analyses are condition-wise across:

- `P0` (control)
- `L1`, `L2`, `L3` (left perturbations)
- `R1`, `R2`, `R3` (right perturbations)

## Inferential strategy

1. Assumption checks (normality and homogeneity).
2. Omnibus tests across conditions.
3. Holm-corrected pairwise post hoc comparisons where omnibus tests are significant.

Generated manuscript tables are exported by:

```bash
python -m scripts.analysis.manuscript_statistical_tables
```

Output files are written to `paper/tables/` in CSV and LaTeX formats.