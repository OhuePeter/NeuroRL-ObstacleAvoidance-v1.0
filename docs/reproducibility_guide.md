# Reproducibility Guide

This document describes the recommended command order to regenerate core results and manuscript figures.

For a complete collaborator onboarding path (fork, sync, run, verify), see `docs/fork_and_reproduce.md`.

## Environment setup

### Conda path

```bash
conda env create -f environment.yml
conda activate neurorl
pip install -e .
```

### Pip path

Use Python 3.11, then:

```bash
pip install -r requirements.txt
pip install -e .
```

## Minimum validation

```bash
pytest -q
```

## One-click execution

After `pip install -e .`, run:

```bash
neurorl-run
```

Options:

- `neurorl-run --full`: include policy training.
- `neurorl-run --no-tests`: skip pytest pre-check.
- `neurorl-run --dry-run`: print command plan only.

## Full execution order

Run from repository root.

### 1. Train policy

```bash
python scripts/train.py
```

### 2. Evaluate Experiment 2

```bash
python -m scripts.evaluate_experiment2
```

Primary outputs are created under:

- experiments/version_2_0/results/evaluation_P0
- experiments/version_2_0/results/evaluation_L1
- experiments/version_2_0/results/evaluation_L2
- experiments/version_2_0/results/evaluation_L3
- experiments/version_2_0/results/evaluation_R1
- experiments/version_2_0/results/evaluation_R2
- experiments/version_2_0/results/evaluation_R3

### 3. Behavioural statistics and manuscript tables

```bash
python scripts/analysis/statistical_analysis.py
python -m scripts.analysis.manuscript_statistical_tables
```

Tables are written to paper/tables.

### 4. Behavioural manuscript figures

```bash
python -m scripts.analysis.manuscript_behavioral_figures
```

Figures are written to paper/figures.

### 5. Neural analysis and manuscript figures

```bash
python -m scripts.analysis.neural_analysis
python -m scripts.analysis.manuscript_neural_figures
```

Neural manuscript figures are written to:

- experiments/version_2_0/results/neural_analysis/manuscript

### 6. Figure 1 schematic

```bash
python scripts/plot_reaching_schematic.py
```

Figure 1 is written to paper/figures.

## Manuscript figure map (8 total)

1. Figure 1: schematic
2. Figure 2: behavioural trajectories
3. Figure 3: behavioural performance
4. Figure 4: behavioural adaptation
5. Figure 5: neural summary
6. Figure 6: neural 3D PCA
7. Figure 7: neural trajectories
8. Figure 8: success-versus-failure neural state comparison

## Reproducibility notes

- Fix random seeds where supported by configs.
- Run scripts from repository root to preserve relative paths.
- Keep generated outputs versioned or archived for release snapshots.
