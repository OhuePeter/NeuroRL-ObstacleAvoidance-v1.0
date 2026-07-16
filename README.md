# NeuroRL-ObstacleAvoidance-v1.0

## Overview

NeuroRL-ObstacleAvoidance-v1.0 is a computational neuroscience and reinforcement-learning project for studying how internal latent neural activity changes when a trained controller performs goal-directed obstacle avoidance under unexpected perturbations.

The core scientific question is whether population-level structure resembling biological motor adaptation emerges inside a reinforcement-learning policy when perturbations are introduced only at evaluation time.

## Manuscript Scope

This repository currently supports a manuscript pipeline with:

- Figure 1: adaptive reaching schematic
- Figures 2-4: behavioural manuscript figures
- Figures 5-8: neural manuscript figures
- manuscript-ready behavioural statistical tables
- reproducible experiment, evaluation, and analysis scripts

The current manuscript assets live primarily under [paper](paper), [paper/figures](paper/figures), and [paper/tables](paper/tables).

## Scientific Summary

- A PPO agent is trained for obstacle-avoiding reaching.
- Perturbations are withheld during training and introduced during Experiment 2 evaluation.
- Evaluation records behaviour, kinematics, hidden-layer activations, actions, rewards, and summary metrics.
- Offline analysis quantifies behavioural robustness and neural population organization using PCA, RSA, decoding, and success-versus-failure comparisons.

## Repository Layout

- [src](src): environments, evaluation logic, perturbations, training, visualization
- [scripts](scripts): training, evaluation, analysis, and manuscript-asset generation entrypoints
- [experiments](experiments): checkpoints, results, figures, logs, and exported analysis outputs
- [paper](paper): manuscript text, figure plan, tables, and paper-facing figures
- [docs](docs): project notes and methodology documents
- [tests](tests): unit and integration tests for core components

## Fork And Reproduce

### 1. Fork the repository on GitHub

1. Open the repository on GitHub.
2. Click `Fork`.
3. Create your fork under your own account or organization.
4. Clone your fork locally:

```bash
git clone https://github.com/<your-user>/NeuroRL-ObstacleAvoidance-v1.0.git
cd NeuroRL-ObstacleAvoidance-v1.0
```

5. Optionally add the original repository as `upstream`:

```bash
git remote add upstream https://github.com/OhuePeter/NeuroRL-ObstacleAvoidance-v1.0.git
git fetch upstream
```

### 2. Create the Python environment

The reproducible target environment is Python 3.11, as defined in [environment.yml](environment.yml).

```bash
conda env create -f environment.yml
conda activate neurorl
```

If you prefer `pip`, install the dependencies directly after creating a Python 3.11 environment.

```bash
pip install -r requirements.txt
```

### 3. Install the repository in editable mode

Editable installation is now supported through [pyproject.toml](pyproject.toml).

```bash
pip install -e .
```

This keeps local source edits live without reinstalling the package.

## Reproducibility Workflow

The commands below reproduce the main experimental and manuscript-facing outputs from the repository root.

### Training

Train the base controller:

```bash
python scripts/train.py
```

### Experiment 2 evaluation

Generate behavioural outputs, neural recordings, and per-condition summaries:

```bash
python -m scripts.evaluate_experiment2
```

This produces outputs under [experiments/version_2_0/results](experiments/version_2_0/results), including:

- `evaluation_P0` through `evaluation_R3`
- `summary.csv` files per condition
- `trajectory_*.csv` and `kinematics_*.csv`
- `neural/policy_*.npy`, `neural/value_*.npy`, and related arrays

### Statistical analysis

Generate omnibus behavioural statistics:

```bash
python scripts/analysis/statistical_analysis.py
```

Generate manuscript-ready statistical tables:

```bash
python -m scripts.analysis.manuscript_statistical_tables
```

### Neural analysis

Generate offline neural analyses from saved latent activations:

```bash
python -m scripts.analysis.neural_analysis
```

Generate manuscript-ready neural figures:

```bash
python -m scripts.analysis.manuscript_neural_figures
```

### Behavioural manuscript figures

Generate manuscript-ready behavioural figures:

```bash
python -m scripts.analysis.manuscript_behavioral_figures
```

### Figure 1 schematic

Generate the publication-ready task schematic:

```bash
python scripts/plot_reaching_schematic.py
```

## Manuscript Assets

Main manuscript assets currently include:

- [paper/manuscript.tex](paper/manuscript.tex)
- [paper/figure_plan.md](paper/figure_plan.md)
- [paper/figures](paper/figures)
- [paper/tables](paper/tables)

The current figure plan is capped at 8 main-text figures, with the schematic fixed as Figure 1.

## Development Notes

- The codebase uses `src.*` imports and now supports editable installation.
- Generated experiment outputs are stored under [experiments](experiments).
- Paper-facing derived assets are stored under [paper](paper).
- Manuscript-only helper scripts live under [scripts/analysis](scripts/analysis).

## Citation And Funding

If you use this repository, please cite the project and manuscript materials as appropriate.

This work was undertaken thanks in part to funding from the Connected Minds Program, supported by the Canada First Research Excellence Fund (CFREF), Grant `CFREF-2022-00010`.

## Institutional Affiliation

Centre for Neuroscience Studies  
Queen's University, Kingston, Ontario, Canada

This repository was developed within the CompSci Lab at Queen's University.