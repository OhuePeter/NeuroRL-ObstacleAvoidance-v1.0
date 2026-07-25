# NeuroRL-ObstacleAvoidance-v1.0

Computational neuroscience and reinforcement learning framework for studying behavioural adaptation and latent neural population dynamics during perturbed obstacle-avoidance reaching.

## Quick links

- Fork + reproduce guide: `docs/fork_and_reproduce.md`
- Reproducibility details: `docs/reproducibility_guide.md`
- Figure catalog with explanations: `docs/figure_catalog.md`
- Citation instructions: `docs/citation_guide.md`
- Zenodo and journal checklist: `docs/zenodo_release_checklist.md`

## Why this repository exists

This project asks a focused scientific question:

Can a policy trained without perturbations show adaptation-like neural population structure when perturbations are introduced only at evaluation time?

The repository contains code, analysis, and manuscript assets needed to reproduce the main results for journal submission and Zenodo archiving.

## What is included

- Training and evaluation pipelines for PPO-based reaching control.
- Behavioural and neural analysis pipelines.
- Publication-ready figure and table generation scripts.
- Manuscript source and supporting paper assets.

## Repository structure

- `src/`: package code (environment, training, evaluation, perturbation logic, analysis utilities).
- `scripts/`: executable entry points for training, evaluation, and figure/table generation.
- `configs/`: environment, training, evaluation, and perturbation configuration files.
- `experiments/`: experiment outputs, checkpoints, logs, and derived analysis files.
- `paper/`: manuscript source, manuscript figures, and manuscript tables.
- `docs/`: reproducibility and project documentation.
- `tests/`: automated tests for core components.

## Quick start

### 1. Fork the repository

1. Open the GitHub repository page.
2. Click Fork.
3. Create the fork under your account or organization.
4. Clone the fork:

```bash
git clone https://github.com/<your-user>/NeuroRL-ObstacleAvoidance-v1.0.git
cd NeuroRL-ObstacleAvoidance-v1.0
```

5. Add upstream (recommended):

```bash
git remote add upstream https://github.com/OhuePeter/NeuroRL-ObstacleAvoidance-v1.0.git
git fetch upstream
```

Detailed fork workflow is in `docs/fork_guide.md`.

### 2. Create environment

Target reproducible environment: Python 3.11.

Conda option:

```bash
conda env create -f environment.yml
conda activate neurorl
```

Pip option (inside a Python 3.11 virtual environment):

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Run tests

```bash
pytest -q
```

### 4. One-click pipeline

After installation, run the reproducibility pipeline with a single command:

```bash
neurorl-run
```

Useful options:

```bash
neurorl-run --dry-run
neurorl-run --full
neurorl-run --no-tests
```

## Reproduce results and regenerate figures

Run all commands from repository root.

If you prefer the one-click workflow, `neurorl-run` executes the full post-training pipeline by default.
Use `neurorl-run --full` to include training.

### Step A. Train controller

```bash
python scripts/train.py
```

### Step B. Run Experiment 2 evaluation

```bash
python -m scripts.evaluate_experiment2
```

### Step C. Compute behavioural statistics

```bash
python scripts/analysis/statistical_analysis.py
python -m scripts.analysis.manuscript_statistical_tables
```

### Step D. Run neural analysis

```bash
python -m scripts.analysis.neural_analysis
```

### Step E. Generate manuscript figures

```bash
python scripts/plot_reaching_schematic.py
python -m scripts.analysis.manuscript_behavioral_figures
python -m scripts.analysis.manuscript_neural_figures
```

Expected key outputs:

- `paper/figures/figure1_schematic.pdf`
- `paper/figures/figure2_behavioural_trajectories.pdf`
- `paper/figures/figure3_behavioural_performance.pdf`
- `paper/figures/figure4_behavioural_adaptation.pdf`
- `experiments/version_2_0/results/neural_analysis/manuscript/figure1_neural_summary.pdf`
- `experiments/version_2_0/results/neural_analysis/manuscript/figure2_neural_pca_3d.pdf`
- `experiments/version_2_0/results/neural_analysis/manuscript/figure3_neural_trajectories.pdf`
- `experiments/version_2_0/results/neural_analysis/manuscript/figure4_success_failure.pdf`

Main-text figure policy: 8 figures total, with Figure 1 reserved for the task schematic.

## Documentation index

- `docs/project_overview.md`
- `docs/methodology.md`
- `docs/experiment_protocol.md`
- `docs/figures.md`
- `docs/figure_catalog.md`
- `docs/statistics.md`
- `docs/reproducibility_guide.md`
- `docs/fork_guide.md`
- `docs/fork_and_reproduce.md`
- `docs/citation_guide.md`
- `docs/paper_extract_and_notes.md`

## Figure gallery

### Figure 1: Adaptive reaching schematic

![Figure 1 schematic](paper/figures/figure1_schematic.png)

- Defines task context, geometry, and control framing.

### Figure 2: Behavioural trajectories

![Figure 2 behavioural trajectories](paper/figures/figure2_behavioural_trajectories.png)

- Shows perturbation-dependent trajectory deformation.

### Figure 3: Behavioural performance

![Figure 3 behavioural performance](paper/figures/figure3_behavioural_performance.png)

- Summarizes reward and kinematic performance distributions.

### Figure 4: Behavioural adaptation

![Figure 4 behavioural adaptation](paper/figures/figure4_behavioural_adaptation.png)

- Reports robustness and compensation metrics.

Full figure descriptions for Figures 1-8 are available in `docs/figure_catalog.md`.

## Citation

Citation metadata is provided in `CITATION.cff`.

You can use the generated BibTeX in `docs/citation_guide.md`.

## License

This repository is licensed under the MIT License. See `LICENSE`.

## Funding and affiliation

This work was undertaken in part with support from the Connected Minds Program, Canada First Research Excellence Fund (CFREF), Grant CFREF-2022-00010.

Centre for Neuroscience Studies, Queen's University, Kingston, Ontario, Canada.