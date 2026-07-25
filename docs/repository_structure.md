# Repository Structure

This document describes the role of each top-level folder for reproducibility, review, and archival submission.

## Top-level map

- `src/`: core Python package code.
- `scripts/`: command-line entry points to run training, evaluation, analysis, and plotting.
- `configs/`: YAML configuration files for environment, training, evaluation, and perturbations.
- `data/`: raw and processed data inputs used for evaluation and analysis.
- `experiments/`: run outputs, checkpoints, logs, result tables, and intermediate products.
- `analysis/`: analysis-specific helper outputs and organization folders.
- `figures/`: repository-level figure assets for manuscript, posters, and slides.
- `paper/`: manuscript source (`manuscript.tex`), bibliography, paper figures, and tables.
- `presentation/`: poster/slides/video materials.
- `docs/`: human-readable methodological and reproducibility documentation.
- `tests/`: unit tests and integration checks.

## Reproducibility-critical files

- `environment.yml`: conda environment specification.
- `requirements.txt`: pip dependency list.
- `pyproject.toml`: package metadata and editable install support.
- `CITATION.cff`: machine-readable citation metadata.
- `LICENSE`: MIT license terms.
- `README.md`: high-level quick start and run order.

## Generated versus source content

Source code and manuscript source are authoritative.

Generated outputs include most content under `experiments/version_*/results/`, manuscript figure outputs in `paper/figures/`, and table exports in `paper/tables/`.

When regenerating analyses, existing generated assets may be replaced by new outputs from the scripts documented in `README.md` and `docs/reproducibility_guide.md`.