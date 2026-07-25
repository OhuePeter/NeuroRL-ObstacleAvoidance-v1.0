# Paper Extract and Notes

This document captures a concise extract and editorial notes from the manuscript source for release packaging.

## Short extract

The study evaluates whether adaptation-like behavioural and latent neural signatures can emerge in a reinforcement learning controller when perturbations are absent during training and introduced only during evaluation.

A PPO policy controls a point-mass agent in a two-dimensional obstacle-avoidance reaching task. Perturbation conditions (P0, L1-L3, R1-R3) are applied in a bounded trigger region during Experiment 2. Behavioural outcomes and latent states are recorded and analyzed with behavioural statistics, principal component analysis, representational similarity analysis, and decoding analyses.

## Manuscript-facing notes

- Figure 1 is the task schematic.
- Main text is constrained to 8 figures total.
- Behavioural figures map to Figures 2-4.
- Neural figures map to Figures 5-8.
- Statistical tables are generated into paper/tables for manuscript inclusion.

## Source of truth

Primary manuscript source is paper/manuscript.tex.

All figure generation commands are documented in README.md and docs/reproducibility_guide.md.
