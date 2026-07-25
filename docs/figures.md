# Figure Plan (Main Text)

Main-text figure cap: 8 total.

## Figure list

1. `Figure 1`: Adaptive reaching schematic (fixed manuscript schematic).
2. `Figure 2`: Behavioural trajectory structure across perturbation conditions.
3. `Figure 3`: Behavioural performance distributions (reward, duration, path length, final lateral error).
4. `Figure 4`: Behavioural robustness and adaptation metrics.
5. `Figure 5`: Neural population summary (PCA + RSA + decoding overview).
6. `Figure 6`: Neural 3D PCA geometry.
7. `Figure 7`: Neural latent trajectories.
8. `Figure 8`: Success-versus-failure neural state comparison.

## Generation commands

Run from repository root.

```bash
python scripts/plot_reaching_schematic.py
python -m scripts.analysis.manuscript_behavioral_figures
python -m scripts.analysis.neural_analysis
python -m scripts.analysis.manuscript_neural_figures
```

## Output locations

- Behavioural figures: `paper/figures/`
- Neural figures: `experiments/version_2_0/results/neural_analysis/manuscript/`
- Schematic: `paper/figures/`