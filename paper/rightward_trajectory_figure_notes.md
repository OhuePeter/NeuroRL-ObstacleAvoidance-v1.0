# Rightward Trajectory Figure Notes

## Behavioural trajectory caption draft

Figure 2 shows the behavioural geometry of the controller after explicit route-waypoint training. The perturbation means separate cleanly by direction: leftward perturbations bend trajectories to the left, while rightward perturbations bend trajectories to the right. The representative panel intentionally overlays successful rightward trajectories (control and small-right trials) with failed right-perturbation trajectories so both adaptive and non-adaptive regimes are visible in one frame. This panel makes the rightward pathway explicit, which was the missing feature in the earlier model.

## Neural trajectory caption draft

The neural trajectory panel highlights latent-state trajectories associated with successful rightward movements and failed rightward perturbation trials. Successful episodes follow compact and repeatable latent paths, while failed episodes branch away early and remain separated from the success manifold. The divergence is structured rather than noisy, supporting an interpretable neural population-level failure mode.

## Discussion draft

The key result after enforcing explicit left-right route control is a stronger directional split in behaviour. The controller now shows true rightward and leftward route families, but perturbation robustness becomes conditional on route and perturbation direction. In this run, control trials remain perfect while perturbed conditions show mixed success and failure (typically around half success for stronger loads). Biologically, this pattern is plausible for a controller that has learned distinct directional motor plans with finite corrective margins: once the perturbation pushes the movement outside the active plan's recovery basin, failures become stereotyped rather than random.

Recommended text for results section:

After retraining with explicit route-waypoint conditioning, trajectories separated into distinct leftward and rightward families as intended. Control trials were stable (100 percent success), whereas perturbation conditions showed graded reductions in robustness (L1: 65 percent success; L2-L3: 50 percent; R1-R3: 50 percent). This training regime therefore solved the route-diversity objective at the cost of broader perturbation sensitivity, consistent with a policy that allocates representational capacity toward directional path specification.

## Inkscape handoff notes

- Use the `.svg` versions first when possible. They preserve separate strokes, text, and panel labels more cleanly for post-processing.
- Keep line widths consistent across panels before final export. For manuscript figures, avoid changing individual trajectory stroke widths independently unless the panel is explicitly meant to emphasize exemplar trials.
- In Inkscape, group each panel before repositioning labels so scale changes do not shift text relative to axes.
- Convert text to paths only at the very end, after all wording is final.
- If a panel feels busy, reduce opacity before deleting trajectories. The failed rightward trajectories should stay visible enough to support the asymmetry claim.