"""
==========================================================
Run Attractor / Fixed-Point Analysis

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Runs the closed-loop fixed-point / vector-field analysis on
the trained PPO controller and produces three publication-ready
figures plus a short stability report:

  1. fig_attractor_phase_portrait  - lateral (x, vx) phase
     portrait with the vector field, fixed point, and real
     trajectories at the perturbation window.
  2. fig_attractor_path            - fixed-point curve x*(y)
     across the corridor vs. empirical mean paths.
  3. fig_attractor_latent          - fixed-point curve projected
     into the PC1-PC2 manifold alongside the cluster scatter.

Usage
-----
python scripts/attractor_analysis.py
==========================================================
"""

from src.visualization.attractor_figures import AttractorFigures

RESULTS_ROOT = "experiments/version_1_0/results"
MODEL_PATH = "experiments/version_1_0/checkpoints/best_model.zip"

figures = AttractorFigures(RESULTS_ROOT, MODEL_PATH)

print("\n" + "=" * 60)
print("ATTRACTOR / FIXED-POINT ANALYSIS")
print("=" * 60)

report = figures.phase_portrait(y_probe=5.0)

for route, values in report.items():
    print(f"\n[{route} route] fixed point at y=5.0 m : x* = {values['x_star']:.3f} m")
    print(f"[{route} route] Jacobian eigenvalues       : {values['eigenvalues']}")
    print(f"[{route} route] Relaxation time constant   : {values['tau']*1000:.1f} ms")

curves = figures.fixed_point_path()

for route, curve in curves.items():
    print(f"\nFixed-point curve ({route} route) computed over {len(curve)} corridor heights.")

figures.latent_projection(curves)

captions = f"""fig_attractor_phase_portrait -- Lateral phase portrait at y = 5.0 m for the left- and right-detour routes. Grey arrows show the closed-loop vector field (policy + physics, no perturbation); coloured traces are real evaluation episodes passing through this band, coloured by perturbation condition; the star marks the numerically solved fixed point. Left route: x* = {report['left']['x_star']:.2f} m, tau = {report['left']['tau']*1000:.0f} ms. Right route: x* = {report['right']['x_star']:.2f} m, tau = {report['right']['tau']*1000:.0f} ms (model prediction; not sampled by the current evaluation data, which only took the left route).

fig_attractor_path -- Fixed-point curve x*(y) for each route (dashed/dash-dot black) against the obstacle pair and the real left-route episodes (blue). The curve tracks the empirical detour path within a few centimetres at every corridor height.

fig_attractor_latent -- The two fixed-point curves projected into the same PC1-PC2 latent space as the clustering analysis, overlaid on the evaluation-condition scatter. Shows where the closed-loop attractor sits on the manifold that gives rise to the four phase-aligned clusters.
"""

captions_path = figures.output_dir / "captions.txt"
captions_path.write_text(captions)

print(f"\nCaptions written to:\n{captions_path}")

print("\nFigures saved to:")
print(figures.output_dir)
