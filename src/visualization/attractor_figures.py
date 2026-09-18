"""
==========================================================
Attractor / Fixed-Point Figures

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Publication-ready figures connecting the closed-loop
fixed-point / vector-field analysis (src/neural_analysis/
attractor_analysis.py) to the observed velocity profiles and
to the latent PCA manifold and cluster structure.

The trained policy is conditioned on a discrete route cue
(route_signal = -1 left detour, +1 right detour, 0 neither),
set by the alternating left/right training curriculum
(configs/environment.yaml, training.route_balancing). This
means the closed loop is multistable: there is a different
lateral fixed point for the left-route context and for the
right-route context. Every real evaluation episode is
automatically classified into one of these two routes from its
own trajectory (by which side of the midline it passes the
obstacle pair on) so it is compared against the matching
fixed point, not a pooled, meaningless average of the two.

Produces three figures:

  fig_attractor_phase_portrait : lateral (x, vx) phase portrait,
      one panel per route, with the closed-loop vector field,
      the route-specific fixed point, and real evaluation
      trajectories overlaid, at the corridor height where the
      perturbation is delivered.

  fig_attractor_path : the two fixed-point curves x*(y) across
      the corridor, compared against the empirical trajectories
      of the matching route and the obstacle boundaries.

  fig_attractor_latent : the fixed-point curves' latent
      activity, projected into the same PC1-PC2 space as the
      clustering result, overlaid on the cluster scatter.

Version:
1.0
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.neural_analysis.attractor_analysis import AttractorAnalysis
from src.visualization.colors import COLORS

ROUTE_STYLE = {
    "left": {"color": "#1F78B4", "label": "Left-detour route"},
    "right": {"color": "#CB181D", "label": "Right-detour route"},
}


class AttractorFigures:

    def __init__(self, results_root, model_path):

        self.root = Path(results_root)
        self.output_dir = self.root / "neural_analysis" / "attractor"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analysis = AttractorAnalysis(model_path)

    # ------------------------------------------------------
    # Data loading
    # ------------------------------------------------------

    def _load_kinematics(self, condition):

        folder = self.root / f"evaluation_{condition}"
        files = sorted(folder.glob("kinematics_*.csv"))

        return [pd.read_csv(file) for file in files]

    @staticmethod
    def _classify_route(df, midline_x):
        """
        Infer which side of the obstacle pair an episode went
        around from its own trajectory, since route is not stored
        per-episode by this evaluator. The two obstacles sit at
        y in [4.5, 5.5]; whichever side of the midline the agent
        is on while passing that band is its route.
        """

        band = df[(df["y"] >= 4.5) & (df["y"] <= 5.5)]

        if band.empty:
            return None

        return "left" if band["x"].mean() < midline_x else "right"

    def _estimate_plateau_speed(self, condition="P0"):
        """Mean forward speed once the agent has reached its
        cruising phase (steps 60-100), used as the fixed vy
        for the reduced 2-D lateral analysis."""

        frames = self._load_kinematics(condition)

        speeds = [
            df.loc[(df["step"] >= 60) & (df["step"] <= 100), "vy"].mean()
            for df in frames
        ]

        return float(np.nanmean(speeds))

    def _trajectories_by_route(self, conditions):
        """Group every loaded episode, across all conditions, by
        its inferred route."""

        midline_x = self.analysis.midline_x
        grouped = {"left": [], "right": []}

        for condition in conditions:

            for df in self._load_kinematics(condition):

                route = self._classify_route(df, midline_x)

                if route is not None:
                    grouped[route].append((condition, df))

        return grouped

    # ------------------------------------------------------
    # Figure 1: lateral phase portrait
    # ------------------------------------------------------

    def phase_portrait(
        self,
        y_probe=5.0,
        conditions=("P0", "L1", "L2", "L3", "R1", "R2", "R3"),
        x_range=np.linspace(1.5, 8.5, 29),
        vx_range=np.linspace(-1.0, 1.0, 18),
    ):

        vy_plateau = self._estimate_plateau_speed("P0")
        grouped = self._trajectories_by_route(conditions)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

        report = {}

        for ax, route in zip(axes, ("left", "right")):

            X, VX, DX, DVX = self.analysis.lateral_vector_field(
                x_range, vx_range, y_probe, vy_plateau, route=route
            )

            x_star = self.analysis.find_lateral_fixed_point(
                y_probe, vy_plateau, route=route
            )
            _, eigenvalues = self.analysis.local_jacobian(
                x_star, y_probe, vy_plateau, route=route
            )
            tau = max(
                self.analysis.time_constant(ev, self.analysis.dt)
                for ev in eigenvalues
            )

            report[route] = {
                "x_star": x_star,
                "eigenvalues": eigenvalues,
                "tau": tau,
            }

            ax.quiver(
                X, VX, DX, DVX,
                color="#B0B0B0",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
                alpha=0.8,
            )

            for condition, df in grouped[route]:

                window = df[
                    (df["y"] >= y_probe - 0.35)
                    & (df["y"] <= y_probe + 0.35)
                ]

                if window.empty:
                    continue

                ax.plot(
                    window["x"],
                    window["vx"],
                    color=COLORS[condition],
                    alpha=0.45,
                    linewidth=1.1,
                )

            ax.scatter(
                [x_star], [0.0],
                color="black",
                marker="*",
                s=220,
                zorder=5,
                label=f"Fixed point ($x^*$={x_star:.2f})",
            )

            ax.axvline(
                self.analysis.midline_x,
                color="gray",
                linestyle=":",
                linewidth=1,
                label="Corridor midline",
            )

            ax.set_xlabel("Lateral position, x (m)", fontsize=12)
            ax.set_title(
                f"{ROUTE_STYLE[route]['label']}\n"
                f"$\\tau$ = {tau*1000:.0f} ms",
                fontsize=12,
            )
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.25)

        axes[0].set_ylabel("Lateral velocity, $v_x$ (m/s)", fontsize=12)

        fig.suptitle(
            f"Lateral phase portrait at y = {y_probe:.1f} m, "
            "one attractor per route",
            fontsize=13,
        )
        fig.tight_layout()

        self._save(fig, "fig_attractor_phase_portrait")

        return report

    # ------------------------------------------------------
    # Figure 2: fixed-point curve across the corridor
    # ------------------------------------------------------

    def fixed_point_path(
        self,
        y_values=np.linspace(1.5, 8.5, 29),
        conditions=("P0", "L1", "L2", "L3", "R1", "R2", "R3"),
    ):

        vy_plateau = self._estimate_plateau_speed("P0")
        grouped = self._trajectories_by_route(conditions)

        curves = {
            route: self.analysis.fixed_point_curve(
                y_values, vy_plateau, route=route
            )
            for route in ("left", "right")
        }

        fig, ax = plt.subplots(figsize=(7.5, 8))

        left = self.analysis.obstacle_left
        right = self.analysis.obstacle_right

        for obstacle in (left, right):

            circle = plt.Circle(
                obstacle.position,
                obstacle.radius,
                color=COLORS["obstacle"],
                alpha=0.5,
            )
            ax.add_patch(circle)

        for route in ("left", "right"):

            for i, (condition, df) in enumerate(grouped[route]):

                ax.plot(
                    df["x"], df["y"],
                    color=ROUTE_STYLE[route]["color"],
                    alpha=0.12,
                    linewidth=1.0,
                    label=(
                        f"{ROUTE_STYLE[route]['label']} episodes"
                        if i == 0
                        else None
                    ),
                )

            curve = curves[route]

            ax.plot(
                curve[:, 1], curve[:, 0],
                color="black" if route == "left" else "dimgray",
                linewidth=2.5,
                linestyle="--" if route == "left" else "-.",
                label=f"Fixed point $x^*(y)$, {route} route",
            )

        ax.axvline(
            self.analysis.midline_x,
            color="gray",
            linestyle=":",
            linewidth=1,
        )

        ax.set_xlim(0.5, 9.5)
        ax.set_xlabel("Lateral position, x (m)", fontsize=12)
        ax.set_ylabel("Corridor height, y (m)", fontsize=12)
        ax.set_title(
            "Route-specific closed-loop fixed points track the\n"
            "empirical detour path around the obstacle pair",
            fontsize=13,
        )
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.grid(alpha=0.25)

        fig.tight_layout()

        self._save(fig, "fig_attractor_path")

        return curves

    # ------------------------------------------------------
    # Figure 3: fixed-point curve in latent PC space
    # ------------------------------------------------------

    def latent_projection(
        self,
        curves,
        conditions=("P0", "L1", "L2", "L3", "R1", "R2", "R3"),
    ):

        dataset = np.load(
            self.root / "latent_dataset.npz", allow_pickle=True
        )

        X = dataset["activations"]
        condition = dataset["condition"]

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)

        vy_plateau = self._estimate_plateau_speed("P0")

        fig, ax = plt.subplots(figsize=(8, 6.5))

        for c in conditions:

            idx = condition == c

            ax.scatter(
                scores[idx, 0], scores[idx, 1],
                s=6, alpha=0.2, color=COLORS[c], label=c,
            )

        for route, curve in curves.items():

            curve_latents = np.array([
                self.analysis.latent(
                    self.analysis._observation(
                        x, y, 0.0, vy_plateau, route=route
                    )
                )
                for y, x in curve
            ])

            curve_scores = pca.transform(curve_latents)

            ax.plot(
                curve_scores[:, 0], curve_scores[:, 1],
                color=ROUTE_STYLE[route]["color"],
                linewidth=2.5,
                marker="o",
                markersize=3,
                label=f"Fixed-point curve, {route} route",
            )

        ax.set_xlabel(
            f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)",
            fontsize=12,
        )
        ax.set_ylabel(
            f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)",
            fontsize=12,
        )
        ax.set_title(
            "The closed-loop attractors trace a path through the\n"
            "same low-dimensional manifold as the evaluation data",
            fontsize=13,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)

        fig.tight_layout()

        self._save(fig, "fig_attractor_latent")

    # ------------------------------------------------------
    # Saving
    # ------------------------------------------------------

    def _save(self, fig, name):

        for ext in ("png", "pdf"):

            fig.savefig(
                self.output_dir / f"{name}.{ext}",
                dpi=600,
                bbox_inches="tight",
            )

        plt.close(fig)
