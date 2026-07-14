"""
==========================================================
Figure 1

Adaptive Goal-Directed Reaching Framework

Author:
Peter Ohue

Description
-----------
Publication-quality schematic illustrating the adaptive
goal-directed reaching paradigm using a PPO-controlled
point-mass agent.

==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
)

from matplotlib.path import Path as MplPath


class ReachingSchematic:
    """
    Publication-quality schematic for the manuscript.
    """

    def __init__(self):

        self.output = Path(
            "experiments/version_1_0/figures"
        )

        self.output.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Colour palette
        # --------------------------------------------------

        self.BLACK = "#222222"
        self.GRAY = "#D9D9D9"
        self.BLUE = "#2F6DB2"
        self.WHITE = "#FFFFFF"

    # ------------------------------------------------------

    def draw_box(
        self,
        ax,
        x,
        y,
        w,
        h,
        label,
    ):
        """
        Draw a rounded publication-style box.
        """

        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03",
            linewidth=1.5,
            edgecolor=self.BLACK,
            facecolor=self.WHITE,
        )

        ax.add_patch(box)

        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
        )

    # ------------------------------------------------------

    def draw(self):

        plt.rcParams["font.family"] = "DejaVu Sans"

        fig, ax = plt.subplots(
            figsize=(15, 8)
        )

        ax.set_xlim(0, 15)
        ax.set_ylim(0, 8)

        ax.set_aspect("equal")
        ax.axis("off")

        # --------------------------------------------------
        # Title
        # --------------------------------------------------

        ax.text(
            7.5,
            7.55,
            "Adaptive Goal-Directed Reaching Under External Perturbations",
            ha="center",
            fontsize=18,
            fontweight="bold",
        )

        # ==================================================
        # Workspace
        # ==================================================

        # Workspace centre
        cx = 4.0

        # Key positions
        start = (cx, 1.0)
        goal = (cx, 6.6)
        obstacle = (cx, 3.8)

        # Current agent position (close to the start)
        agent = (cx, 1.35)

        # --------------------------------------------------
        # Start
        # --------------------------------------------------

        ax.plot(
            start[0],
            start[1],
            "o",
            color=self.BLACK,
            markersize=8,
            zorder=10,
        )

        ax.text(
            start[0],
            start[1] - 0.35,
            "Start",
            ha="center",
            fontsize=11,
        )

        # --------------------------------------------------
        # Goal
        # --------------------------------------------------

        ax.add_patch(
            Circle(
                goal,
                radius=0.15,
                facecolor="white",
                edgecolor=self.BLUE,
                linewidth=2,
                zorder=10,
            )
        )

        ax.text(
            goal[0],
            goal[1] + 0.35,
            "Goal",
            ha="center",
            fontsize=11,
        )

        # --------------------------------------------------
        # Obstacle
        # --------------------------------------------------

        ax.add_patch(
            Circle(
                obstacle,
                radius=0.45,
                facecolor=self.GRAY,
                edgecolor=self.BLACK,
                linewidth=1.5,
                zorder=5,
            )
        )

        ax.text(
            obstacle[0],
            obstacle[1] - 0.75,
            "Obstacle",
            ha="center",
            fontsize=10,
        )

        # --------------------------------------------------
        # Smooth trajectory
        # --------------------------------------------------

        verts = [

            start,

            (2.70, 2.20),

            (2.60, 5.30),

            goal,

        ]

        codes = [

            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,

        ]

        trajectory = MplPath(
            verts,
            codes,
        )

        ax.add_patch(

            PathPatch(

                trajectory,

                facecolor="none",

                edgecolor=self.BLACK,

                linewidth=2.0,

                linestyle="--",

                zorder=1,

            )

        )

        # Direction arrow near the goal

        ax.annotate(

            "",

            xy=(3.95, 6.35),

            xytext=(3.55, 5.95),

            arrowprops=dict(

                arrowstyle="-|>",

                lw=1.8,

                color=self.BLACK,

            ),

        )

        # --------------------------------------------------
        # Agent
        # --------------------------------------------------

        ax.plot(

            agent[0],

            agent[1],

            "o",

            color=self.BLACK,

            markersize=9,

            zorder=20,

        )

        ax.text(

            agent[0] + 0.25,

            agent[1],

            "Agent",

            fontsize=10,

            va="center",

        )

        # ==================================================
        # Perturbation
        # Applied shortly after movement onset
        # ==================================================

        perturb_y = 2.15

        # Left perturbation

        ax.annotate(

            "",

            xy=(3.25, perturb_y),

            xytext=(3.75, perturb_y),

            arrowprops=dict(

                arrowstyle="-|>",

                lw=2,

                color=self.BLACK,

            ),

        )

        # Right perturbation

        ax.annotate(

            "",

            xy=(4.75, perturb_y),

            xytext=(4.25, perturb_y),

            arrowprops=dict(

                arrowstyle="-|>",

                lw=2,

                color=self.BLACK,

            ),

        )

        ax.text(

            cx,

            perturb_y + 0.35,

            "External perturbation",

            ha="center",

            fontsize=10,

        )

        # ==================================================
        # PPO Closed-Loop Controller
        # ==================================================

        # Position of the RL loop
        box_x = 10.0
        controller_y = 5.0
        environment_y = 2.0
        box_w = 2.8
        box_h = 0.8

        # Controller
        self.draw_box(
            ax,
            box_x,
            controller_y,
            box_w,
            box_h,
            "PPO\nController",
        )

        # Environment
        self.draw_box(
            ax,
            box_x,
            environment_y,
            box_w,
            box_h,
            "Point-Mass\nEnvironment",
        )

        # --------------------------------------------------
        # Action (downward arrow)
        # --------------------------------------------------

        ax.annotate(
            "",
            xy=(box_x + box_w / 2, environment_y + box_h),
            xytext=(box_x + box_w / 2, controller_y),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.8,
                color=self.BLACK,
            ),
        )

        ax.text(
            box_x + box_w / 2 + 0.25,
            4.0,
            "Action",
            fontsize=10,
            va="center",
        )

        # --------------------------------------------------
        # Observation (left curved arrow)
        # --------------------------------------------------

        observation = FancyArrowPatch(
            (box_x, environment_y + box_h / 2),
            (box_x, controller_y + box_h / 2),
            connectionstyle="arc3,rad=0.45",
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.8,
            color=self.BLACK,
        )

        ax.add_patch(observation)

        ax.text(
            box_x - 1.35,
            4.0,
            "Observation",
            fontsize=10,
            ha="center",
        )

        # --------------------------------------------------
        # Reward (right curved arrow)
        # --------------------------------------------------

        reward = FancyArrowPatch(
            (box_x + box_w, controller_y + box_h / 2),
            (box_x + box_w, environment_y + box_h / 2),
            connectionstyle="arc3,rad=-0.45",
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.8,
            color=self.BLACK,
        )

        ax.add_patch(reward)

        ax.text(
            box_x + box_w + 1.0,
            4.0,
            "Reward",
            fontsize=10,
            ha="center",
        )

        # --------------------------------------------------
        # Connection from workspace to environment
        # --------------------------------------------------

        workspace_connection = FancyArrowPatch(
            (5.7, 1.8),
            (box_x, 2.4),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            linestyle="--",
            color="gray",
        )

        ax.add_patch(workspace_connection)

        ax.text(
            7.8,
            2.45,
            "Interaction",
            fontsize=9,
            color="gray",
        )

        # ==================================================
        # Figure Label (optional)
        # ==================================================

        ax.text(
            0.4,
            0.35,
            "Figure 1",
            fontsize=11,
            fontweight="bold",
            color=self.BLACK,
        )

        # ==================================================
        # Tight layout
        # ==================================================

        plt.tight_layout(pad=0.8)

        # ==================================================
        # Save figure
        # ==================================================

        filename = "Figure1_Adaptive_Reaching_Framework"

        for ext in ["png", "pdf", "svg"]:

            outfile = self.output / f"{filename}.{ext}"

            plt.savefig(
                outfile,
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
            )

            print(f"Saved: {outfile}")

        plt.close(fig)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    schematic = ReachingSchematic()

    schematic.draw()