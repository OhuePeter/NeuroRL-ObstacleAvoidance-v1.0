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
    Rectangle,
    PathPatch,
)

from matplotlib.path import Path as MplPath


class ReachingSchematic:
    """
    Publication-quality schematic for the manuscript.
    """

    def __init__(self):

        self.outputs = [
            Path("paper/figures"),
            Path("experiments/version_1_0/figures"),
        ]

        for output in self.outputs:
            output.mkdir(
                parents=True,
                exist_ok=True,
            )

        # --------------------------------------------------
        # Colour palette
        # --------------------------------------------------

        self.BLACK = "#222222"
        self.GRAY = "#D9D9D9"
        self.LIGHT_GRAY = "#F2F2F2"
        self.SLATE = "#4C566A"
        self.BLUE = "#2F6DB2"
        self.TEAL = "#2A9D8F"
        self.ORANGE = "#E07A2F"
        self.RED = "#C8553D"
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

        plt.rcParams["font.family"] = "DejaVu Serif"

        fig, ax = plt.subplots(
            figsize=(14, 7)
        )

        ax.set_xlim(0, 15.2)
        ax.set_ylim(0, 8.2)

        ax.set_aspect("equal")
        ax.axis("off")

        # --------------------------------------------------
        # Panel backgrounds
        # --------------------------------------------------

        task_panel = FancyBboxPatch(
            (0.45, 0.55),
            5.5,
            6.9,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            linewidth=1.0,
            edgecolor="#CFCFCF",
            facecolor="#FBFBFB",
            zorder=0,
        )
        ax.add_patch(task_panel)

        loop_panel = FancyBboxPatch(
            (8.35, 1.35),
            5.9,
            4.9,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            linewidth=1.0,
            edgecolor="#CFCFCF",
            facecolor="#FBFBFB",
            zorder=0,
        )
        ax.add_patch(loop_panel)

        ax.text(0.75, 7.0, "Task setup", fontsize=14, fontweight="bold", color=self.SLATE)
        ax.text(8.65, 5.85, "Closed-loop control architecture", fontsize=14, fontweight="bold", color=self.SLATE)

        # ==================================================
        # Workspace
        # ==================================================

        # Workspace centre
        cx = 3.2

        # Key positions
        start = (cx, 1.0)
        goal = (cx, 6.6)
        obstacle = (cx, 3.8)

        # Current agent position (close to the start)
        agent = (cx, 1.35)

        # --------------------------------------------------
        # Start
        # --------------------------------------------------

        workspace = Rectangle(
            (1.2, 1.0),
            4.0,
            5.8,
            linewidth=1.2,
            edgecolor="#D7D7D7",
            facecolor=self.WHITE,
            zorder=0.5,
        )
        ax.add_patch(workspace)

        ax.text(1.35, 6.45, "2D reaching workspace", fontsize=10, color=self.SLATE)

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
                facecolor=self.WHITE,
                edgecolor=self.TEAL,
                linewidth=2.2,
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
                facecolor="#D8DEE9",
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

                edgecolor=self.BLUE,

                linewidth=2.0,

                linestyle="-",

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

                color=self.BLUE,

            ),

        )

        # --------------------------------------------------
        # Agent
        # --------------------------------------------------

        ax.plot(

            agent[0],

            agent[1],

            "o",

            color=self.BLUE,

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

            "Lateral perturbation",

            ha="center",

            fontsize=10,

            color=self.RED,

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
            "PPO policy\ncontroller",
        )

        # Environment
        self.draw_box(
            ax,
            box_x,
            environment_y,
            box_w,
            box_h,
            "Point-mass\nenvironment",
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
            "Motor action",
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
            color=self.TEAL,
        )

        ax.add_patch(observation)

        ax.text(
            box_x - 1.35,
            4.0,
            "State observation",
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
            color=self.ORANGE,
        )

        ax.add_patch(reward)

        ax.text(
            box_x + box_w + 1.0,
            4.0,
            "Reward signal",
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
            linewidth=1.6,
            linestyle="--",
            color="#8E8E8E",
        )

        ax.add_patch(workspace_connection)

        ax.text(
            7.8,
            2.45,
            "Simulated reach",
            fontsize=9,
            color="#7A7A7A",
        )

        ax.text(
            8.65,
            1.8,
            "Observation and reward are fed back each timestep\nwhile the policy updates the reach trajectory online.",
            fontsize=9.5,
            color=self.SLATE,
        )

        # ==================================================
        # Tight layout
        # ==================================================

        plt.tight_layout(pad=0.8)

        # ==================================================
        # Save figure
        # ==================================================

        filename = "figure1_schematic"

        for ext in ["png", "pdf", "svg"]:

            for output in self.outputs:

                outfile = output / f"{filename}.{ext}"

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