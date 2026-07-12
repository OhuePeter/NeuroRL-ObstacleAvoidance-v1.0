"""
==========================================================
Observation Space

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Constructs the observation vector presented to the
reinforcement learning agent.

Version:
1.0
==========================================================
"""

import numpy as np


class ObservationBuilder:
    """
    Builds the observation vector.
    """

    def build(self, world):

        agent = world.agent
        goal = world.goal

        obstacle1 = world.obstacles[0]
        obstacle2 = world.obstacles[1]

        observation = np.array([

            # -----------------------------
            # Agent Position
            # -----------------------------

            agent.x,
            agent.y,

            # -----------------------------
            # Agent Velocity
            # -----------------------------

            agent.vx,
            agent.vy,

            # -----------------------------
            # Relative Goal Position
            # -----------------------------

            goal.x - agent.x,
            goal.y - agent.y,

            # -----------------------------
            # Relative Obstacle 1 Position
            # -----------------------------

            obstacle1.x - agent.x,
            obstacle1.y - agent.y,

            # -----------------------------
            # Relative Obstacle 2 Position
            # -----------------------------

            obstacle2.x - agent.x,
            obstacle2.y - agent.y,

            # -----------------------------
            # Distance to Goal
            # -----------------------------

            np.linalg.norm([
                goal.x - agent.x,
                goal.y - agent.y
            ]),

            # -----------------------------
            # Heading
            # -----------------------------

            agent.heading

        ], dtype=np.float32)

        return observation