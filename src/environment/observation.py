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

    def build(self, world, target_position=None):

        route_signal = {
            "left": -1.0,
            "right": 1.0,
        }.get(world.desired_route, 0.0)

        agent = world.agent
        goal_x, goal_y = target_position if target_position is not None else world.goal.position

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

            goal_x - agent.x,
            goal_y - agent.y,

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
                goal_x - agent.x,
                goal_y - agent.y
            ]),

            # -----------------------------
            # Heading
            # -----------------------------

            agent.heading,

            # -----------------------------
            # Desired Route Cue
            # -----------------------------

            route_signal

        ], dtype=np.float32)

        return observation