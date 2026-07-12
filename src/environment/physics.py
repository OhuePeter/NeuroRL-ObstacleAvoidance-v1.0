"""
==========================================================
Physics Engine

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Implements the physical dynamics of the navigation
environment.

Responsibilities
----------------
- Update position
- Update velocity
- Compute heading
- Compute acceleration
- Detect collisions
- Detect goal reaching
- Enforce environment boundaries

Version:
1.0
==========================================================
"""

import math
from typing import Tuple

from src.environment.agent import Agent
from src.environment.goal import Goal
from src.environment.obstacle import Obstacle


class PhysicsEngine:
    """
    Physics engine for the obstacle avoidance environment.
    """

    def __init__(self,
                 world_width: float,
                 world_height: float):

        self.world_width = world_width
        self.world_height = world_height

    def update(
        self,
        agent: Agent,
        action: Tuple[float, float],
        dt: float
    ):
        """
        Update the agent state.

        Parameters
        ----------
        agent : Agent

        action : tuple
            Desired velocity (vx, vy)

        dt : float
            Time step.
        """

        vx, vy = action

        agent.move(vx, vy, dt)

        self._keep_inside_world(agent)

    def _keep_inside_world(self, agent: Agent):
        """
        Keep agent inside the environment.
        """

        agent.x = max(0.0, min(agent.x, self.world_width))
        agent.y = max(0.0, min(agent.y, self.world_height))

    @staticmethod
    def distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """
        Euclidean distance.
        """

        return math.sqrt(

            (p1[0] - p2[0]) ** 2 +

            (p1[1] - p2[1]) ** 2

        )

    def goal_reached(
        self,
        agent: Agent,
        goal: Goal
    ) -> bool:
        """
        Check if the goal has been reached.
        """

        d = self.distance(

            agent.position,

            goal.position

        )

        return d <= goal.radius

    def collision(
        self,
        agent: Agent,
        obstacles
    ) -> bool:
        """
        Check obstacle collision.
        """

        for obstacle in obstacles:

            d = self.distance(

                agent.position,

                obstacle.position

            )

            if d <= agent.radius + obstacle.radius:

                return True

        return False

    @staticmethod
    def speed(agent: Agent) -> float:
        """
        Return speed magnitude.
        """

        return math.sqrt(

            agent.vx ** 2 +

            agent.vy ** 2

        )