"""
==========================================================
Physics Engine

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Physics engine for the obstacle avoidance environment.

Responsibilities
----------------
- Integrate agent dynamics
- Keep agent inside world
- Compute distances
- Detect collisions
- Detect goal reaching

Version:
2.0
==========================================================
"""

import math
from typing import Tuple

from src.environment.agent import Agent
from src.environment.goal import Goal


class PhysicsEngine:
    """
    Physics engine for the obstacle avoidance environment.
    """

    def __init__(
        self,
        world_width: float,
        world_height: float
    ):

        self.world_width = world_width
        self.world_height = world_height

    def update(
        self,
        agent: Agent,
        action: Tuple[float, float],
        dt: float,
    ):
        """
        Update the agent using acceleration commands.
        """

        ax = float(action[0])
        ay = float(action[1])

        agent.move(ax, ay, dt)

        self._keep_inside_world(agent)

    def _keep_inside_world(self, agent: Agent):

        if agent.x < 0.0:
            agent.x = 0.0
            agent.vx = 0.0

        elif agent.x > self.world_width:
            agent.x = self.world_width
            agent.vx = 0.0

        if agent.y < 0.0:
            agent.y = 0.0
            agent.vy = 0.0

        elif agent.y > self.world_height:
            agent.y = self.world_height
            agent.vy = 0.0

    @staticmethod
    def distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def goal_reached(
        self,
        agent: Agent,
        goal: Goal
    ) -> bool:

        return (
            self.distance(
                agent.position,
                goal.position
            )
            <= goal.radius
        )

    def collision(
        self,
        agent: Agent,
        obstacles
    ) -> bool:

        for obstacle in obstacles:

            if (
                self.distance(
                    agent.position,
                    obstacle.position
                )
                <= agent.radius + obstacle.radius
            ):
                return True

        return False

    @staticmethod
    def speed(agent: Agent) -> float:

        return math.sqrt(
            agent.vx ** 2 +
            agent.vy ** 2
        )