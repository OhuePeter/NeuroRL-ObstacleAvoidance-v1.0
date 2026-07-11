"""
==========================================================
World
==========================================================

Creates the simulation world.

Authors:
- Peter Ohue
- Gunnar Blohm
"""

from .goal import Goal
from .obstacle import Obstacle
from .agent import Agent


class World:
    """
    Simulation world.

    Stores

    - boundaries
    - agent
    - goal
    - obstacles
    """

    def __init__(self):

        self.width = 10.0
        self.height = 10.0

        self.agent = Agent(
            x=5.0,
            y=1.0
        )

        self.goal = Goal(
            x=5.0,
            y=9.0
        )

        self.obstacles = [

            Obstacle(
                x=4.25,
                y=5.0,
                radius=0.50
            ),

            Obstacle(
                x=5.75,
                y=5.0,
                radius=0.50
            )

        ]

    def reset(self):
        """Reset environment."""

        self.agent.reset()