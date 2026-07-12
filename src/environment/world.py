"""
==========================================================
World

Creates the simulation world using configuration files.

Author:
Peter Ohue
==========================================================
"""

from src.environment.goal import Goal
from src.environment.obstacle import Obstacle
from src.environment.agent import Agent

from src.utils.config import ConfigLoader


class World:
    """
    Simulation world.

    Loads all parameters from
    configs/environment.yaml.
    """

    def __init__(self):

        config = ConfigLoader.load_environment()

        env_cfg = config["environment"]
        agent_cfg = config["agent"]
        goal_cfg = config["goal"]
        obstacle_cfg = config["obstacles"]

        self.width = env_cfg["width"]
        self.height = env_cfg["height"]

        self.agent = Agent(
            x=self.width / 2,
            y=1.0,
            radius=agent_cfg["radius"]
        )

        self.goal = Goal(
            x=self.width / 2,
            y=self.height - 1,
            radius=goal_cfg["radius"]
        )

        self.obstacles = []

        for position in obstacle_cfg["positions"]:

            self.obstacles.append(

                Obstacle(
                    x=position[0],
                    y=position[1],
                    radius=obstacle_cfg["radius"]
                )

            )

    def reset(self):
        """Reset the world."""

        self.agent.reset()