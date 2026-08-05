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
        training_cfg = config.get("training", {})

        self.width = env_cfg["width"]
        self.height = env_cfg["height"]
        self.midline_x = self.width / 2
        self.route_cfg = training_cfg.get(
            "route_balancing",
            {},
        )
        self._route_reset_count = 0
        self.desired_route = "right"

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

    def _next_route(self, desired_route=None):
        if desired_route in {"left", "right", "either"}:
            return desired_route

        if not self.route_cfg.get("enabled", False):
            return "either"

        if self.route_cfg.get("mode", "alternating") == "alternating":
            route = "left" if self._route_reset_count % 2 == 0 else "right"
            self._route_reset_count += 1
            return route

        return "either"

    def route_waypoint(self):
        if self.desired_route not in {"left", "right"}:
            return self.goal.position

        x_offset = self.route_cfg.get(
            "target_offset_x",
            1.8,
        )
        waypoint_y = self.route_cfg.get(
            "waypoint_y",
            5.0,
        )
        direction = -1.0 if self.desired_route == "left" else 1.0

        return (
            self.midline_x + direction * x_offset,
            waypoint_y,
        )

    def waypoint_reached(self, agent_x, agent_y):
        if self.desired_route not in {"left", "right"}:
            return True

        waypoint_x, waypoint_y = self.route_waypoint()
        tolerance = self.route_cfg.get(
            "waypoint_tolerance",
            0.45,
        )

        return (
            abs(agent_x - waypoint_x) <= tolerance
            and agent_y >= waypoint_y - tolerance
        )

    def reset(self, desired_route=None):
        """Reset the world."""

        self.agent.reset()
        self.desired_route = self._next_route(
            desired_route=desired_route
        )