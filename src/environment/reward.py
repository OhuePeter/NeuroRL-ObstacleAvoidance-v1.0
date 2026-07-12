"""
==========================================================
Reward Function

Authors:
Peter Ohue
Gunnar Blohm

Version:
2.0
==========================================================
"""

from dataclasses import dataclass


@dataclass
class RewardWeights:

    goal = 300.0
    collision = -300.0

    progress = 5.0

    smoothness = 0.02

    time = -0.01

    clearance = 0.10

    distance = 2.0


class RewardFunction:

    def __init__(self):

        self.weights = RewardWeights()

    def compute_total_reward(
        self,
        previous_goal_distance,
        current_goal_distance,
        minimum_obstacle_distance,
        goal_reached,
        collision,
        ax,
        ay,
    ):

        reward = {}

        reward["goal"] = (
            self.weights.goal
            if goal_reached
            else 0.0
        )

        reward["collision"] = (
            self.weights.collision
            if collision
            else 0.0
        )

        improvement = (
            previous_goal_distance -
            current_goal_distance
        )

        reward["progress"] = (
            self.weights.progress *
            improvement
        )

        reward["distance"] = (
            -self.weights.distance *
            current_goal_distance / 10.0
        )

        reward["clearance"] = (
            self.weights.clearance *
            min(minimum_obstacle_distance, 1.0)
        )

        acceleration = (
            ax ** 2 +
            ay ** 2
        ) ** 0.5

        reward["smoothness"] = (
            -self.weights.smoothness *
            acceleration
        )

        reward["time"] = self.weights.time

        reward["total"] = sum(
            reward.values()
        )

        return reward