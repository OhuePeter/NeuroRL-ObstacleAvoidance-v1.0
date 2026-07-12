"""
==========================================================
Reward Function

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Computes the reward for the obstacle avoidance task.

Each reward component is computed independently,
making the reward function transparent, modular,
and easy to analyse.

Version:
1.0
==========================================================
"""

from dataclasses import dataclass


@dataclass
class RewardWeights:
    """
    Reward weights used during training.
    """

    goal: float = 100.0
    collision: float = -100.0
    progress: float = 1.0
    smoothness: float = 0.10
    time: float = -0.01
    clearance: float = 0.25


class RewardFunction:
    """
    Computes reward components.
    """

    def __init__(self):

        self.weights = RewardWeights()

    def goal_reward(self, goal_reached: bool):

        return self.weights.goal if goal_reached else 0.0

    def collision_penalty(self, collision: bool):

        return self.weights.collision if collision else 0.0

    def progress_reward(
        self,
        previous_distance: float,
        current_distance: float,
    ):
        """
        Positive reward when moving closer to the goal.
        """

        return self.weights.progress * (
            previous_distance - current_distance
        )

    def smoothness_reward(
        self,
        ax: float,
        ay: float,
    ):
        """
        Penalize large accelerations.
        """

        acceleration = (ax**2 + ay**2) ** 0.5

        return -self.weights.smoothness * acceleration

    def time_penalty(self):

        return self.weights.time

    def clearance_reward(
        self,
        minimum_distance: float,
    ):
        """
        Encourage maintaining a safe distance
        from obstacles.
        """

        return self.weights.clearance * minimum_distance

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

        breakdown = {

            "goal":
                self.goal_reward(goal_reached),

            "collision":
                self.collision_penalty(collision),

            "progress":
                self.progress_reward(
                    previous_goal_distance,
                    current_goal_distance
                ),

            "smoothness":
                self.smoothness_reward(ax, ay),

            "time":
                self.time_penalty(),

            "clearance":
                self.clearance_reward(
                    minimum_obstacle_distance
                )

        }

        breakdown["total"] = sum(breakdown.values())

        return breakdown