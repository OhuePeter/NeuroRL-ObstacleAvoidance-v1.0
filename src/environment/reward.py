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


def _clamp(value, lower, upper):
    return max(lower, min(value, upper))


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
        agent_x=None,
        agent_y=None,
        corridor_mid_x=None,
        corridor_mid_y_min=None,
        corridor_mid_y_max=None,
        desired_route_signal=0.0,
        route_target_offset_x=0.0,
        route_shaping_scale=0.0,
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

        reward["route"] = 0.0

        if (
            desired_route_signal != 0.0
            and agent_x is not None
            and agent_y is not None
            and corridor_mid_x is not None
            and corridor_mid_y_min is not None
            and corridor_mid_y_max is not None
            and corridor_mid_y_min <= agent_y <= corridor_mid_y_max
            and route_target_offset_x > 0.0
            and route_shaping_scale > 0.0
        ):
            target_x = (
                corridor_mid_x +
                desired_route_signal * route_target_offset_x
            )
            normalized_error = abs(agent_x - target_x) / route_target_offset_x
            reward["route"] = route_shaping_scale * (
                1.0 - _clamp(normalized_error, 0.0, 1.0)
            )

        reward["total"] = sum(
            reward.values()
        )

        return reward