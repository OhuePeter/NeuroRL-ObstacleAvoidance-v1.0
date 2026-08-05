"""
==========================================================
Gymnasium Environment

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Main Gymnasium environment for NeuroRL
Obstacle Avoidance.

Version:
2.0
==========================================================
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.world import World
from src.environment.physics import PhysicsEngine
from src.environment.reward import RewardFunction
from src.environment.observation import ObservationBuilder
from src.utils.logger import ExperimentLogger
from src.perturbations.perturbation import Perturbation
from src.evaluation.biological_variability import BiologicalVariability


class NeuroRLEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, condition="P0", biological_variability=False):

        super().__init__()

        self.condition = condition


        self.biological_variability = biological_variability
        if self.biological_variability:
            self.variability = BiologicalVariability()
        else:
            self.variability = None

        self.world = World()

        self.physics = PhysicsEngine(
            self.world.width,
            self.world.height
        )

        self.reward_function = RewardFunction()

        self.observation_builder = ObservationBuilder()

        self.logger = ExperimentLogger()

        self.perturbation = Perturbation(
            condition=condition,
            variability=self.variability
        )

        self.dt = 0.05

        self.max_steps = 400

        self.current_step = 0

        self.previous_goal_distance = None
        self.route_signal = 0.0
        self.route_waypoint_complete = True

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.world.reset(
            desired_route=options.get("desired_route")
            if options is not None
            else None
        )

        self.route_signal = {
            "left": -1.0,
            "right": 1.0,
        }.get(self.world.desired_route, 0.0)
        self.route_waypoint_complete = self.world.desired_route == "either"
        
        if self.biological_variability:
            x, y = self.variability.random_start(
                self.world.agent.x,
                self.world.agent.y
            )
            self.world.agent.x = x
            self.world.agent.y = y

            self.world.agent.start_x = x
            self.world.agent.start_y = y

        self.current_step = 0        

        self.previous_goal_distance = self.physics.distance(
            self.world.agent.position,
            self.world.goal.position
        )

        navigation_target = self._navigation_target()

        observation = self.observation_builder.build(
            self.world,
            target_position=navigation_target
        )

        if self.biological_variability:
            observation = self.variability.observation_noise(
                observation
            )

        info = {
            "goal_distance": self.previous_goal_distance,
            "goal_reached": False,
            "collision": False,
            "step": 0,
            "desired_route": self.world.desired_route,
            "route_signal": self.route_signal,
            "start_x": self.world.agent.start_x,
            "start_y": self.world.agent.start_y,
            "target_x": navigation_target[0],
            "target_y": navigation_target[1],
            "waypoint_active": not self.route_waypoint_complete,
        }

        return observation, info

    def step(self, action):

        self.current_step += 1
        previous_position = self.world.agent.position

        if self.biological_variability:
            action = self.variability.action_noise(action)

        # Action now represents acceleration
        force = self.perturbation.get_force(
            self.current_step
        )

        self.physics.update(
            self.world.agent,
            action,
            self.dt,
            external_force=force
        )

        if not self.route_waypoint_complete:
            self.route_waypoint_complete = self.world.waypoint_reached(
                self.world.agent.x,
                self.world.agent.y
            )

        navigation_target = self._navigation_target()

        previous_navigation_distance = self.physics.distance(
            previous_position,
            navigation_target
        )

        current_navigation_distance = self.physics.distance(
            self.world.agent.position,
            navigation_target
        )

        current_goal_distance = self.physics.distance(
            self.world.agent.position,
            self.world.goal.position
        )

        obstacle_distances = [

            self.physics.distance(
                self.world.agent.position,
                obstacle.position
            )

            for obstacle in self.world.obstacles
        ]

        minimum_obstacle_distance = min(obstacle_distances)

        collision = self.physics.collision(
            self.world.agent,
            self.world.obstacles
        )

        goal_reached = self.physics.goal_reached(
            self.world.agent,
            self.world.goal
        )

        reward = self.reward_function.compute_total_reward(

            previous_goal_distance=previous_navigation_distance,

            current_goal_distance=current_navigation_distance,

            minimum_obstacle_distance=minimum_obstacle_distance,

            goal_reached=goal_reached,

            collision=collision,

            ax=self.world.agent.ax,

            ay=self.world.agent.ay,

            agent_x=self.world.agent.x,

            agent_y=self.world.agent.y,

            corridor_mid_x=self.world.midline_x,

            corridor_mid_y_min=self.world.route_cfg.get(
                "corridor_mid_y_min"
            ),

            corridor_mid_y_max=self.world.route_cfg.get(
                "corridor_mid_y_max"
            ),

            desired_route_signal=self.route_signal,

            route_target_offset_x=self.world.route_cfg.get(
                "target_offset_x",
                0.0
            ),

            route_shaping_scale=self.world.route_cfg.get(
                "shaping_scale",
                0.0
            ),
        )

        self.logger.log(

            episode=1,

            trial=1,

            step=self.current_step,

            time=self.current_step * self.dt,

            seed=42,

            condition=self.condition,

            agent=self.world.agent,

            goal_distance=current_goal_distance,

            obstacle1_distance=obstacle_distances[0],

            obstacle2_distance=obstacle_distances[1],

            reward=reward["total"],

            success=goal_reached,

            collision=collision,

            route=self.world.desired_route

        )

        self.previous_goal_distance = current_navigation_distance

        observation = self.observation_builder.build(
            self.world,
            target_position=navigation_target
        )

        if self.biological_variability:
            observation = self.variability.observation_noise(
                observation
            )

        terminated = goal_reached or collision

        truncated = self.current_step >= self.max_steps

        info = {

            **reward,

            "step": self.current_step,

            "goal_distance": current_goal_distance,

            "goal_reached": goal_reached,

            "collision": collision,

            "agent_x": self.world.agent.x,

            "agent_y": self.world.agent.y,

            "vx": self.world.agent.vx,

            "vy": self.world.agent.vy,

            "heading": self.world.agent.heading,

            "desired_route": self.world.desired_route,

            "route_signal": self.route_signal,

            "start_x": self.world.agent.start_x,

            "start_y": self.world.agent.start_y,

            "target_x": navigation_target[0],

            "target_y": navigation_target[1],

            "waypoint_active": not self.route_waypoint_complete,

        }

        return (
            observation,
            reward["total"],
            terminated,
            truncated,
            info
        )

    def render(self):
        pass

    def close(self):
        pass

    def _navigation_target(self):
        if self.route_waypoint_complete:
            return self.world.goal.position

        return self.world.route_waypoint()