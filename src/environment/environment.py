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


class NeuroRLEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self):

        super().__init__()

        self.world = World()

        self.physics = PhysicsEngine(
            self.world.width,
            self.world.height
        )

        self.reward_function = RewardFunction()

        self.observation_builder = ObservationBuilder()

        self.logger = ExperimentLogger()

        self.dt = 0.05

        self.max_steps = 400

        self.current_step = 0

        self.previous_goal_distance = None

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
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

        self.world.reset()

        self.current_step = 0

        self.previous_goal_distance = self.physics.distance(
            self.world.agent.position,
            self.world.goal.position
        )

        observation = self.observation_builder.build(self.world)

        info = {
            "goal_distance": self.previous_goal_distance,
            "goal_reached": False,
            "collision": False,
            "step": 0,
        }

        return observation, info

    def step(self, action):

        self.current_step += 1

        # Action now represents acceleration
        self.physics.update(
            self.world.agent,
            action,
            self.dt
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

            previous_goal_distance=self.previous_goal_distance,

            current_goal_distance=current_goal_distance,

            minimum_obstacle_distance=minimum_obstacle_distance,

            goal_reached=goal_reached,

            collision=collision,

            ax=self.world.agent.ax,

            ay=self.world.agent.ay,
        )

        self.logger.log(

            episode=1,

            trial=1,

            step=self.current_step,

            time=self.current_step * self.dt,

            seed=42,

            condition="P0",

            agent=self.world.agent,

            goal_distance=current_goal_distance,

            obstacle1_distance=obstacle_distances[0],

            obstacle2_distance=obstacle_distances[1],

            reward=reward["total"],

            success=goal_reached,

            collision=collision,

            route="Unknown"

        )

        self.previous_goal_distance = current_goal_distance

        observation = self.observation_builder.build(self.world)

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