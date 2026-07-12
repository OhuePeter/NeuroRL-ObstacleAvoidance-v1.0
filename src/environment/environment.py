"""
==========================================================
Gymnasium Environment

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Main Gymnasium environment for NeuroRL Obstacle Avoidance.

Version:
1.1
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
    """
    NeuroRL Obstacle Avoidance Environment.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Core Components
        # --------------------------------------------------

        self.world = World()

        self.physics = PhysicsEngine(
            self.world.width,
            self.world.height
        )

        self.reward_function = RewardFunction()

        self.logger = ExperimentLogger()

        self.observation_builder = ObservationBuilder()

        # --------------------------------------------------
        # Environment Parameters
        # --------------------------------------------------

        self.dt = 0.05

        # Reduced from 400 to improve training speed
        self.max_steps = 200

        self.current_step = 0

        self.previous_goal_distance = None

        # --------------------------------------------------
        # Observation Space
        # --------------------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

        # --------------------------------------------------
        # Action Space
        # --------------------------------------------------

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

        info = {}

        return observation, info

    def step(self, action):

        self.current_step += 1

        # --------------------------------------------------
        # Physics Update
        # --------------------------------------------------

        self.physics.update(
            self.world.agent,
            action,
            self.dt
        )

        # --------------------------------------------------
        # Distances
        # --------------------------------------------------

        current_goal_distance = self.physics.distance(
            self.world.agent.position,
            self.world.goal.position
        )

        obstacle1_distance = self.physics.distance(
            self.world.agent.position,
            self.world.obstacles[0].position
        )

        obstacle2_distance = self.physics.distance(
            self.world.agent.position,
            self.world.obstacles[1].position
        )

        # --------------------------------------------------
        # Status
        # --------------------------------------------------

        collision = self.physics.collision(
            self.world.agent,
            self.world.obstacles
        )

        goal_reached = self.physics.goal_reached(
            self.world.agent,
            self.world.goal
        )

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        reward = self.reward_function.compute_total_reward(

            previous_goal_distance=self.previous_goal_distance,

            current_goal_distance=current_goal_distance,

            minimum_obstacle_distance=min(
                obstacle1_distance,
                obstacle2_distance
            ),

            goal_reached=goal_reached,

            collision=collision,

            ax=self.world.agent.ax,

            ay=self.world.agent.ay

        )

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        self.logger.log(

            episode=1,

            trial=1,

            step=self.current_step,

            time=self.current_step * self.dt,

            seed=42,

            condition="P0",

            agent=self.world.agent,

            goal_distance=current_goal_distance,

            obstacle1_distance=obstacle1_distance,

            obstacle2_distance=obstacle2_distance,

            reward=reward["total"],

            success=goal_reached,

            collision=collision,

            route="Unknown"

        )

        self.previous_goal_distance = current_goal_distance

        observation = self.observation_builder.build(
            self.world
        )

        terminated = goal_reached or collision

        truncated = self.current_step >= self.max_steps

        # --------------------------------------------------
        # Information returned to PPO / Evaluator
        # --------------------------------------------------

        info = {

            # Episode status

            "goal_reached": goal_reached,
            "collision": collision,

            # Reward components

            "total_reward": reward["total"],
            "goal_reward": reward["goal"],
            "collision_penalty": reward["collision"],
            "progress_reward": reward["progress"],
            "smoothness_reward": reward["smoothness"],
            "time_penalty": reward["time"],
            "clearance_reward": reward["clearance"],

            # Distances

            "goal_distance": current_goal_distance,
            "obstacle1_distance": obstacle1_distance,
            "obstacle2_distance": obstacle2_distance,

            # Current step

            "step": self.current_step

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