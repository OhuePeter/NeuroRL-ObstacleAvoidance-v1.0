"""
==========================================================
Gymnasium Environment

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Experiment 2 environment for robustness testing.

This environment is based on the Version 1 environment but
uses the stronger Experiment 2 perturbation model while
keeping the trained PPO controller frozen.

Version:
3.0
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

from src.perturbations.perturbation_v2 import Perturbation
from src.evaluation.biological_variability import BiologicalVariability


class NeuroRLEnvironmentV2(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        condition="P0",
        biological_variability=True,
    ):

        super().__init__()

        self.condition = condition
        self.biological_variability = biological_variability

        if biological_variability:
            self.variability = BiologicalVariability()
        else:
            self.variability = None

        # --------------------------------------------------
        # World
        # --------------------------------------------------

        self.world = World()

        self.physics = PhysicsEngine(
            self.world.width,
            self.world.height,
        )

        self.reward_function = RewardFunction()

        self.observation_builder = ObservationBuilder()

        self.logger = ExperimentLogger()

        # --------------------------------------------------
        # Experiment 2 perturbation model
        # --------------------------------------------------

        self.perturbation = Perturbation(
            condition=condition,
            variability=self.variability,
        )

        # --------------------------------------------------
        # Simulation parameters
        # --------------------------------------------------

        self.dt = 0.05

        self.max_steps = 400

        self.current_step = 0

        self.previous_goal_distance = None

        # --------------------------------------------------
        # Gym spaces
        # --------------------------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    # ======================================================
    # Reset
    # ======================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.world.reset()

        if self.biological_variability:

            x, y = self.variability.random_start(
                self.world.agent.x,
                self.world.agent.y,
            )

            self.world.agent.x = x
            self.world.agent.y = y

            self.world.agent.start_x = x
            self.world.agent.start_y = y

        self.current_step = 0

        self.previous_goal_distance = self.physics.distance(
            self.world.agent.position,
            self.world.goal.position,
        )

        observation = self.observation_builder.build(
            self.world
        )

        if self.biological_variability:

            observation = (
                self.variability.observation_noise(
                    observation
                )
            )

        info = {

            "goal_distance": self.previous_goal_distance,

            "goal_reached": False,

            "collision": False,

            "step": 0,

        }

        return observation, info
    
    # ======================================================
    # Step
    # ======================================================

    def step(self, action):

        self.current_step += 1

        # --------------------------------------------------
        # Biological motor noise
        # --------------------------------------------------

        if self.biological_variability:
            action = self.variability.action_noise(action)

        # --------------------------------------------------
        # External perturbation
        # --------------------------------------------------

        force = self.perturbation.get_force(
            self.current_step
        )

        # --------------------------------------------------
        # Physics update
        # --------------------------------------------------

        self.physics.update(
            self.world.agent,
            action,
            self.dt,
            external_force=force,
        )

        # --------------------------------------------------
        # Distances
        # --------------------------------------------------

        current_goal_distance = self.physics.distance(
            self.world.agent.position,
            self.world.goal.position,
        )

        obstacle_distances = [
            self.physics.distance(
                self.world.agent.position,
                obstacle.position,
            )
            for obstacle in self.world.obstacles
        ]

        minimum_obstacle_distance = min(obstacle_distances)

        # --------------------------------------------------
        # Events
        # --------------------------------------------------

        collision = self.physics.collision(
            self.world.agent,
            self.world.obstacles,
        )

        goal_reached = self.physics.goal_reached(
            self.world.agent,
            self.world.goal,
        )

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        reward = self.reward_function.compute_total_reward(
            previous_goal_distance=self.previous_goal_distance,
            current_goal_distance=current_goal_distance,
            minimum_obstacle_distance=minimum_obstacle_distance,
            goal_reached=goal_reached,
            collision=collision,
            ax=self.world.agent.ax,
            ay=self.world.agent.ay,
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
            condition=self.condition,
            agent=self.world.agent,
            goal_distance=current_goal_distance,
            obstacle1_distance=obstacle_distances[0],
            obstacle2_distance=obstacle_distances[1],
            reward=reward["total"],
            success=goal_reached,
            collision=collision,
            route="Unknown",
        )

        self.previous_goal_distance = current_goal_distance

        # --------------------------------------------------
        # Observation
        # --------------------------------------------------

        observation = self.observation_builder.build(
            self.world
        )

        if self.biological_variability:
            observation = self.variability.observation_noise(
                observation
            )

        # --------------------------------------------------
        # Episode termination
        # --------------------------------------------------

        terminated = goal_reached or collision

        truncated = (
            self.current_step >= self.max_steps
        )

        # --------------------------------------------------
        # Info dictionary
        # --------------------------------------------------

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

            "ax": self.world.agent.ax,

            "ay": self.world.agent.ay,

            "heading": self.world.agent.heading,

            "perturbation_force": force,

            "condition": self.condition,
        }

        return (
            observation,
            reward["total"],
            terminated,
            truncated,
            info,
        )
    
    # ======================================================
    # Render
    # ======================================================

    def render(self):
        """
        Placeholder renderer.

        Video recording for Experiment 2 will be implemented
        separately so that it does not interfere with the
        simulation or evaluation pipeline.
        """
        pass

    # ======================================================
    # Close
    # ======================================================

    def close(self):
        """
        Cleanly close the environment.
        """
        pass