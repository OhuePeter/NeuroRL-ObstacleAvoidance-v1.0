"""
==========================================================
Policy Evaluator

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from src.environment.environment import NeuroRLEnvironment
from src.evaluation.metrics import BehaviourMetrics


class PolicyEvaluator:

    def __init__(self, model_path):

        self.model = PPO.load(model_path)

        self.env = NeuroRLEnvironment()

    def evaluate(self, episodes=20):

        output = Path(
            "experiments/version_1_0/results/evaluation_P0"
        )

        output.mkdir(
            parents=True,
            exist_ok=True
        )

        summary = []

        for episode in range(episodes):

            observation, _ = self.env.reset()

            done = False

            trajectory = []

            speeds = []

            total_reward = 0.0

            collision = False

            success = False

            while not done:

                action, _ = self.model.predict(

                    observation,

                    deterministic=True

                )

                observation, reward, terminated, truncated, info = self.env.step(action)

                total_reward += reward

                trajectory.append([

                    self.env.world.agent.x,

                    self.env.world.agent.y

                ])

                speeds.append(

                    self.env.physics.speed(

                        self.env.world.agent

                    )

                )

                collision = info["collision"]

                success = info["goal"]

                done = terminated or truncated

            trajectory = np.array(trajectory)

            pd.DataFrame(

                trajectory,

                columns=["x", "y"]

            ).to_csv(

                output / f"trajectory_{episode:03d}.csv",

                index=False

            )

            summary.append({

                "episode": episode,

                "reward": total_reward,

                "success":

                    BehaviourMetrics.success(success),

                "collision":

                    BehaviourMetrics.collision(collision),

                "path_length":

                    BehaviourMetrics.path_length(trajectory),

                "mean_speed":

                    BehaviourMetrics.mean_speed(speeds),

                "max_speed":

                    BehaviourMetrics.max_speed(speeds),

                "steps":

                    len(trajectory)

            })

        pd.DataFrame(summary).to_csv(

            output / "summary.csv",

            index=False

        )

        print()

        print("=" * 60)

        print("Evaluation Complete")

        print("=" * 60)

        print(output)