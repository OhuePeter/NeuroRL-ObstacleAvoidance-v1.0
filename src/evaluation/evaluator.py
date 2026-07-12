"""
==========================================================
Policy Evaluator

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Evaluates a trained PPO policy in the nominal (P0)
environment and saves behavioural data.

Version:
1.1
==========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from src.environment.environment import NeuroRLEnvironment
from src.evaluation.metrics import BehaviourMetrics


class PolicyEvaluator:
    """
    Evaluate a trained policy.
    """

    def __init__(self, model_path):

        self.model = PPO.load(model_path)

        self.env = NeuroRLEnvironment()

    def evaluate(self, episodes=20):

        output_dir = Path(
            "experiments/version_1_0/results/evaluation_P0"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        summary = []

        for episode in range(episodes):

            observation, _ = self.env.reset()

            terminated = False
            truncated = False

            trajectory = []
            kinematics = []

            total_reward = 0.0

            while not (terminated or truncated):

                action, _ = self.model.predict(
                    observation,
                    deterministic=True
                )

                observation, reward, terminated, truncated, info = self.env.step(action)

                total_reward += reward

                agent = self.env.world.agent

                trajectory.append([
                    agent.x,
                    agent.y
                ])

                kinematics.append([
                    info["step"],
                    agent.vx,
                    agent.vy,
                    agent.ax,
                    agent.ay,
                    agent.heading,
                    info["goal_distance"],
                    reward
                ])

            trajectory = np.asarray(trajectory)

            trajectory_df = pd.DataFrame(

                trajectory,

                columns=[
                    "x",
                    "y"
                ]

            )

            trajectory_df.to_csv(

                output_dir /
                f"trajectory_{episode:03d}.csv",

                index=False

            )

            kinematics_df = pd.DataFrame(

                kinematics,

                columns=[

                    "step",

                    "vx",

                    "vy",

                    "ax",

                    "ay",

                    "heading",

                    "goal_distance",

                    "reward"

                ]

            )

            kinematics_df.to_csv(

                output_dir /
                f"kinematics_{episode:03d}.csv",

                index=False

            )

            speeds = np.sqrt(

                kinematics_df["vx"]**2 +
                kinematics_df["vy"]**2

            )

            summary.append({

                "episode": episode,

                "reward": total_reward,

                "success": info["goal_reached"],

                "collision": info["collision"],

                "steps": len(trajectory),

                "path_length":

                    BehaviourMetrics.path_length(
                        trajectory
                    ),

                "mean_speed":

                    BehaviourMetrics.mean_speed(
                        speeds
                    ),

                "max_speed":

                    BehaviourMetrics.max_speed(
                        speeds
                    )

            })

        summary_df = pd.DataFrame(summary)

        summary_df.to_csv(

            output_dir / "summary.csv",

            index=False

        )

        print()

        print("=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        print()

        print(summary_df)

        print()

        print(f"Results saved to:\n{output_dir}")