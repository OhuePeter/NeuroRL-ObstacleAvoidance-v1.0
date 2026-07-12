"""
==========================================================
Policy Evaluator

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Evaluates a trained PPO policy and saves
episode trajectories and behavioural metrics.

Version:
1.3
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
    Evaluate a trained PPO policy.
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

        successes = 0
        collisions = 0

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

                speed = self.env.physics.speed(agent)

                kinematics.append([
                    info["step"],
                    agent.x,
                    agent.y,
                    agent.vx,
                    agent.vy,
                    speed,
                    agent.ax,
                    agent.ay,
                    agent.heading,
                    info["goal_distance"],
                    reward
                ])

            trajectory = np.asarray(trajectory)

            pd.DataFrame(
                trajectory,
                columns=[
                    "x",
                    "y"
                ]
            ).to_csv(
                output_dir / f"trajectory_{episode:03d}.csv",
                index=False
            )

            kinematics_df = pd.DataFrame(
                kinematics,
                columns=[
                    "step",
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "speed",
                    "ax",
                    "ay",
                    "heading",
                    "goal_distance",
                    "reward"
                ]
            )

            kinematics_df.to_csv(
                output_dir / f"kinematics_{episode:03d}.csv",
                index=False
            )

            success = info["goal_reached"]
            collision = info["collision"]

            if success:
                successes += 1

            if collision:
                collisions += 1

            summary.append({

                "episode": episode,

                "reward": total_reward,

                "success": success,

                "collision": collision,

                "steps": len(trajectory),

                "path_length": BehaviourMetrics.path_length(
                    trajectory
                ),

                "mean_speed": BehaviourMetrics.mean_speed(
                    kinematics_df["speed"]
                ),

                "max_speed": BehaviourMetrics.max_speed(
                    kinematics_df["speed"]
                )

            })

            print("\n" + "=" * 60)
            print(f"EPISODE {episode:02d}")
            print("=" * 60)

            print(
                f"Start Position : "
                f"({agent.start_x:.3f}, {agent.start_y:.3f})"
            )

            print(
                f"Final Position : "
                f"({agent.x:.3f}, {agent.y:.3f})"
            )

            print(
                f"Goal Position  : "
                f"({self.env.world.goal.x:.3f}, {self.env.world.goal.y:.3f})"
            )

            print(
                f"Goal Distance  : "
                f"{info['goal_distance']:.3f}"
            )

            print(
                f"Steps          : "
                f"{len(trajectory)}"
            )

            print(
                f"Total Reward   : "
                f"{total_reward:.3f}"
            )

            print(
                f"Success        : "
                f"{success}"
            )

            print(
                f"Collision      : "
                f"{collision}"
            )

            print(
                f"Final Velocity : "
                f"({agent.vx:.3f}, {agent.vy:.3f})"
            )

            print(
                f"Heading        : "
                f"{agent.heading:.3f} rad"
            )

        summary_df = pd.DataFrame(summary)

        summary_df.to_csv(
            output_dir / "summary.csv",
            index=False
        )

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(summary_df)

        print("\nOverall Statistics")
        print("------------------------------")
        print(f"Episodes        : {episodes}")
        print(f"Successes       : {successes}")
        print(f"Success Rate    : {100 * successes / episodes:.1f}%")
        print(f"Collisions      : {collisions}")
        print(f"Collision Rate  : {100 * collisions / episodes:.1f}%")
        print(f"Average Reward  : {summary_df['reward'].mean():.2f}")
        print(f"Average Steps   : {summary_df['steps'].mean():.1f}")

        print(f"\nResults saved to:\n{output_dir}")