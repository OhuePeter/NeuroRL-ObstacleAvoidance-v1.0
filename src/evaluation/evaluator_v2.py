"""
==========================================================
Policy Evaluator V2

Authors:
Peter Ohue
Gunnar Blohm

Experiment 2
Robustness Evaluation

Description
-----------
Evaluates a frozen PPO policy under stronger perturbations.

Features
--------
- Uses Experiment 1 checkpoint
- Supports P0, L1-L3, R1-R3
- Configurable number of episodes
- Saves trajectories
- Saves kinematics
- Saves behavioural summary
- Saves experiment metadata
- Designed for future video generation

==========================================================
"""

from pathlib import Path
import json
from datetime import datetime

from experiments.version_2_0.evaluation.config import N_EPISODES
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from src.environment.environment_v2 import NeuroRLEnvironmentV2
from src.evaluation.metrics import BehaviourMetrics

import torch

class PolicyEvaluatorV2:
    """
    Experiment 2 evaluator.

    Designed specifically for robustness experiments while
    leaving Experiment 1 untouched.
    """

    def __init__(self, model_path):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{self.model_path}"
            )

        print("=" * 60)
        print("Loading PPO checkpoint")
        print(self.model_path)
        print("=" * 60)

        self.model = PPO.load(str(self.model_path))

    def _save_episode_metadata(
        self,
        output_dir,
        episode,
        condition,
        success,
        collision,
        total_reward,
        steps,
        goal_distance,
        path_length,
        mean_speed,
        max_speed,
        heading_deviation,
        final_lateral_error,
    ):
        """
        Save metadata describing one evaluation episode.

        Parameters
        ----------
        output_dir : Path
            Directory where evaluation results are stored.
        """

        import json
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {

            # --------------------------------------------------
            # Episode information
            # --------------------------------------------------
            "episode": int(episode),
            "condition": str(condition),

            # --------------------------------------------------
            # Outcome
            # --------------------------------------------------
            "success": bool(success),
            "failure": not bool(success),
            "collision": bool(collision),

            # Failure type
            "failure_type": (
                "collision"
                if collision
                else (
                    "timeout"
                    if not success
                    else "none"
                )
            ),

            # --------------------------------------------------
            # Performance
            # --------------------------------------------------
            "reward": float(total_reward),
            "steps": int(steps),

            # --------------------------------------------------
            # Behavioural metrics
            # --------------------------------------------------
            "goal_distance": float(goal_distance),
            "path_length": float(path_length),
            "mean_speed": float(mean_speed),
            "maximum_speed": float(max_speed),
            "heading_deviation": float(heading_deviation),
            "final_lateral_error": float(final_lateral_error),
        }

        filename = output_dir / f"metadata_{episode:03d}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                indent=4,
            )
    def evaluate(
        self,
        episodes=N_EPISODES,
        condition="P0",
        output_root="experiments/version_2_0/results",
        save_video=False,
        biological_variability=True,
        save_metadata=True
    ):

        self.output_dir = (
            Path(output_root)
            / f"evaluation_{condition}"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.env = NeuroRLEnvironmentV2(
            condition=condition,
            biological_variability=True,
        )

        summary = []

        successes = 0

        collisions = 0

    
        """
        Evaluate one perturbation condition.

        Parameters
        ----------
        episodes : int
            Number of evaluation episodes.

        condition : str
            P0, L1-L3 or R1-R3

        output_root : str
            Root directory for Experiment 2 outputs.

        biological_variability : bool
            Enables action, observation and start variability.

        save_metadata : bool
            Save experiment metadata.

        save_video : bool
            Placeholder for future video generation.
        """

        output_dir = (
            Path(output_root)
            / f"evaluation_{condition}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.env = NeuroRLEnvironmentV2(
            condition=condition,
            biological_variability=biological_variability
        )

        summary = []

        successes = 0
        collisions = 0

        print()
        print("=" * 60)
        print(f"Evaluating {condition}")
        print("=" * 60)

        # ----------------------------------------------------------
        # Neural recordings
        # ----------------------------------------------------------

        # ======================================================
        # Evaluate all episodes
        # ======================================================

        for episode in range(episodes):

            policy_activations = []
            value_activations = []
            observations = []
            actions = []
            rewards = []

            observation, _ = self.env.reset()

            terminated = False
            truncated = False

            trajectory = []
            kinematics = []

            total_reward = 0.0

            while not (terminated or truncated):

                # ----------------------------------------------------------
                # Record observation
                # ----------------------------------------------------------

                obs_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                ).unsqueeze(0)

                with torch.no_grad():

                    features = self.model.policy.extract_features(obs_tensor)

                    latent_pi, latent_vf = (
                        self.model.policy.mlp_extractor(features)
                    )

                policy_activations.append(
                    latent_pi.cpu().numpy().squeeze()
                )

                value_activations.append(
                    latent_vf.cpu().numpy().squeeze()
                )

                observations.append(
                    observation.copy()
                )
                
                action, _ = self.model.predict(
                    observation,
                    deterministic=True,
                )
                actions.append(action.copy())

                observation, reward, terminated, truncated, info = (
                    self.env.step(action)
                )
                rewards.append(reward)

                total_reward += reward

                agent = self.env.world.agent

                trajectory.append([
                    agent.x,
                    agent.y,
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
                    reward,
                ])

            trajectory = np.asarray(trajectory)

            trajectory_df = pd.DataFrame(
                trajectory,
                columns=[
                    "x",
                    "y",
                ],
            )

            trajectory_df.to_csv(
                output_dir / f"trajectory_{episode:03d}.csv",
                index=False,
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
                    "reward",
                ],
            )

            kinematics_df.to_csv(
                output_dir / f"kinematics_{episode:03d}.csv",
                index=False,
            )
            # ----------------------------------------------------------
            # Save neural recordings
            # ----------------------------------------------------------

            neural_dir = output_dir / "neural"
            neural_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                neural_dir / f"policy_{episode:03d}.npy",
                np.asarray(policy_activations),
            )

            np.save(
                neural_dir / f"value_{episode:03d}.npy",
                np.asarray(value_activations),
            )

            np.save(
                neural_dir / f"observations_{episode:03d}.npy",
                np.asarray(observations),
            )

            np.save(
                neural_dir / f"actions_{episode:03d}.npy",
                np.asarray(actions),
            )

            np.save(
                neural_dir / f"rewards_{episode:03d}.npy",
                np.asarray(rewards),
            )

            success = info["goal_reached"]
            collision = info["collision"]

            if success:
                successes += 1

            if collision:
                collisions += 1

            # ==================================================
            # Behavioural Metrics
            # ==================================================

            summary.append({
                "episode": episode,
                "condition": condition,
                "outcome": (
                    "success"
                    if success
                    else "failure"
                ),
                "reward": total_reward,
                "success": success,
                "collision": collision,
                "steps": len(trajectory),
                # ----------------------------------------------
                # Existing behavioural metrics
                # ----------------------------------------------
                "path_length": BehaviourMetrics.path_length(
                    trajectory
                ),
                "mean_speed": BehaviourMetrics.mean_speed(
                    kinematics_df["speed"]
                ),

                "max_speed": BehaviourMetrics.max_speed(
                    kinematics_df["speed"]
                ),

                # ----------------------------------------------
                # Additional Experiment 2 metrics
                # ----------------------------------------------

                "peak_lateral_velocity": np.abs(
                    kinematics_df["vx"]
                ).max(),

                "max_heading_deviation": np.max(
                    np.abs(
                        kinematics_df["heading"]
                        - np.pi / 2
                    )
                ),

                "final_lateral_error": abs(
                    agent.x -
                    self.env.world.goal.x
                ),

                "final_goal_distance": info[
                    "goal_distance"
                ],
            })

            # ==================================================
            # Episode Summary
            # ==================================================

            print()
            print("=" * 60)
            print(f"Episode {episode + 1}/{episodes}")
            print("=" * 60)

            print(
                f"Condition           : {condition}"
            )

            print(
                f"Outcome             : "
                f"{'SUCCESS' if success else 'FAILURE'}"
            )

            print(
                f"Reward              : "
                f"{total_reward:.2f}"
            )

            print(
                f"Steps               : "
                f"{len(trajectory)}"
            )

            print(
                f"Goal Distance       : "
                f"{info['goal_distance']:.3f}"
            )

            print(
                f"Collision           : "
                f"{collision}"
            )

            print(
                f"Path Length         : "
                f"{summary[-1]['path_length']:.3f}"
            )

            print(
                f"Mean Speed          : "
                f"{summary[-1]['mean_speed']:.3f}"
            )

            print(
                f"Maximum Speed       : "
                f"{summary[-1]['max_speed']:.3f}"
            )

            print(
                f"Heading Deviation   : "
                f"{summary[-1]['max_heading_deviation']:.3f}"
            )

            print(
                f"Final Lateral Error : "
                f"{summary[-1]['final_lateral_error']:.3f}"
            )

            self._save_episode_metadata(
                output_dir=output_dir,
                episode=episode,
                condition=condition,
                success=success,
                collision=collision,
                total_reward=total_reward,
                steps=len(trajectory),
                goal_distance=info["goal_distance"],
                path_length=summary[-1]["path_length"],
                mean_speed=summary[-1]["mean_speed"],
                max_speed=summary[-1]["max_speed"],
                heading_deviation=summary[-1]["max_heading_deviation"],
                final_lateral_error=summary[-1]["final_lateral_error"],
            )

        # ======================================================
        # Save Behavioural Summary
        # ======================================================

        summary_df = pd.DataFrame(summary)

        summary_file = output_dir / "summary.csv"

        summary_df.to_csv(
            summary_file,
            index=False,
        )

        # ======================================================
        # Overall Statistics
        # ======================================================

        success_rate = (
            100.0 * successes / episodes
            if episodes > 0
            else 0.0
        )

        collision_rate = (
            100.0 * collisions / episodes
            if episodes > 0
            else 0.0
        )

        # ======================================================
        # Metadata
        # ======================================================

        if save_metadata:

            metadata = {
                "experiment": "Experiment 2",
                "condition": condition,
                "date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "policy_checkpoint": str(
                    self.model_path
                ),
                "episodes": episodes,
                "biological_variability": biological_variability,
                "successes": successes,
                "failures": episodes - successes,
                "collisions": collisions,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
                "average_reward": float(
                    summary_df["reward"].mean()
                ),
                "average_steps": float(
                    summary_df["steps"].mean()
                ),
                "average_path_length": float(
                    summary_df["path_length"].mean()
                ),
                "average_mean_speed": float(
                    summary_df["mean_speed"].mean()
                ),
                "average_max_speed": float(
                    summary_df["max_speed"].mean()
                ),
            }

            metadata_file = (
                output_dir / "metadata.json"
            )

            with open(
                metadata_file,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=4,
                )

        # ======================================================
        # Console Summary
        # ======================================================

        print()
        print("=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        print(
            f"Condition         : {condition}"
        )

        print(
            f"Episodes          : {episodes}"
        )

        print(
            f"Successes         : {successes}"
        )

        print(
            f"Failures          : "
            f"{episodes - successes}"
        )

        print(
            f"Success Rate      : "
            f"{success_rate:.1f}%"
        )

        print(
            f"Collision Rate    : "
            f"{collision_rate:.1f}%"
        )

        print(
            f"Average Reward    : "
            f"{summary_df['reward'].mean():.2f}"
        )

        print(
            f"Average Steps     : "
            f"{summary_df['steps'].mean():.1f}"
        )

        print(
            f"Average Path Len. : "
            f"{summary_df['path_length'].mean():.3f}"
        )

        print(
            f"Average Speed     : "
            f"{summary_df['mean_speed'].mean():.3f}"
        )

        print()
        print(
            f"Results saved to:\n{output_dir}"
        )

        # ======================================================
        # Future Video Export
        # ======================================================

        if save_video:

            print()
            print("=" * 60)
            print("VIDEO EXPORT")
            print("=" * 60)
            print(
                "Video generation is not yet implemented."
            )
            print(
                "Placeholder retained for future renderer."
            )

        # ======================================================
        # Close Environment
        # ======================================================

        self.env.close()

        # ======================================================
        # Return Results
        # ======================================================

        return {
            "condition": condition,
            "episodes": episodes,
            "successes": successes,
            "failures": episodes - successes,
            "collisions": collisions,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "summary": summary_df,
            "output_directory": output_dir,
        }


        # ----------------------------------------------------------
