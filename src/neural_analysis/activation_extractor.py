"""
==========================================================
Latent Activation Extractor

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Extracts the 256-dimensional latent policy
representation from a trained PPO controller
during evaluation.

The latent activations are saved for every
timestep of every episode.

Version:
1.0
==========================================================
"""

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from src.environment.environment import NeuroRLEnvironment


class ActivationExtractor:
    """
    Extract latent policy activations from PPO.
    """

    def __init__(self, model_path):

        self.model = PPO.load(model_path)

        self.device = self.model.device

    def _latent_policy(self, observation):
        """
        Compute the latent policy representation.
        """

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            features = self.model.policy.extract_features(
                observation
            )

            latent_pi, _ = self.model.policy.mlp_extractor(
                features
            )

        return (
            latent_pi
            .cpu()
            .numpy()
            .squeeze()
        )

    def extract_condition(
        self,
        condition,
        episodes=20
    ):
        """
        Extract latent activations for one condition.
        """

        output_dir = Path(
            "experiments/version_1_0/results"
        ) / "neural_activity" / condition

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        env = NeuroRLEnvironment(
            condition=condition,
            biological_variability=True
        )

        print()
        print("=" * 60)
        print(f"Condition: {condition}")
        print("=" * 60)

        for episode in range(episodes):

            observation, _ = env.reset()

            terminated = False
            truncated = False

            activations = []

            while not (terminated or truncated):

                latent = self._latent_policy(
                    observation
                )

                activations.append(latent)

                action, _ = self.model.predict(
                    observation,
                    deterministic=True
                )

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

            activations = np.asarray(
                activations,
                dtype=np.float32
            )

            filename = (
                output_dir /
                f"episode_{episode:03d}.npy"
            )

            np.save(
                filename,
                activations
            )

            print(
                f"Episode {episode:02d} "
                f"saved "
                f"{activations.shape}"
            )

        print()
        print(
            f"Finished condition {condition}"
        )

    def extract_all(
        self,
        episodes=20
    ):
        """
        Extract latent activity for all conditions.
        """

        conditions = [
            "P0",
            "L1",
            "L2",
            "L3",
            "R1",
            "R2",
            "R3",
        ]

        for condition in conditions:

            self.extract_condition(
                condition,
                episodes
            )

        print()
        print("=" * 60)
        print("LATENT EXTRACTION COMPLETE")
        print("=" * 60)