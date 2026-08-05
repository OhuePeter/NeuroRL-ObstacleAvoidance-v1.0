"""
==========================================================
Biological Variability

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Introduces realistic trial-to-trial variability
during evaluation only.

Version:
2.0
==========================================================
"""

import numpy as np

from src.utils.config import ConfigLoader


class BiologicalVariability:

    def __init__(self):

        config = ConfigLoader.load_environment()
        evaluation_cfg = config.get("evaluation", {})

        self.rng = np.random.default_rng()
        self.start_position_jitter_std = evaluation_cfg.get(
            "start_position_jitter_std",
            0.05,
        )
        self.observation_noise_std = evaluation_cfg.get(
            "observation_noise_std",
            0.01,
        )
        self.action_noise_std = evaluation_cfg.get(
            "action_noise_std",
            0.02,
        )
        self.perturbation_step_min = evaluation_cfg.get(
            "perturbation_step_min",
            35,
        )
        self.perturbation_step_max = evaluation_cfg.get(
            "perturbation_step_max",
            46,
        )

    def random_start(self, x, y):
        """
        Small variability in initial position.
        """

        return (
            x + self.rng.normal(0.0, self.start_position_jitter_std),
            y + self.rng.normal(0.0, self.start_position_jitter_std),
        )

    def observation_noise(self, observation):
        """
        Small sensory noise.
        """

        noise = self.rng.normal(
            0.0,
            self.observation_noise_std,
            size=observation.shape
        )

        return observation + noise

    def action_noise(self, action):
        """
        Small motor noise.
        """

        noise = self.rng.normal(
            0.0,
            self.action_noise_std,
            size=action.shape
        )

        return action + noise

    def perturbation_step(self):
        """
        Random perturbation onset.
        """

        return int(
            self.rng.integers(
                self.perturbation_step_min,
                self.perturbation_step_max
            )
        )