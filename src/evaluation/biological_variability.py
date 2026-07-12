"""
==========================================================
Biological Variability Module

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Introduces biologically plausible variability
during evaluation only.

Training remains deterministic.

Features
--------
• Random perturbation onset
• Perturbation magnitude variability
• Randomized initial hand position
• Observation noise

Version:
1.0
==========================================================
"""

import numpy as np


class BiologicalVariability:

    def __init__(
        self,
        start_noise_std=0.05,
        observation_noise_std=0.01,
        perturbation_time=(35, 45),
        perturbation_force_std=0.03,
        random_seed=None
    ):

        self.rng = np.random.default_rng(random_seed)

        self.start_noise_std = start_noise_std
        self.observation_noise_std = observation_noise_std

        self.perturbation_min = perturbation_time[0]
        self.perturbation_max = perturbation_time[1]

        self.perturbation_force_std = perturbation_force_std

    def random_start(self, x, y):

        x += self.rng.normal(0.0, self.start_noise_std)
        y += self.rng.normal(0.0, self.start_noise_std)

        return x, y

    def observation_noise(self, observation):

        return (
            observation +
            self.rng.normal(
                0.0,
                self.observation_noise_std,
                observation.shape
            )
        )

    def perturbation_step(self):

        return self.rng.integers(
            self.perturbation_min,
            self.perturbation_max + 1
        )

    def perturbation_force(self, base_force):

        fx, fy = base_force

        fx += self.rng.normal(
            0.0,
            self.perturbation_force_std
        )

        return (fx, fy)