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


class BiologicalVariability:

    def __init__(self):

        self.rng = np.random.default_rng()

    def random_start(self, x, y):
        """
        Small variability in initial position.
        """

        return (
            x + self.rng.normal(0.0, 0.05),
            y + self.rng.normal(0.0, 0.05),
        )

    def observation_noise(self, observation):
        """
        Small sensory noise.
        """

        noise = self.rng.normal(
            0.0,
            0.01,
            size=observation.shape
        )

        return observation + noise

    def action_noise(self, action):
        """
        Small motor noise.
        """

        noise = self.rng.normal(
            0.0,
            0.02,
            size=action.shape
        )

        return action + noise

    def perturbation_step(self):
        """
        Random perturbation onset.
        """

        return int(
            self.rng.integers(
                35,
                46
            )
        )