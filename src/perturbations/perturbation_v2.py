"""
==========================================================
Perturbation Module

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Implements external perturbations for Experiment 2.

Experiment 2 evaluates robustness of the frozen PPO policy
using stronger perturbation magnitudes while preserving the
same six perturbation levels as Experiment 1.

Conditions
----------
P0 : No perturbation
L1 : Small leftward
L2 : Medium leftward
L3 : Large leftward
R1 : Small rightward
R2 : Medium rightward
R3 : Large rightward

Version:
3.0
==========================================================
"""

import numpy as np


class Perturbation:
    """
    External perturbation generator for Experiment 2.
    """

    def __init__(self, condition="P0", variability=None):

        self.condition = condition
        self.variability = variability

        # --------------------------------------------------
        # Perturbation timing
        # --------------------------------------------------

        if variability is None:
            self.start_step = 25
        else:
            self.start_step = variability.perturbation_step()

        # Perturbation duration (simulation steps)
        self.duration = 15

        # --------------------------------------------------
        # Stronger perturbation magnitudes
        # --------------------------------------------------

        self.forces = {

            "P0": (0.0, 0.0),

            "L1": (-0.60, 0.0),
            "L2": (-1.30, 0.0),
            "L3": (-2.20, 0.0),

            "R1": (0.60, 0.0),
            "R2": (1.30, 0.0),
            "R3": (2.20, 0.0),
        }

    def get_force(self, current_step):
        """
        Return perturbation force.

        The perturbation is applied for a short interval
        early in the reaching movement.
        """

        if (
            self.start_step
            <= current_step
            < self.start_step + self.duration
        ):

            fx, fy = self.forces[self.condition]

            # Small trial-to-trial variability
            if self.variability is not None:

                jitter = np.random.normal(
                    loc=0.0,
                    scale=0.05
                )

                fx += jitter

            return (fx, fy)

        return (0.0, 0.0)