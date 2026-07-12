"""
==========================================================
Perturbation Module

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Implements external perturbations applied during movement.

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
2.0
==========================================================
"""


class Perturbation:

    def __init__(self, condition="P0"):

        self.condition = condition

        self.start_step = 40
        self.duration = 10

        self.forces = {
            "P0": (0.0, 0.0),
            "L1": (-0.20, 0.0),
            "L2": (-0.40, 0.0),
            "L3": (-0.80, 0.0),
            "R1": (0.20, 0.0),
            "R2": (0.40, 0.0),
            "R3": (0.80, 0.0),
        }

    def get_force(self, current_step):
        """Return the perturbation force for the current step."""

        if self.start_step <= current_step < self.start_step + self.duration:
            return self.forces[self.condition]

        return (0.0, 0.0)