"""
==========================================================
Goal Object
==========================================================

Defines the navigation goal.

Authors:
Peter Ohue
Gunnar Blohm
"""

from dataclasses import dataclass


@dataclass
class Goal:
    """
    Goal location in the environment.

    Attributes
    ----------
    x : float
        X coordinate.

    y : float
        Y coordinate.

    radius : float
        Goal radius.
    """

    x: float
    y: float
    radius: float = 0.35

    @property
    def position(self):
        """Return goal position."""
        return (self.x, self.y)
