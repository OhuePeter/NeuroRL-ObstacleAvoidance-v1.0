"""
==========================================================
Obstacle Object
==========================================================

Defines circular obstacles.

Authors:
- Peter Ohue
- Gunnar Blohm
"""

from dataclasses import dataclass


@dataclass
class Obstacle:
    """
    Circular obstacle.

    Attributes
    ----------
    x : float
        X position.

    y : float
        Y position.

    radius : float
        Radius of obstacle.
    """

    x: float
    y: float
    radius: float = 0.50

    @property
    def position(self):
        """Return obstacle position."""
        return (self.x, self.y)