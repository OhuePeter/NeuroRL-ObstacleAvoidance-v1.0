"""
==========================================================
Agent
==========================================================

Defines the navigating agent.

Authors:
- Peter Ohue
- Gunnar Blohm
"""

import math


class Agent:
    """
    Mobile agent.

    Parameters
    ----------
    x : float
        Initial x position.

    y : float
        Initial y position.
    """

    def __init__(self,
                 x: float,
                 y: float,
                 radius: float = 0.20):

        self.start_x = x
        self.start_y = y

        self.radius = radius

        self.reset()

    def reset(self):
        """Reset agent."""

        self.x = self.start_x
        self.y = self.start_y

        self.vx = 0.0
        self.vy = 0.0

        self.heading = 0.0

    def move(self,
             vx: float,
             vy: float,
             dt: float):

        self.vx = vx
        self.vy = vy

        self.x += vx * dt
        self.y += vy * dt

        if vx != 0 or vy != 0:
            self.heading = math.atan2(vy, vx)

    @property
    def position(self):
        return (self.x, self.y)

    @property
    def velocity(self):
        return (self.vx, self.vy)