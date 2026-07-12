"""
==========================================================
Agent

Defines the navigating agent.

Authors:
Peter Ohue
Gunnar Blohm

Version:
2.0
==========================================================
"""

import math


class Agent:
    """
    Mobile agent controlled by acceleration.
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 0.20,
        max_speed: float = 1.0,
    ):

        self.start_x = x
        self.start_y = y

        self.radius = radius
        self.max_speed = max_speed

        self.reset()

    def reset(self):

        self.x = self.start_x
        self.y = self.start_y

        self.vx = 0.0
        self.vy = 0.0

        self.ax = 0.0
        self.ay = 0.0

        self.heading = math.pi / 2

    def move(
        self,
        ax: float,
        ay: float,
        dt: float
    ):
        """
        Actions represent accelerations.
        """

        # Save commanded acceleration
        self.ax = float(ax)
        self.ay = float(ay)

        # Integrate velocity
        self.vx += self.ax * dt
        self.vy += self.ay * dt

        # Limit speed
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

        if speed > self.max_speed:

            scale = self.max_speed / speed

            self.vx *= scale
            self.vy *= scale

        # Integrate position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Update heading
        if speed > 1e-6:
            self.heading = math.atan2(
                self.vy,
                self.vx
            )

    @property
    def position(self):
        return (self.x, self.y)

    @property
    def velocity(self):
        return (self.vx, self.vy)

    @property
    def acceleration(self):
        return (self.ax, self.ay)