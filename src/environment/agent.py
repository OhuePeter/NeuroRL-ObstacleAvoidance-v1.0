"""
==========================================================
Agent

Defines the navigating agent.

Authors:
Peter Ohue
Gunnar Blohm
Version:
1.0
==========================================================
"""

import math


class Agent:
    """
    Mobile agent used in the obstacle avoidance environment.

    Parameters
    ----------
    x : float
        Initial x-coordinate.

    y : float
        Initial y-coordinate.

    radius : float
        Radius of the agent.
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 0.20
    ):

        # Initial position
        self.start_x = x
        self.start_y = y

        self.radius = radius

        self.reset()

    def reset(self):
        """
        Reset the agent to its initial state.
        """

        # Position
        self.x = self.start_x
        self.y = self.start_y

        # Velocity
        self.vx = 0.0
        self.vy = 0.0

        # Acceleration
        self.ax = 0.0
        self.ay = 0.0

        # Heading angle (radians)
        self.heading = 0.0

    def move(
        self,
        vx: float,
        vy: float,
        dt: float
    ):
        """
        Move the agent.

        Parameters
        ----------
        vx : float
            Desired x velocity.

        vy : float
            Desired y velocity.

        dt : float
            Simulation timestep.
        """

        # Compute acceleration
        self.ax = (vx - self.vx) / dt
        self.ay = (vy - self.vy) / dt

        # Update velocity
        self.vx = vx
        self.vy = vy

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Update heading
        if self.vx != 0 or self.vy != 0:
            self.heading = math.atan2(self.vy, self.vx)

    @property
    def position(self):
        """
        Current position.
        """
        return (self.x, self.y)

    @property
    def velocity(self):
        """
        Current velocity.
        """
        return (self.vx, self.vy)

    @property
    def acceleration(self):
        """
        Current acceleration.
        """
        return (self.ax, self.ay)