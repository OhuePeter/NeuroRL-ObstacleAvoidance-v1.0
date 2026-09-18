"""
==========================================================
Closed-Loop Attractor / Fixed-Point Analysis

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
The trained controller is a feedforward PPO policy (net_arch
[256, 256]), not a recurrent network, so its hidden layer has
no autonomous dynamics of its own: latent_pi = g(s_t) is just a
static function of the instantaneous observation. There is
nothing to find a "fixed point" of in the hidden layer alone.

What *is* autonomous is the closed loop formed by the policy and
the physics engine. At every step the agent observes its state,
the policy emits an acceleration, the physics engine integrates
it, and the new state is fed back in. With no perturbation force
active, this closed loop

    s_{t+1} = F(s_t) = physics_step(s_t, policy(s_t))

is a genuine autonomous dynamical system in the agent's physical
state (x, y, vx, vy), holding the goal and obstacle geometry
fixed. This module finds and characterises the fixed points and
local vector field of that closed loop, restricted to the lateral
(x, vx) subsystem at a given height y in the corridor, since the
forward (y) motion is close to a constant-speed translation and
the interesting, condition-dependent structure lives in the
lateral dimension.

This class never touches gym's reset()/step() machinery, since
that also updates counters, rewards and route bookkeeping we do
not need. Instead it builds a minimal probe world and constructs
observations directly, which lets us evaluate the policy at any
(x, y, vx, vy) we choose, including states that never occur in a
real rollout.

Version:
1.0
==========================================================
"""

import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from stable_baselines3 import PPO

from src.environment.world import World
from src.environment.observation import ObservationBuilder


class AttractorAnalysis:
    """
    Closed-loop fixed-point and vector-field analysis of the
    trained controller.
    """

    def __init__(self, model_path):

        self.model = PPO.load(model_path)
        self.device = self.model.device

        self.world = World()
        self.world.desired_route = "either"

        self.observation_builder = ObservationBuilder()

        self.dt = 0.05

        self.obstacle_left = self.world.obstacles[0]
        self.obstacle_right = self.world.obstacles[1]
        self.goal = self.world.goal
        self.midline_x = self.world.midline_x

    # ------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------

    def _observation(self, x, y, vx, vy, route="either"):
        """
        Build the 13-dimensional observation the policy actually
        sees for a hand-chosen physical state (x, y, vx, vy),
        bypassing gym reset/step.

        `route` fixes the discrete route cue in {"left", "right",
        "either"}. This is not a nuisance input: the policy was
        trained with an alternating left/right detour curriculum
        (config training.route_balancing), so route_signal is fed
        to the network exactly like the goal and obstacle offsets
        are. Changing it changes which fixed point the closed loop
        has (multistability), so it must be held fixed and reported
        alongside every attractor result, not left implicit.
        """

        agent = self.world.agent

        agent.x = float(x)
        agent.y = float(y)
        agent.vx = float(vx)
        agent.vy = float(vy)

        speed = math.sqrt(vx ** 2 + vy ** 2)

        if speed > 1e-6:
            agent.heading = math.atan2(vy, vx)

        self.world.desired_route = route

        target_position = (
            self.world.route_waypoint()
            if route in ("left", "right")
            else self.goal.position
        )

        return self.observation_builder.build(
            self.world,
            target_position=target_position,
        )

    # ------------------------------------------------------
    # Policy queries
    # ------------------------------------------------------

    def _action(self, observation):
        """Deterministic policy action for one observation."""

        action, _ = self.model.predict(
            observation,
            deterministic=True,
        )

        return action

    def latent(self, observation):
        """256-dimensional latent_pi activation for one observation."""

        obs_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            features = self.model.policy.extract_features(obs_tensor)
            latent_pi, _ = self.model.policy.mlp_extractor(features)

        return latent_pi.cpu().numpy().squeeze()

    # ------------------------------------------------------
    # Lateral vector field
    # ------------------------------------------------------

    def lateral_acceleration(self, x, y, vx, vy, route="either"):
        """
        ax at state (x, y, vx, vy): the lateral component of the
        closed-loop vector field, holding the perturbation off and
        the route cue fixed at `route`.
        """

        observation = self._observation(x, y, vx, vy, route=route)
        action = self._action(observation)

        return float(action[0])

    def lateral_vector_field(
        self, x_range, vx_range, y, vy_plateau, route="either"
    ):
        """
        Evaluate the reduced 2-D lateral vector field
        (dx/dt, dvx/dt) = (vx, ax) over a grid of (x, vx) at a
        fixed corridor height y, fixed forward speed vy, and fixed
        route cue.

        Returns
        -------
        X, VX : meshgrid coordinates
        DX, DVX : vector field components (already scaled by dt,
            i.e. the per-step displacement of the reduced state)
        """

        X, VX = np.meshgrid(x_range, vx_range)

        DX = VX * self.dt
        DVX = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):

                ax = self.lateral_acceleration(
                    X[i, j], y, VX[i, j], vy_plateau, route=route
                )

                DVX[i, j] = ax * self.dt

        return X, VX, DX, DVX

    # ------------------------------------------------------
    # Fixed points
    # ------------------------------------------------------

    def find_lateral_fixed_point(
        self,
        y,
        vy_plateau,
        route="either",
        x_bracket=(0.5, 9.5),
        n_scan=121,
    ):
        """
        Solve for x* such that ax(x*, y, vx=0, vy_plateau) = 0 under
        a fixed route cue.

        At vx = 0 the reduced map has a fixed point exactly where
        the commanded lateral acceleration vanishes, since then
        x_{t+1} = x_t and vx_{t+1} = vx_t = 0. We scan the bracket
        for a sign change in ax and refine it with Brent's method.
        Returns None if no sign change is found (no fixed point in
        the bracket).
        """

        xs = np.linspace(x_bracket[0], x_bracket[1], n_scan)
        accels = np.array([
            self.lateral_acceleration(x, y, 0.0, vy_plateau, route=route)
            for x in xs
        ])

        for i in range(len(xs) - 1):

            if accels[i] == 0.0:
                return xs[i]

            if accels[i] * accels[i + 1] < 0.0:

                return brentq(
                    lambda x: self.lateral_acceleration(
                        x, y, 0.0, vy_plateau, route=route
                    ),
                    xs[i],
                    xs[i + 1],
                    xtol=1e-5,
                )

        return None

    def fixed_point_curve(
        self, y_values, vy_plateau, route="either", x_bracket=(0.5, 9.5)
    ):
        """
        Fixed-point location x*(y) at each requested corridor
        height, for a fixed route cue. This is the controller's
        implicit "preferred path" under that route: the lateral
        attractor the closed loop relaxes back to after a
        perturbation.
        """

        curve = []

        for y in y_values:

            x_star = self.find_lateral_fixed_point(
                y, vy_plateau, route=route, x_bracket=x_bracket
            )

            if x_star is not None:
                curve.append((y, x_star))

        return np.array(curve)

    # ------------------------------------------------------
    # Local stability
    # ------------------------------------------------------

    def local_jacobian(self, x_star, y, vy_plateau, route="either", eps=1e-3):
        """
        Finite-difference Jacobian of the reduced one-step map
            (x, vx) -> (x + vx*dt, vx + ax(x, y, vx, vy)*dt)
        evaluated at the fixed point (x_star, 0), for a fixed route
        cue. Its eigenvalues describe how the lateral state relaxes
        back to the attractor after a perturbation: |eigenvalue| < 1
        means the fixed point is locally stable (an attractor); a
        nonzero imaginary part means the relaxation is oscillatory.
        """

        def step(x, vx):

            ax = self.lateral_acceleration(x, y, vx, vy_plateau, route=route)

            return np.array([
                x + vx * self.dt,
                vx + ax * self.dt,
            ])

        state0 = np.array([x_star, 0.0])

        jacobian = np.zeros((2, 2))

        for j in range(2):

            perturbed = state0.copy()
            perturbed[j] += eps

            jacobian[:, j] = (
                step(*perturbed) - step(*state0)
            ) / eps

        eigenvalues = np.linalg.eigvals(jacobian)

        return jacobian, eigenvalues

    @staticmethod
    def time_constant(eigenvalue, dt):
        """
        Approximate exponential relaxation time constant (in
        seconds) implied by one eigenvalue of the discrete-time
        Jacobian: |eigenvalue| = exp(-dt / tau).
        """

        magnitude = abs(eigenvalue)

        if magnitude <= 0.0 or magnitude >= 1.0:
            return math.inf

        return -dt / math.log(magnitude)
