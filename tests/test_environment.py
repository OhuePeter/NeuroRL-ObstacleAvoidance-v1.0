"""
==========================================================
Environment Test

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

import numpy as np

from src.environment.environment import NeuroRLEnvironment


def main():

    env = NeuroRLEnvironment()

    observation, info = env.reset()

    print("=" * 60)
    print("INITIAL OBSERVATION")
    print("=" * 60)

    print(observation)

    print()

    for step in range(10):

        action = np.array(
            [0.0, 1.0],
            dtype=np.float32
        )

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1:02d}")

        print(f"Reward      : {reward:.3f}")

        print(
            f"Position    : ({observation[0]:.2f}, {observation[1]:.2f})"
        )

        print(
            f"Goal Distance : {observation[10]:.2f}"
        )

        print(
            f"Heading       : {observation[11]:.2f}"
        )

        print()

        if terminated or truncated:
            break

    print("=" * 60)
    print("Environment working successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()