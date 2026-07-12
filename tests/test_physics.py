"""
==========================================================
Physics Engine Test

Tests the movement of the agent.

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from src.environment.world import World
from src.environment.physics import PhysicsEngine


def main():

    world = World()

    physics = PhysicsEngine(
        world.width,
        world.height
    )

    print("=" * 60)
    print("INITIAL AGENT STATE")
    print("=" * 60)

    print(f"Position     : {world.agent.position}")
    print(f"Velocity     : {world.agent.velocity}")
    print(f"Acceleration : {world.agent.acceleration}")

    print()

    action = (0.0, 1.0)

    print("=" * 60)
    print("SIMULATION")
    print("=" * 60)

    for step in range(10):

        physics.update(
            world.agent,
            action,
            dt=0.1
        )

        print(
            f"Step {step+1:02d}"
            f" | Position {world.agent.position}"
            f" | Velocity {world.agent.velocity}"
            f" | Speed {physics.speed(world.agent):.2f}"
        )

    print()

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Collision   : {physics.collision(world.agent, world.obstacles)}")
    print(f"Goal Reached: {physics.goal_reached(world.agent, world.goal)}")

if __name__ == "__main__":
    main()