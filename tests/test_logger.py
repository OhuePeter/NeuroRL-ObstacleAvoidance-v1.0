"""
==========================================================
Logger Test

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from src.environment.world import World
from src.environment.physics import PhysicsEngine
from src.utils.logger import ExperimentLogger


def main():

    world = World()

    physics = PhysicsEngine(
        world.width,
        world.height
    )

    logger = ExperimentLogger()

    action = (0.0, 1.0)

    dt = 0.1

    for step in range(20):

        physics.update(
            world.agent,
            action,
            dt
        )

        goal_distance = physics.distance(
            world.agent.position,
            world.goal.position
        )

        obstacle1_distance = physics.distance(
            world.agent.position,
            world.obstacles[0].position
        )

        obstacle2_distance = physics.distance(
            world.agent.position,
            world.obstacles[1].position
        )

        logger.log(

            episode=1,

            step=step,

            time=step * dt,

            agent=world.agent,

            goal_distance=goal_distance,

            obstacle1_distance=obstacle1_distance,

            obstacle2_distance=obstacle2_distance

        )

    print()

    print(logger.dataframe().head())

    logger.save("experiment_001.csv")


if __name__ == "__main__":
    main()