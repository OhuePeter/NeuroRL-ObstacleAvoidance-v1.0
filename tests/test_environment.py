from src.environment.world import World


def main():

    world = World()

    print("=" * 40)

    print("World Created")

    print("=" * 40)

    print()

    print("Agent")

    print(world.agent.position)

    print()

    print("Goal")

    print(world.goal.position)

    print()

    print("Obstacles")

    for obstacle in world.obstacles:
        print(obstacle.position)

    print()

    print("Environment initialized successfully.")


if __name__ == "__main__":
    main()