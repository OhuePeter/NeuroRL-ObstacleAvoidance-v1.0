from src.environment.world import World
from src.environment.physics import PhysicsEngine


def test_physics_update_moves_agent_forward():
    world = World()
    physics = PhysicsEngine(
        world.width,
        world.height
    )
    start_y = world.agent.y

    action = (0.0, 1.0)
    for _ in range(10):
        physics.update(
            world.agent,
            action,
            dt=0.1
        )

    assert world.agent.y > start_y
    assert physics.speed(world.agent) <= world.agent.max_speed + 1e-9


def test_physics_distance_and_collision_helpers():
    world = World()
    physics = PhysicsEngine(world.width, world.height)

    d = physics.distance((0.0, 0.0), (3.0, 4.0))
    assert d == 5.0

    assert isinstance(
        physics.collision(world.agent, world.obstacles),
        bool,
    )

    assert isinstance(
        physics.goal_reached(world.agent, world.goal),
        bool,
    )