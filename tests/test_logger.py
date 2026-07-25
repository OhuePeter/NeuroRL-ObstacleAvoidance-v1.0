from src.environment.world import World
from src.environment.physics import PhysicsEngine
from src.utils.logger import ExperimentLogger


def test_logger_records_rows_and_columns():
    world = World()
    physics = PhysicsEngine(
        world.width,
        world.height
    )
    logger = ExperimentLogger()
    dt = 0.1
    action = (0.0, 1.0)

    for step in range(5):
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

            trial=1,

            step=step,

            time=step * dt,

            seed=42,

            condition="P0",

            agent=world.agent,

            goal_distance=goal_distance,

            obstacle1_distance=obstacle1_distance,

            obstacle2_distance=obstacle2_distance,

            reward=0.0,

            success=False,

            collision=False,

            route="Centre"

        )

    df = logger.dataframe()
    assert len(df) == 5
    assert "x" in df.columns
    assert "goal_distance" in df.columns
    assert "success" in df.columns


def test_logger_save_writes_csv(tmp_path):
    world = World()
    logger = ExperimentLogger()

    logger.log(
        episode=1,
        trial=1,
        step=1,
        time=0.1,
        seed=42,
        condition="P0",
        agent=world.agent,
        goal_distance=1.0,
        obstacle1_distance=2.0,
        obstacle2_distance=2.0,
        reward=0.0,
        success=False,
        collision=False,
        route="Centre"
    )

    out_file = tmp_path / "behaviour.csv"
    logger.save(out_file)
    assert out_file.exists()