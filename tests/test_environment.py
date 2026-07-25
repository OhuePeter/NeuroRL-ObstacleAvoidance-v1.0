import numpy as np

from src.environment.environment import NeuroRLEnvironment


def test_environment_reset_shape_and_info_keys():
    env = NeuroRLEnvironment()

    observation, info = env.reset()

    assert observation.shape == (12,)
    assert "goal_distance" in info
    assert "goal_reached" in info
    assert "collision" in info


def test_environment_step_returns_valid_types():
    env = NeuroRLEnvironment()
    env.reset()

    action = np.array([0.0, 1.0], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == (12,)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "goal_distance" in info
    assert "step" in info


def test_environment_progresses_over_multiple_steps():
    env = NeuroRLEnvironment()
    env.reset()
    action = np.array([0.0, 1.0], dtype=np.float32)

    final_info = None
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(action)
        final_info = info
        if terminated or truncated:
            break

    assert final_info is not None
    assert final_info["step"] >= 1