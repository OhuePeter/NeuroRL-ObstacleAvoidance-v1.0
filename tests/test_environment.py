import numpy as np

from src.evaluation.biological_variability import BiologicalVariability
from src.environment.environment import NeuroRLEnvironment


def test_environment_reset_shape_and_info_keys():
    env = NeuroRLEnvironment()

    observation, info = env.reset()

    assert observation.shape == (13,)
    assert "goal_distance" in info
    assert "goal_reached" in info
    assert "collision" in info
    assert info["desired_route"] in {"left", "right", "either"}
    assert "target_x" in info
    assert "target_y" in info


def test_environment_step_returns_valid_types():
    env = NeuroRLEnvironment()
    env.reset()

    action = np.array([0.0, 1.0], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == (13,)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "goal_distance" in info
    assert "step" in info
    assert "desired_route" in info
    assert "waypoint_active" in info


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


def test_environment_alternates_route_cues_between_resets():
    env = NeuroRLEnvironment()

    first_observation, first_info = env.reset()
    second_observation, second_info = env.reset()

    assert first_info["desired_route"] in {"left", "right"}
    assert second_info["desired_route"] in {"left", "right"}
    assert first_info["desired_route"] != second_info["desired_route"]
    assert first_observation[-1] == -second_observation[-1]


def test_environment_route_targets_split_left_and_right():
    env = NeuroRLEnvironment()

    _, left_info = env.reset(options={"desired_route": "left"})
    _, right_info = env.reset(options={"desired_route": "right"})

    assert left_info["target_x"] < 5.0
    assert right_info["target_x"] > 5.0
    assert left_info["target_y"] == right_info["target_y"]
    assert left_info["waypoint_active"] is True
    assert right_info["waypoint_active"] is True


def test_biological_variability_uses_tightened_start_jitter():
    variability = BiologicalVariability()

    assert variability.start_position_jitter_std == 0.01