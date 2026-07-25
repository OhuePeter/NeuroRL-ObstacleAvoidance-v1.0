from src.environment.reward import RewardFunction


def test_reward_contains_expected_components_and_total():
    reward = RewardFunction()

    results = reward.compute_total_reward(

        previous_goal_distance=8.0,

        current_goal_distance=7.6,

        minimum_obstacle_distance=1.2,

        goal_reached=False,

        collision=False,

        ax=0.05,

        ay=0.02,

    )

    expected = {
        "goal",
        "collision",
        "progress",
        "distance",
        "clearance",
        "smoothness",
        "time",
        "total",
    }

    assert expected.issubset(set(results.keys()))

    component_sum = sum(
        value for key, value in results.items() if key != "total"
    )

    assert abs(component_sum - results["total"]) < 1e-10