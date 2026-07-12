"""
==========================================================
Reward Function Test

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from src.environment.reward import RewardFunction


def main():

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

    print("=" * 60)
    print("REWARD BREAKDOWN")
    print("=" * 60)

    for key, value in results.items():

        print(f"{key:12s}: {value:.3f}")


if __name__ == "__main__":
    main()