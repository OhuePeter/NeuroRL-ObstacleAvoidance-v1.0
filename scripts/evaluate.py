"""
==========================================================
Evaluate Trained PPO Policy

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from src.evaluation.evaluator import PolicyEvaluator


def main():

    MODEL = "experiments/version_1_0/checkpoints/ppo_baseline_P0"

    CONDITION = "R3"

    evaluator = PolicyEvaluator(
        model_path=MODEL
    )

    evaluator.evaluate(
        episodes=20,
        condition=CONDITION
    )


if __name__ == "__main__":
    main()