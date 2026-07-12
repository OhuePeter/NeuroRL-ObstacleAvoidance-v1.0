"""
==========================================================
Evaluation Script

Authors:
Peter Ohue
Gunnar Blohm

Version:
1.1
==========================================================
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.evaluation.evaluator import PolicyEvaluator


def main():

    model_path = (
        "experiments/version_1_0/checkpoints/ppo_baseline_P0.zip"
    )

    evaluator = PolicyEvaluator(model_path)

    evaluator.evaluate(
        episodes=20
    )


if __name__ == "__main__":
    main()