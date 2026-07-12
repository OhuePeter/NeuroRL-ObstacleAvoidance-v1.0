"""
==========================================================
Evaluation Script

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.evaluation.evaluator import PolicyEvaluator


def main():

    evaluator = PolicyEvaluator(

        "experiments/version_1_0/checkpoints/ppo_final.zip"

    )

    evaluator.evaluate(

        episodes=20

    )


if __name__ == "__main__":

    main()