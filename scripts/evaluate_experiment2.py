"""
==========================================================
Experiment 2 Evaluation Driver

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Runs the frozen PPO controller under the
Experiment 2 robustness protocol.

Loads the policy trained in Experiment 1 and
evaluates it under progressively stronger
perturbations.

==========================================================
"""

from pathlib import Path

from src.evaluation.evaluator_v2 import PolicyEvaluatorV2

from experiments.version_2_0.evaluation.config import (
    N_EPISODES,
)

MODEL = Path(
    "experiments/version_1_0/checkpoints/ppo_baseline_P0.zip"
)

CONDITIONS = [
    "P0",
    "L1",
    "L2",
    "L3",
    "R1",
    "R2",
    "R3",
]


def main():

    print("=" * 70)
    print("EXPERIMENT 2")
    print("Robustness Evaluation")
    print("=" * 70)

    evaluator = PolicyEvaluatorV2(
        model_path=str(MODEL)
    )

    all_results = {}

    for condition in CONDITIONS:

        print()
        print("=" * 70)
        print(f"Running Condition: {condition}")
        print("=" * 70)

        result = evaluator.evaluate(

            episodes=N_EPISODES,

            condition=condition,

            output_root="experiments/version_2_0/results",

            save_video=False,

        )

        all_results[condition] = result

    print()

    print("=" * 70)
    print("Experiment 2 Finished")
    print("=" * 70)

    print()

    print("Summary")

    print("-" * 70)

    for condition, result in all_results.items():

        print(
            f"{condition:>3} | "
            f"{result['successes']:3d}/"
            f"{result['episodes']:3d} successes "
            f"({100*result['successes']/result['episodes']:.1f}%)"
        )


if __name__ == "__main__":
    main()