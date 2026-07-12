"""
==========================================================
Experiment Configuration Test

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from src.utils.experiment_config import ExperimentConfig


def main():

    config = ExperimentConfig.load()

    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)

    print(config.environment)

    print()

    print("=" * 60)
    print("PERTURBATION")
    print("=" * 60)

    print(config.perturbation)


if __name__ == "__main__":
    main()
    