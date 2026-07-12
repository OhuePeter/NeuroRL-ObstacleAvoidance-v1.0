"""
==========================================================
Training Script

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.training.trainer import PPOTrainer


def main():

    trainer = PPOTrainer()

    # -------------------------------------------------
    # Development test
    # -------------------------------------------------

    trainer.train(

        total_timesteps=50000

    )


if __name__ == "__main__":
    main()