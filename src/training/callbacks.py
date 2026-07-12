"""
==========================================================
Training Callbacks

Authors:
Peter Ohue
Gunnar Blohm

Version:
1.0
==========================================================
"""

from pathlib import Path

from stable_baselines3.common.callbacks import (

    CheckpointCallback,

    EvalCallback,

    CallbackList

)

from stable_baselines3.common.monitor import Monitor

from src.environment.environment import NeuroRLEnvironment


def build_callbacks():

    """
    Construct all callbacks used during training.
    """

    checkpoint_dir = Path(
        "experiments/version_1_0/checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_dir = Path(
        "experiments/version_1_0/evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    eval_env = Monitor(

        NeuroRLEnvironment()

    )

    checkpoint_callback = CheckpointCallback(

        save_freq=50000,

        save_path=str(checkpoint_dir),

        name_prefix="ppo"

    )

    evaluation_callback = EvalCallback(

        eval_env,

        best_model_save_path=str(checkpoint_dir),

        log_path=str(evaluation_dir),

        eval_freq=50000,

        deterministic=True,

        render=False

    )

    callback = CallbackList([

        checkpoint_callback,

        evaluation_callback

    ])

    return callback