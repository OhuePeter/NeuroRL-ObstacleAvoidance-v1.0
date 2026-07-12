"""
==========================================================
PPO Trainer

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.environment.environment import NeuroRLEnvironment
from src.training.callbacks import build_callbacks


class PPOTrainer:

    """
    PPO Training Manager.
    """

    def __init__(self):

        self.env = make_vec_env(

            NeuroRLEnvironment,

            n_envs=8

        )

        policy_kwargs = dict(

            net_arch=[256, 256]

        )

        self.model = PPO(

            "MlpPolicy",

            self.env,

            learning_rate=3e-4,

            gamma=0.99,

            gae_lambda=0.95,

            n_steps=1024,

            batch_size=512,

            ent_coef=0.01,

            verbose=1,

            tensorboard_log="experiments/version_1_0/logs",

            policy_kwargs=policy_kwargs

        )

        self.callbacks = build_callbacks()

    def train(

        self,

        total_timesteps=3_000_000

    ):

        self.model.learn(

            total_timesteps=total_timesteps,

            callback=self.callbacks,

            progress_bar=True

        )

        checkpoint_dir = Path(

            "experiments/version_1_0/checkpoints"

        )

        checkpoint_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        self.model.save(

            checkpoint_dir /

            "ppo_final"

        )

        print()

        print("=" * 60)

        print("TRAINING FINISHED")

        print("=" * 60)