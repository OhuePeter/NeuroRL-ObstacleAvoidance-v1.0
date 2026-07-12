"""
==========================================================
Experiment Logger

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Logs behavioural, experimental and (later) neural
data for every simulation timestep.

Version:
1.0
==========================================================
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


class ExperimentLogger:
    """
    Records one row of data for every simulation timestep.
    """

    def __init__(self):

        self.records = []

    def log(
        self,
        episode,
        trial,
        step,
        time,
        seed,
        condition,
        agent,
        goal_distance,
        obstacle1_distance,
        obstacle2_distance,
        reward=0.0,
        success=False,
        collision=False,
        route="Unknown"
    ):
        """
        Record one timestep of behavioural data.
        """

        self.records.append({

            # -----------------------------
            # Experiment Information
            # -----------------------------

            "episode": episode,
            "trial": trial,
            "step": step,
            "time": time,
            "timestamp": datetime.now(),

            "seed": seed,
            "condition": condition,

            # -----------------------------
            # Behaviour
            # -----------------------------

            "x": agent.x,
            "y": agent.y,

            "vx": agent.vx,
            "vy": agent.vy,

            "speed": (agent.vx ** 2 + agent.vy ** 2) ** 0.5,

            "ax": agent.ax,
            "ay": agent.ay,

            "heading": agent.heading,

            # -----------------------------
            # Distances
            # -----------------------------

            "goal_distance": goal_distance,
            "obstacle1_distance": obstacle1_distance,
            "obstacle2_distance": obstacle2_distance,

            # -----------------------------
            # Outcome
            # -----------------------------

            "reward": reward,
            "collision": collision,
            "success": success,
            "route": route

        })

    def dataframe(self):
        """
        Return logged data as a pandas DataFrame.
        """
        return pd.DataFrame(self.records)

    def save(self, filepath):
        """
        Save behavioural data.

        Parameters
        ----------
        filepath : pathlib.Path
            Full path to behaviour.csv
        """

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.dataframe().to_csv(
            filepath,
            index=False
        )

        print("\nBehaviour saved to:")
        print(filepath)