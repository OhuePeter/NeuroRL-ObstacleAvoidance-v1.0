"""
==========================================================
Experiment Logger

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Records behavioural, experimental and (later) neural
data for every simulation timestep.

Version:
1.0
==========================================================
"""

from pathlib import Path
from datetime import datetime

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

            "speed": (agent.vx**2 + agent.vy**2)**0.5,

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
            # Experiment Outcome
            # -----------------------------

            "reward": reward,

            "collision": collision,

            "success": success,

            "route": route

        })

    def dataframe(self):

        return pd.DataFrame(self.records)

    def save(self, filename):

        output = Path("data/raw")

        output.mkdir(parents=True, exist_ok=True)

        filepath = output / filename

        self.dataframe().to_csv(filepath, index=False)

        print(f"\nSaved log to:\n{filepath}")