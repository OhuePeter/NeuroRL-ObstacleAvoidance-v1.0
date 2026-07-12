"""
==========================================================
Experiment Logger

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Logs behavioural and neural data for every timestep.

Version:
1.0
==========================================================
"""

from pathlib import Path
import pandas as pd


class ExperimentLogger:
    """
    Logs experiment data.

    Each row corresponds to one simulation timestep.
    """

    def __init__(self):

        self.records = []

    def log(
        self,
        episode,
        step,
        time,
        agent,
        goal_distance,
        obstacle1_distance,
        obstacle2_distance,
        perturbation="P0",
        reward=0.0,
        success=False,
        collision=False,
    ):
        """
        Record one timestep.
        """

        self.records.append({

            "episode": episode,

            "step": step,

            "time": time,

            "x": agent.x,
            "y": agent.y,

            "vx": agent.vx,
            "vy": agent.vy,

            "speed": (agent.vx**2 + agent.vy**2)**0.5,

            "ax": agent.ax,
            "ay": agent.ay,

            "heading": agent.heading,

            "goal_distance": goal_distance,

            "obstacle1_distance": obstacle1_distance,

            "obstacle2_distance": obstacle2_distance,

            "perturbation": perturbation,

            "reward": reward,

            "success": success,

            "collision": collision

        })

    def dataframe(self):
        """
        Return the logged data as a pandas DataFrame.
        """
        return pd.DataFrame(self.records)

    def save(self, filename):
        """
        Save the logged data.
        """

        output_dir = Path("data/raw")

        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        self.dataframe().to_csv(filepath, index=False)

        print(f"\nSaved log to:\n{filepath}")