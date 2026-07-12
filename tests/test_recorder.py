"""
==========================================================
Recorder Test

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from src.utils.recorder import ExperimentRecorder
from src.utils.logger import ExperimentLogger


class DummyAgent:
    """
    Dummy agent used only for testing.
    """

    def __init__(self):

        self.x = 5.0
        self.y = 1.0

        self.vx = 0.0
        self.vy = 1.0

        self.ax = 0.0
        self.ay = 0.0

        self.heading = 1.57


def main():

    recorder = ExperimentRecorder()

    recorder.create_experiment()

    logger = ExperimentLogger()

    agent = DummyAgent()

    logger.log(

        episode=1,

        trial=1,

        step=0,

        time=0.0,

        seed=42,

        condition="P0",

        agent=agent,

        goal_distance=8.0,

        obstacle1_distance=3.0,

        obstacle2_distance=3.2,

        reward=0.0,

        success=False,

        collision=False,

        route="Centre"

    )

    logger.save(
        recorder.behaviour_path
    )

    recorder.save_metadata({

        "experiment": 1,

        "condition": "P0",

        "seed": 42,

        "agent": "PPO",

        "description":
        "Recorder functionality test."

    })

    recorder.save_summary({

        "success": False,

        "collision": False,

        "episode_length": 1,

        "reward": 0.0

    })

    recorder.save_readme()

    print()

    print("=" * 60)
    print("Experiment Successfully Created")
    print("=" * 60)

    print(recorder.path)


if __name__ == "__main__":
    main()