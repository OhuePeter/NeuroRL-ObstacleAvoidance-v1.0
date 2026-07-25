import json

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


def test_recorder_creates_experiment_and_writes_artifacts(tmp_path):
    recorder = ExperimentRecorder(root=tmp_path)

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

    assert recorder.path.exists()
    assert recorder.behaviour_path.exists()
    assert (recorder.path / "metadata.json").exists()
    assert (recorder.path / "summary.json").exists()
    assert (recorder.path / "README.txt").exists()

    metadata = json.loads((recorder.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["condition"] == "P0"