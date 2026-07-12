"""
==========================================================
Experiment Recorder

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Creates a self-contained directory for every evaluation
episode and saves behavioural data, metadata and summaries.

Version:
1.0
==========================================================
"""

from pathlib import Path
import json
from datetime import datetime


class ExperimentRecorder:
    """
    Manages experiment output directories.
    """

    def __init__(
        self,
        root="experiments/version_1_0/results"
    ):

        self.root = Path(root)

        self.root.mkdir(
            parents=True,
            exist_ok=True
        )

    def create_experiment(self):

        existing = sorted(

            self.root.glob("experiment_*")

        )

        experiment_number = len(existing) + 1

        experiment_name = (
            f"experiment_{experiment_number:04d}"
        )

        self.path = self.root / experiment_name

        self.path.mkdir(exist_ok=True)

        return self.path

    def save_metadata(self, metadata):

        metadata["created"] = str(datetime.now())

        with open(

            self.path / "metadata.json",

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                metadata,

                file,

                indent=4

            )

    def save_summary(self, summary):

        with open(

            self.path / "summary.json",

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                summary,

                file,

                indent=4

            )

    def save_readme(self):

        text = """
Experiment Output

Files

behaviour.csv
Behavioural trajectory.

metadata.json
Experiment settings.

summary.json
Performance metrics.

Additional files will be generated during
analysis.
"""

        with open(

            self.path / "README.txt",

            "w",

            encoding="utf-8"

        ) as file:

            file.write(text)

    @property
    def behaviour_path(self):

        return self.path / "behaviour.csv"