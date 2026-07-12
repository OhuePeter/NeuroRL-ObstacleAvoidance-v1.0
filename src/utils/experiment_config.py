"""
==========================================================
Experiment Configuration

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Creates a unified configuration object for the entire
project by combining all YAML configuration files.

Version:
1.0
==========================================================
"""

from dataclasses import dataclass

from src.utils.config import ConfigLoader


@dataclass
class ExperimentConfig:
    """
    Unified experiment configuration.
    """

    environment: dict
    perturbation: dict

    @classmethod
    def load(cls):
        """
        Load all project configuration files.
        """

        return cls(

            environment=ConfigLoader.load_environment(),

            perturbation=ConfigLoader.load_perturbation()

        )