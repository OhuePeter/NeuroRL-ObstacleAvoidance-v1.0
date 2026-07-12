"""
==========================================================
Configuration Loader

Authors:
Peter Ohue
Gunnar Blohm

Description
-----------
Loads YAML configuration files used throughout the project.
==========================================================
"""

from pathlib import Path
import yaml


class ConfigLoader:
    """
    Loads YAML configuration files.

    Examples
    --------
    env = ConfigLoader.load_environment()

    perturb = ConfigLoader.load_perturbation()
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    CONFIG_DIR = PROJECT_ROOT / "configs"

    @staticmethod
    def load_yaml(filename: str):
        """
        Load a YAML file.

        Parameters
        ----------
        filename : str
            YAML filename.

        Returns
        -------
        dict
            Parsed YAML dictionary.
        """

        filepath = ConfigLoader.CONFIG_DIR / filename

        with open(filepath, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    @staticmethod
    def load_environment():
        """Load environment configuration."""
        return ConfigLoader.load_yaml("environment.yaml")

    @staticmethod
    def load_perturbation():
        """Load perturbation configuration."""
        return ConfigLoader.load_yaml("perturbation.yaml")