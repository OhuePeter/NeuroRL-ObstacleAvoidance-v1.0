from src.utils.experiment_config import ExperimentConfig


def test_experiment_config_loads_environment_and_perturbation():
    config = ExperimentConfig.load()

    assert isinstance(config.environment, dict)
    assert isinstance(config.perturbation, dict)
    assert "environment" in config.environment
    assert "perturbation" in config.perturbation
from src.utils.experiment_config import ExperimentConfig
