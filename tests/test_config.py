from src.utils.config import ConfigLoader


def test_load_environment_has_expected_sections():
    env = ConfigLoader.load_environment()

    assert isinstance(env, dict)
    assert "environment" in env
    assert "agent" in env
    assert "goal" in env
    assert "obstacles" in env


def test_load_perturbation_has_conditions():
    perturb = ConfigLoader.load_perturbation()

    assert isinstance(perturb, dict)
    assert "perturbation" in perturb
    assert "conditions" in perturb["perturbation"]
    assert "P0" in perturb["perturbation"]["conditions"]