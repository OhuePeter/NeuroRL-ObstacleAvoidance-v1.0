from src.neural_analysis.latent_trajectory import LatentTrajectory

analysis = LatentTrajectory(
    "experiments/version_1_0/results"
)

analysis.analyse()

print("\nLatent trajectory analysis complete.")