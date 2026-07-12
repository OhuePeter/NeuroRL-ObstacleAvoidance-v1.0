"""
Plot velocity profiles
"""

from src.visualization.velocity_profiles import VelocityProfilePlotter


plotter = VelocityProfilePlotter(
    "experiments/version_1_0/results"
)

for condition in [

    "P0",

    "L1",
    "L2",
    "L3",

    "R1",
    "R2",
    "R3"

]:

    plotter.plot(condition)

print("Velocity profiles complete.")