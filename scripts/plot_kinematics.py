"""
Generate kinematic profile figures.
"""

from src.visualization.kinematic_profiles import KinematicProfilePlotter


plotter = KinematicProfilePlotter(
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

print("All kinematic figures generated.")