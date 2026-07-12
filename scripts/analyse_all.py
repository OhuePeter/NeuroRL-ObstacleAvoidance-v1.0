"""
==========================================================
Complete Behavioural Analysis

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from pathlib import Path

from src.statistics.descriptive_statistics import DescriptiveStatistics
from src.visualization.kinematics_profiles import KinematicProfilePlotter
from src.visualization.velocity_profiles import VelocityProfilePlotter


RESULTS = "experiments/version_1_0/results"

conditions = [
    "P0",
    "L1",
    "L2",
    "L3",
    "R1",
    "R2",
    "R3",
]

print("=" * 60)
print("GENERATING KINEMATIC FIGURES")
print("=" * 60)

kin = KinematicProfilePlotter(RESULTS)

for c in conditions:
    print(f"  {c}")
    kin.plot(c)

print("\nGenerating velocity figures...")

vel = VelocityProfilePlotter(RESULTS)

for c in conditions:
    print(f"  {c}")
    vel.plot(c)

print("\nComputing descriptive statistics...")

stats = DescriptiveStatistics(RESULTS)

stats.analyse_all()

print("\nAnalysis complete.")