"""
Generate descriptive statistics.
"""

from src.statistics.descriptive_statistics import DescriptiveStatistics


stats = DescriptiveStatistics(
    "experiments/version_1_0/results"
)

stats.analyse_all()

print("Statistics complete.")