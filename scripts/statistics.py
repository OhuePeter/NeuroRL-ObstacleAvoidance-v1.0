"""
==========================================================
Generate Descriptive Statistics

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

from src.statistics.descriptive_statistics import (
    DescriptiveStatistics,
)

RESULTS = "experiments/version_1_0/results"

stats = DescriptiveStatistics(
    RESULTS
)

stats.analyse_all()

print("\nDescriptive statistics complete.")