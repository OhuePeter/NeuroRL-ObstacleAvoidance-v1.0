"""
Run inferential statistics.
"""

from src.statistics.inferential_statistics import InferentialStatistics

stats = InferentialStatistics(
    "experiments/version_1_0/results"
)

stats.analyse_all()

print("\nInferential statistics complete.")