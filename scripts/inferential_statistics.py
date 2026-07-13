from src.statistics.inferential_statistics import InferentialStatistics

stats = InferentialStatistics(
    "experiments/version_1_0/results"
)

for metric in [
    "reward",
    "steps",
    "path_length",
    "mean_speed",
    "max_speed"
]:
    stats.analyse_metric(metric)