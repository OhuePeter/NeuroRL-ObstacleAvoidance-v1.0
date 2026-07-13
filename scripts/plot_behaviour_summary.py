from src.visualization.behaviour_summary import BehaviourSummaryPlotter

plotter = BehaviourSummaryPlotter(
    "experiments/version_1_0/results"
)

plotter.plot()

print("Behaviour summary figure complete.")