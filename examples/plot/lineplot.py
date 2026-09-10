from hypertorch.train import LinePlotter, LogParser

# Initialize parser with the experiment log root
parser = LogParser("hypertorch_logs")

# Automatically locate and parse the newest experiment run into a ParsedMetrics container
metrics = parser.parse()
latest_dir = parser.find_latest_experiment_dir()

# Generate line plots directly from the parsed metrics container
plotter = LinePlotter(latest_dir)
saved_plots = plotter.plot(metrics)
# When wishing for only certain metrics:
# saved_plots = plotter.plot(metrics, metric_names=["loss", "f1"])

print(f"Generated {len(saved_plots)} plots in {latest_dir / 'plots'}")