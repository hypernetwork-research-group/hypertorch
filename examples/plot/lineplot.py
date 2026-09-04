from hypertorch.train import LinePlotter, LogParser

#Initialize parser with the experiment log root
parser = LogParser("hypertorch_logs")

#Automatically locate and load the newest experiment run
df, csv_path = parser.load_latest_metrics()
latest_dir = parser.find_latest_experiment_dir()

#Generate line plots for all tracked metrics
plotter = LinePlotter(latest_dir)
saved_plots = plotter.plot(df, csv_path)

print(f"Generated {len(saved_plots)} plots in {latest_dir / 'plots'}")