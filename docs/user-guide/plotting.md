# Plotting

HyperTorch provides lightweight utilities to inspect, parse, and visualize training metrics logged across experiments. 
The visualization pipeline centers around two primary components:

* **`LogParser`**: Scans the experiment logging tree (`hypertorch_logs/`), locates experiment runs, and extracts tabular data into Pandas DataFrames.
* **`LinePlotter`**: Generates publication-ready metric line charts from parsed logs, organizing them by model and metric tag.

For a runnable example script, see: 
- [`examples/plot/lineplot.py`](../../examples/plot/lineplot.py)

---

## Basic Plot Generation

By default, calling `plot()` plots every numerical metric tracked in the CSV:

```python
from hypertorch.train import LinePlotter, LogParser

#Initialize parser (defaults to "hypertorch_logs")
parser = LogParser()

# Locate the latest experiment directory and load metrics
latest_dir = parser.find_latest_experiment_dir()
df, csv_path = parser.load_latest_metrics()

#Generate line plots for all available metrics
plotter = LinePlotter(latest_dir)
saved_plots = plotter.plot(df, csv_path)
```

Plots are saved to an isolated `plots/` subdirectory inside the experiment folder:

```text
hypertorch_logs/
└── experiment_*/
    ├── <model_name>/
    └── plots/
        ├── train_loss.png
        ├── val_loss.png
        └── val_accuracy.png
```

---

### Selective Plotting (Filtering Metrics)

If you only want to visualize specific curves—such as loss curves or evaluation metrics—pass 
a list of metric column names via the `metrics` argument:

```python
from hypertorch.train import LinePlotter, LogParser

parser = LogParser()
latest_dir = parser.find_latest_experiment_dir()
df, csv_path = parser.load_latest_metrics()

plotter = LinePlotter(latest_dir)

#Only generate plots for training and validation loss
target_metrics = ["train/loss", "val/loss"]
saved_plots = plotter.plot(df, csv_path, metrics=target_metrics)
```

Alternatively, you can filter the Pandas DataFrame directly before plotting:

```python
#Keep metadata columns (epoch/step) and your desired metrics
selected_cols = [col for col in ["epoch", "step", "val/accuracy"] if col in df.columns]
filtered_df = df[selected_cols]

saved_plots = plotter.plot(filtered_df, csv_path)
```

### Loading Specific Runs with `load_csv`

To inspect a specific historical run or individual model directory, 
pass either a relative or absolute path to `load_csv`:

```python
# Relative paths resolve against base_logs_dir
df, resolved_path = parser.load_csv("experiment_1/gcn/version_0/metrics.csv")

# Absolute paths are accepted as-is
df, resolved_path = parser.load_csv("C:/hypertorch_logs/experiment_0/metrics.csv")
```