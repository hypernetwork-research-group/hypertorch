# Plotting

HyperTorch provides lightweight utilities to inspect, parse, and visualize training metrics logged across experiments. 
The visualization pipeline centers around two primary components:

* **`LogParser`**: Scans the experiment logging tree (Default: `hypertorch_logs/`), locates experiment runs, extracts data into Pandas DataFrames and parses them in tidy subset of DataFrames called `ParsedMetrics`.
* **`Plotter`**: Generates charts from `ParsedMetrics`, organizing them by model and metric tag. 

Currently, there is only one type of plot chart generation, `LinePlotter`.

---

## Basic Plot Generation
Initializing a `LogParser` automatically points it to the default experiment folder. The `parse()` function will automatically locate and parse the latest csv file, unless it's been given a specific path.
All `Plotter` should be initialized in the same experiment folder where the csv files are found, as it will look up the name of the experiment to name the image files, as well as create the necessary subfolder for saving files.
By default, calling `plot()` plots every numerical metric tracked in the `ParsedMetrics`:

```python
# Initialize parser
parser = LogParser()

# Automatically locate and parse the newest experiment
metrics = parser.parse()

# Generate line plots directly from the parsed metrics container
latest_dir = parser.find_latest_experiment_dir()
plotter = LinePlotter(latest_dir)
saved_plots = plotter.plot(metrics)
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

If you only want to visualize specific curves, the `parse()` function can filter them with the `metrics_names=[]` argument:

```python
# When wishing for only certain metrics:
saved_plots = plotter.plot(metrics, metric_names=["loss", "f1"])
```
