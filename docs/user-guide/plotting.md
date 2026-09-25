# Plotting

HyperTorch provides lightweight utilities to inspect, parse, and visualize training metrics logged across experiments. 
The visualization pipeline centers around three primary components:

* **`LogParser`**: Scans the experiment logging tree (default: `hypertorch_logs/`), discovers experiment runs, and reshapes raw CSV metric logs into a tidy `ParsedMetrics` container.
* **`ParsedMetrics`**: Stores the tidied DataFrames optimized for Seaborn, alongside metric names and experiment directory metadata.
* **`Plotter`**: Abstract base class for chart generation from `ParsedMetrics`. Currently implemented via `LinePlotter`.

!!! note "Optional Dependencies"
    Plotting requires `matplotlib` and `seaborn`. Install them with:
    ```bash
    pip install "hypertorch[plotting]"
    ```

---

## Basic Plot Generation

Initializing `LogParser` automatically points it to the default `hypertorch_logs/` folder. Calling `parse()` locates the latest experiment run and reshapes its metrics into a `ParsedMetrics` container. 

By default, calling `plot()` renders every metric found across all splits (train, validation, test) onto individual metric charts:

```python
from hypertorch.train import LinePlotter, LogParser

# Initialize parser and plotter
parser = LogParser()
plotter = LinePlotter()

# Automatically locate and parse the latest experiment run
metrics = parser.parse()

# Generate line plots directly from the parsed metrics
plotter.plot(metrics)
```

By default, plots are saved to an isolated `plots/` subdirectory inside the experiment folder:

```text
hypertorch_logs/
└── experiment_*/
    ├── <model_name>/
    └── plots/
        ├── LinePlot_loss_0.png
        └── LinePlot_accuracy_0.png
```
---

### Specified Parsing and Refresh (Lazy Caching)
`parse()` can accept an explicit path to a CSV file instead of auto-discovering the latest run. When called without arguments, `LogParser` caches the resolved experiment path so subsequent calls reuse it until `refresh()` is called.

Pass a new path to refresh() to scan a different base directory:

```python
parser = LogParser()
metrics = parser.parse()

# Refresh cached paths after running a new experiment
parser.refresh()
new_metrics = parser.parse()

# Switch to a custom base directory
parser.refresh("custom_logs_dir")
```

---

### Selective Plotting (Filtering Metrics)

If you only want to visualize specific curves, the `plot()` function can filter them with the `metric_names=[]` argument:

```python
# When wishing for only certain metrics:
saved_plots = plotter.plot(metrics, metric_names=["loss", "f1"])
```

---

### Custom Plotting Directory

Override the destination directory or toggle the creation of the `plots/` subfolder using keyword arguments:

```python
# Save directly into custom_dir/ without creating a plots/ subfolder
saved_plots = plotter.plot(
    metrics,
    output_dir="custom_dir",
    create_subfolder=False,
)
```