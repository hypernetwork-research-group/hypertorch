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

Initializing `LogParser` validates and points to the base logging directory (default: `hypertorch_logs/`).

To plot the most recent experiment run, discover it via `discover_latest(num=1)`, parse the queued runs with `parse_all()`, and render the curves using `LinePlotter`:

```python
from hypertorch.train import LinePlotter, LogParser

# Initialize parser and plotter
parser = LogParser()
plotter = LinePlotter()

# Locate the most recent run directory and parse it
parser.discover_latest(num=1)
metrics = parser.parse_all()[0]

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

## Discovery & Batch Parsing
`LogParser` maintains an internal queue (`directory_paths`) mapping discovered run folders to their metric files without performing redundant filesystem scans during parsing.

### Discovering Recent Runs
Queue the latest $N$ runs across all experiment folders using `discover_latest(num=N)`:

```python
parser = LogParser()
parser.discover_latest(num=3)

# Parse all queued runs and plot them individually
for metrics in parser.parse_all():
    plotter.plot(metrics)
```

### Discovering Runs by Folder Name
Queue all subdirectories containing metric logs within a specific experiment folder using `discover_directory()`:
```python
parser = LogParser()

# Queue all runs inside 'hypertorch_logs/experiment_3'
parser.discover_directory("experiment_3")

for metrics in parser.parse_all():
    plotter.plot(metrics)
```
---

## Direct Path Parsing
If you already know the folder or CSV file you want to inspect, `parse_from_path()` parses it directly without modifying the internal discovery queue:

```python
parser = LogParser()

# Parse directly from a run directory (automatically loads the newest CSV inside it)
metrics = parser.parse_from_path("experiment_3/mlp/version_0")
plotter.plot(metrics)

# Or parse directly from an explicit CSV file
metrics = parser.parse_from_path("experiment_3/mlp/version_0/metrics.csv")
plotter.plot(metrics)
```

## Switching the Base Directory
To retarget `LogParser` to another root directory without losing previously discovered runs, use `move_base_dir()`:

```python
parser = LogParser("hypertorch_logs")
parser.discover_directory("experiment_1")

# Move base directory to a new location (validates existence immediately)
parser.move_base_dir("archived_logs")
parser.discover_directory("experiment_old")

# parse_all() now parses runs from both locations
all_metrics = parser.parse_all()
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