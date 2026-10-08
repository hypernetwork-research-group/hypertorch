from pathlib import Path
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from hypertorch.data import (
    AlgebraDataset,
    DataLoader,
    LaplacianPositionalEncodingEnricher,
    RandomNegativeSampler,
)
from hypertorch.hyperlink_prediction import MLPPredictor
from hypertorch.train import LinePlotter, ScatterPlotter, LogParser, MultiModelTrainer
from hypertorch.types import ModelConfig


def main() -> None:


    """Application of LogParser and LinePlot can be seen here:"""

    # Instancing LogParser to discover and parse the latest metrics run
    print("\nParsing experiment metrics...")
    parser = LogParser()
    parser.discover_latest_metrics(2)
    parsed_runs = parser.parse_all()
    averaged_runs = [m.average() for m in parsed_runs]

>>>>>>> 91952f8 (Initial scatterplot implementation)
    # Instantiating LinePlotter to plot metrics
    print("Generating line plots...")
    plotter = ScatterPlotter()
    saved_plots: list[Path] = []

    for metrics in parsed_runs:
        # Metrics stores the directory as well as the names of the metrics found
        print(f"Discovered experiment directory: {metrics.experiment_dir}")
        print(f"Available metrics to plot: {metrics.names()}")
    saved_plots.extend(plotter.plot(averaged_runs))

    print(f"\nGenerated {len(saved_plots)} plot(s):")
    for plot_path in saved_plots:
        print(f" -> {Path(plot_path).resolve()}")


if __name__ == "__main__":
    main()