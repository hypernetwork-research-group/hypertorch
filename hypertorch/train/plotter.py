from abc import ABC, abstractmethod
from pathlib import Path
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .parsed_metrics import ParsedMetrics


class Plotter(ABC):
    """Abstract Base Class (ABC) for all experiment plotters in HyperTorch.

    Establishes a common structure for plotters specializing in specific
    visualizations (Line, Scatter, etc.).

    Args:
        experiment_dir: Path to the experiment directory (e.g., 'hypertorch_logs/experiment_0').
    """

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def plot(
        self,
        metrics: ParsedMetrics,
        metric_names: list[str] | None = None,
    ) -> list[Path]:
        """Renders and saves plot images from parsed metrics.

        Args:
            metrics: ParsedMetrics container holding tidy metric DataFrames.
            metric_names: Optional subset of metric variables to plot.
                         If None, plots all available metrics.

        Returns:
            A list of Paths pointing to created image files.
        """


class LinePlotter(Plotter):
    """Generates Seaborn line plots for training and evaluation metrics."""

    def __init__(self, experiment_dir: str | Path) -> None:
        super().__init__(experiment_dir)
        match = re.search(r"experiment_(\d+)", self.experiment_dir.name)
        self.num_exp = match.group(1) if match else "0"

    def plot(
        self,
        metrics: ParsedMetrics,
        metric_names: list[str] | None = None,
    ) -> list[Path]:
        """Renders and saves line plots for metrics across epochs/steps.

        Args:
            metrics: ParsedMetrics container holding tidy metric DataFrames.
            metric_names: Optional subset of metric variables to plot (e.g., ['loss']).
                         If None, plots all metrics present in the container.

        Returns:
            List of generated plot image file paths.
        """
        sns.set_theme(style="darkgrid")
        saved_plots: list[Path] = []

        # Determine which variables to plot
        targets = metric_names if metric_names is not None else metrics.names()
        x_col = metrics.x_col

        for var_name in targets:
            if var_name not in metrics:
                continue

            # Directly fetch the tidy DataFrame
            tidy_df = metrics.fetch(var_name)

            fig, ax = plt.subplots(figsize=(8, 5))

            # Separate single-point evaluations from curves
            split_counts = tidy_df["split"].value_counts()
            single_point_splits = split_counts[split_counts == 1].index.tolist()

            continuous_df = tidy_df[~tidy_df["split"].isin(single_point_splits)]
            single_df = tidy_df[tidy_df["split"].isin(single_point_splits)]

            # Draw baseline horizontal lines for single-evaluation splits
            for split in single_point_splits:
                val = single_df[single_df["split"] == split]["value"].iloc[0]
                ax.axhline(
                    y=val,
                    color="#4C72B0" if split == "test" else "gray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=1,
                    label=f"{split} ({val:.4f})",
                )

            # Draw curves for multi-point metrics
            if not continuous_df.empty:
                sns.lineplot(
                    data=continuous_df,
                    x=x_col,
                    y="value",
                    hue="split",
                    marker="o",
                    ax=ax,
                    zorder=3,
                )

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles=handles, labels=labels, title="Split", loc="best")

            formatted_title = var_name.replace("_", " ").capitalize()
            ax.set_title(f"Experiment {self.num_exp} — {formatted_title}")
            ax.set_xlabel(x_col.capitalize())
            ax.set_ylabel(formatted_title)

            clean_filename_var = var_name.replace("/", "_")
            output_filename = f"LinePlot_{clean_filename_var}_{self.num_exp}.png"
            output_path = self.plots_dir / output_filename

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)

            saved_plots.append(output_path)

        return saved_plots