from abc import ABC, abstractmethod
import importlib
import importlib.util
from pathlib import Path

from hypertorch.types import ParsedMetrics


class Plotter(ABC):
    """Abstract Base Class (ABC) for all experiment plotters in HyperTorch.

    Establishes a common structure, to be inherited by all the future classes
    that specialize in a type of plot (Line, Scatter, etc.).
    """

    @staticmethod
    def _is_plotting_available() -> bool:
        """Check whether matplotlib and seaborn are importable.

        Returns:
            available: ``True`` when both matplotlib and seaborn are installed.
        """
        return (
            importlib.util.find_spec("matplotlib") is not None
            and importlib.util.find_spec("seaborn") is not None
        )

    @staticmethod
    def _validate_metrics(metrics: ParsedMetrics) -> None:
        """Validates that ParsedMetrics contains the necessary plotting data.

        Raises:
            ValueError: If primary tracking column (x_col) is missing or if
                        no metrics are stored.
        """
        if not getattr(metrics, "x_col", None):
            raise ValueError("ParsedMetrics contains no primary column.")
        if len(metrics) == 0:
            raise ValueError("ParsedMetrics contains no metrics to plot.")

    @staticmethod
    def _resolve_plots_dir(
        metrics: ParsedMetrics,
        output_dir: str | Path | None,
        create_subfolder: bool,
    ) -> Path:
        """Resolves, validates, and creates the target directory for plots.

        Raises:
            ValueError: If no destination path can be found.
            FileNotFoundError: If the target directory does not exist.
            NotADirectoryError: If the target path is not a directory.
        """
        if output_dir is not None:
            base_dir = Path(output_dir)
        elif metrics.experiment_dir is not None:
            base_dir = Path(metrics.experiment_dir)
        else:
            raise ValueError("Could not find a destination path.")

        if not base_dir.exists():
            raise FileNotFoundError(f"Destination directory '{base_dir}' does not exist.")
        if not base_dir.is_dir():
            raise NotADirectoryError(f"Destination path '{base_dir}' is not a directory.")

        plots_dir = base_dir / "plots" if create_subfolder else base_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        return plots_dir

    @abstractmethod
    def plot(
        self,
        metrics: ParsedMetrics,
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """Renders and saves plot images from parsed metrics.

        Args:
            metrics: ParsedMetrics container holding tidy metric DataFrames.
            metric_names: Optional subset of metric variables to plot.
                         If None, plots all available metrics.
            output_dir: Optional path where the plots will be saved. If None,
                        defaults to metrics.experiment_dir.
            create_subfolder: If True, creates and saves into a 'plots' subfolder
                              within the target directory. Defaults to True.

        Returns:
            A list of Paths pointing to created image files.

        Raises:
            ImportError: If matplotlib or seaborn is not installed.
            TypeError: when unimplemented by concrete classes
        """


class LinePlotter(Plotter):
    """Generates Seaborn line plots for training and evaluation metrics."""

    def plot(
        self,
        metrics: ParsedMetrics,
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """Renders and saves line plots for metrics across epochs/steps.

        Args:
            metrics: ParsedMetrics container holding tidy metric DataFrames.
            metric_names: Optional subset of metric variables to plot.
                         If None, plots all available metrics.
            output_dir: Optional path where the plots will be saved. If None,
                        defaults to metrics.experiment_dir.
            create_subfolder: If True, creates and saves into a 'plots' subfolder
                              within the target directory. Defaults to True.

        Returns:
            List of generated plot image file paths.

        Raises:
            ImportError: If matplotlib or seaborn is not installed.
            ValueError: If parsedMetrics contains no destination path and no further path was given,
                        if metrics.x_col is None or empty, or if parsedMetrics contains no metrics.
            FileNotFoundError: If the destination folder does not exist.
            NotADirectoryError: If the destination path doesn't point to a directory.
        """
        if not self._is_plotting_available():
            raise ImportError(
                "Plotting dependencies are not available. "
                "Install them with `pip install hypertorch[plotting]`"
            )

        # Base class validation and destination resolution
        self._validate_metrics(metrics)
        plots_dir = self._resolve_plots_dir(metrics, output_dir, create_subfolder)
        num_exp = metrics.experiment_name or "0"

        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        sns = importlib.import_module("seaborn")

        sns.set_theme(style="darkgrid")
        saved_plots: list[Path] = []

        # Determine which variables to plot
        targets = metric_names if metric_names is not None else metrics.names()
        x_col = metrics.x_col

        for var_name in targets:
            if var_name not in metrics:
                continue

            tidy_df = metrics.fetch(var_name)
            fig, ax = plt.subplots(figsize=(8, 5))

            # Separate single-point evaluations from curves
            split_counts = tidy_df["split"].value_counts()
            single_point_splits = split_counts[split_counts == 1].index.tolist()

            continuous_df = tidy_df[~tidy_df["split"].isin(single_point_splits)]
            single_df = tidy_df[tidy_df["split"].isin(single_point_splits)]

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
            ax.set_title(f"Experiment {num_exp} — {formatted_title}")
            ax.set_xlabel(x_col.capitalize())
            ax.set_ylabel(formatted_title)

            clean_filename_var = var_name.replace("/", "_")
            output_filename = f"LinePlot_{clean_filename_var}_{num_exp}.png"
            output_path = plots_dir / output_filename

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)

            saved_plots.append(output_path)

        return saved_plots
