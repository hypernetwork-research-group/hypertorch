from abc import ABC, abstractmethod
from collections.abc import Sequence
import colorsys
import importlib
import importlib.util
from pathlib import Path
from typing import Any

from hypertorch.types import ParsedMetrics
from hypertorch.utils import validate_base_dir


class Plotter(ABC):
    """Abstract Base Class (ABC) for all experiment plotters in HyperTorch.

    Establishes a common structure, inherited by classes that specialize
    in specific chart types (Line, Scatter, etc.).
    """

    @staticmethod
    def _is_plotting_available() -> bool:
        """
        Check whether matplotlib and seaborn are importable in the environment.

        Returns:
            available: True when both matplotlib and seaborn can be imported.
        """
        return (
            importlib.util.find_spec("matplotlib") is not None
            and importlib.util.find_spec("seaborn") is not None
        )

    @classmethod
    def _init_plotting(cls) -> tuple[Any, Any]:
        """
        Validate plotting dependencies and configure matplotlib and seaborn backends.

        Returns:
            modules: Tuple of (matplotlib.pyplot, seaborn) modules.

        Raises:
            ImportError: If matplotlib or seaborn is not installed.
        """
        if not cls._is_plotting_available():
            raise ImportError(
                "Plotting dependencies are not available. "
                "Install them with `pip install hypertorch[plotting]`"
            )
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        sns = importlib.import_module("seaborn")
        sns.set_theme(style="darkgrid")
        return plt, sns

    @staticmethod
    def _format_label(name: str) -> str:
        """
        Format a metric or variable name into a human-readable title label.

        Args:
            name: Raw metric or variable name (e.g., 'local_loss').

        Returns:
            label: Capitalized display label with underscores replaced by spaces.
        """
        return name.replace("_", " ").capitalize()

    @staticmethod
    def _sanitize(name: str) -> str:
        """
        Sanitize an identifier for safe filesystem path and filename usage.

        Args:
            name: Path component or variable name.

        Returns:
            sanitized: Cleaned string with forward slashes replaced by underscores.
        """
        return name.replace("/", "_")

    @staticmethod
    def _resolve_run_naming(is_multi: bool, run_name: str = "0") -> tuple[str, str]:
        """
        Resolve the title prefix and file stem based on comparison context.

        Args:
            is_multi: Whether the context involves multiple experiment runs.
            run_name: Identifier of the primary or single experiment run.

        Returns:
            naming: Tuple of (title_prefix, file_stem).
        """
        title_prefix = "Comparison" if is_multi else f"Experiment {run_name}"
        file_stem = "comparison" if is_multi else run_name
        return title_prefix, file_stem

    @staticmethod
    def _resolve_dest_dir(
        plots_dir: Path,
        is_multi: bool,
        total_runs: int,
        create_subfolder: bool,
    ) -> Path:
        """
        Route destination directory to categorized subfolders or root plot directory.

        Args:
            plots_dir: Base directory where plots are written.
            is_multi: Whether the metric or pair is shared across multiple runs.
            total_runs: Total number of experiment runs being processed.
            create_subfolder: Whether to route plots into categorized subdirectories.

        Returns:
            dest_dir: Target directory path for saving plot artifacts.
        """
        if total_runs > 1:
            subfolder_name = "comparison" if is_multi else "unique"
            dest_dir = plots_dir / subfolder_name if create_subfolder else plots_dir
        else:
            dest_dir = plots_dir

        dest_dir.mkdir(parents=True, exist_ok=True)
        return dest_dir

    @staticmethod
    def _get_shade(
        base_color: tuple[float, ...] | list[float],
        split_idx: int,
        num_splits: int,
    ) -> tuple[float, float, float]:
        """
        Generate a shade of a base color by adjusting lightness while preserving hue.

        Args:
            base_color: Base RGB or RGBA color tuple or list.
            split_idx: Zero-based index of the current data split.
            num_splits: Total number of splits sharing this base color.

        Returns:
            color: Adjusted RGB color tuple maintaining hue with scaled lightness.
        """
        if num_splits <= 1:
            return (base_color[0], base_color[1], base_color[2])

        r, g, b = base_color[:3]
        h, _, s = colorsys.rgb_to_hls(r, g, b)

        min_l, max_l = 0.30, 0.72
        new_l = min_l + (split_idx / (num_splits - 1)) * (max_l - min_l)
        new_s = min(1.0, max(s, 0.65))

        return colorsys.hls_to_rgb(h, new_l, new_s)

    @staticmethod
    def _normalize_metrics(
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
    ) -> list[ParsedMetrics]:
        """
        Normalize input into a uniform list of ParsedMetrics instances.

        Args:
            metrics: Single ParsedMetrics instance or sequence of instances.

        Returns:
            metrics_list: Standardized list of ParsedMetrics instances.

        Raises:
            ValueError: If the input collection is empty.
        """
        if isinstance(metrics, ParsedMetrics):
            return [metrics]
        metrics_list = list(metrics)
        if not metrics_list:
            raise ValueError("No ParsedMetrics provided to plot.")
        return metrics_list

    @staticmethod
    def _validate_metrics(metrics: ParsedMetrics) -> None:
        """
        Validate that a single ParsedMetrics instance contains usable plotting data.

        Args:
            metrics: ParsedMetrics instance to inspect.

        Raises:
            ValueError: If primary tracking column (x_col) is missing or metrics store is empty.
        """
        if not getattr(metrics, "x_col", None):
            raise ValueError("ParsedMetrics contains no primary column.")
        if len(metrics) == 0:
            raise ValueError("ParsedMetrics contains no metrics to plot.")

    @classmethod
    def _validate_metrics_list(cls, metrics_list: list[ParsedMetrics]) -> None:
        """
        Validate that all ParsedMetrics instances in a list are compatible.

        Args:
            metrics_list: List of ParsedMetrics instances to inspect.

        Raises:
            ValueError: If list is empty, any instance is invalid, or x_col values mismatch.
        """
        if not metrics_list:
            raise ValueError("ParsedMetrics list cannot be empty.")
        primary_x = metrics_list[0].x_col
        for m in metrics_list:
            cls._validate_metrics(m)
            if m.x_col != primary_x:
                raise ValueError(
                    f"Mismatched primary column '{m.x_col}' found; "
                    f"expected '{primary_x}' across all experiments."
                )

    @staticmethod
    def _resolve_target_metrics(
        metrics_list: list[ParsedMetrics],
        metric_names: list[str] | None,
    ) -> list[str]:
        """
        Resolve target metric names while strictly preserving insertion order.

        Args:
            metrics_list: Parsed experiment metrics to query.
            metric_names: Optional specific subset of metric names to filter by.

        Returns:
            targets: Ordered list of target metric names to plot.
        """
        if metric_names is not None:
            return [name for name in metric_names if any(name in m for m in metrics_list)]
        seen: set[str] = set()
        targets: list[str] = []
        for m in metrics_list:
            for name in m.names():
                if name not in seen:
                    seen.add(name)
                    targets.append(name)
        return targets

    @staticmethod
    def _resolve_plots_dir(
        metrics_list: list[ParsedMetrics],
        output_dir: str | Path | None,
        create_subfolder: bool,
    ) -> Path:
        """
        Resolve, validate, and create the destination directory for plot artifacts.

        Args:
            metrics_list: Parsed metrics containing source experiment directory paths.
            output_dir: Optional explicit output directory path override.
            create_subfolder: Whether to append a 'plots' subfolder to the target path.

        Returns:
            plots_dir: Resolved and created target plots directory.

        Raises:
            ValueError: If no valid destination directory can be derived.
            FileNotFoundError: If the target parent directory does not exist.
            NotADirectoryError: If the target path is not a directory.
        """
        if output_dir is not None:
            target = output_dir
        else:
            first_dir = next(
                (m.experiment_dir for m in metrics_list if m.experiment_dir is not None),
                None,
            )
            if first_dir is None:
                raise ValueError("Could not find a destination path.")

            target = Path(first_dir).parent if len(metrics_list) > 1 else Path(first_dir)

        base_dir = validate_base_dir(target)
        plots_dir = base_dir / "plots" if create_subfolder else base_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        return plots_dir

    @abstractmethod
    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """
        Render and save plot images from parsed metrics.

        Args:
            metrics: Single ParsedMetrics or sequence of metrics to plot.
            metric_names: Optional subset of metric variables to plot.
            output_dir: Optional destination path for plot outputs.
            create_subfolder: Whether to save inside a 'plots' subfolder.

        Returns:
            paths: List of file paths to generated plot images.
        """
