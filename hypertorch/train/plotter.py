from abc import ABC, abstractmethod
from collections.abc import Sequence
import importlib
import importlib.util
import itertools
from pathlib import Path

from hypertorch.types import ParsedMetrics
from hypertorch.utils.file_utils import validate_base_dir


class Plotter(ABC):
    """Abstract Base Class (ABC) for all experiment plotters in HyperTorch.

    Establishes a common structure, inherited by classes that specialize
    in specific chart types (Line, Scatter, etc.).
    """

    MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "<", ">", "P", "X", "*")

    @staticmethod
    def _is_plotting_available() -> bool:
        """Check whether matplotlib and seaborn are importable."""
        return (
            importlib.util.find_spec("matplotlib") is not None
            and importlib.util.find_spec("seaborn") is not None
        )

    @staticmethod
    def _normalize_metrics(
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
    ) -> list[ParsedMetrics]:
        """Normalizes input into a list of ParsedMetrics instances."""
        if isinstance(metrics, ParsedMetrics):
            return [metrics]
        metrics_list = list(metrics)
        if not metrics_list:
            raise ValueError("No ParsedMetrics provided to plot.")
        return metrics_list

    @staticmethod
    def _validate_metrics(metrics: ParsedMetrics) -> None:
        """Validates that a single ParsedMetrics instance contains usable data."""
        if not getattr(metrics, "x_col", None):
            raise ValueError("ParsedMetrics contains no primary column.")
        if len(metrics) == 0:
            raise ValueError("ParsedMetrics contains no metrics to plot.")

    @classmethod
    def _validate_metrics_list(cls, metrics_list: list[ParsedMetrics]) -> None:
        """Validates that all ParsedMetrics in the list are compatible."""
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
        """Resolves target metric names strictly preserving insertion order."""
        if metric_names is not None:
            return [
                name for name in metric_names
                if any(name in m for m in metrics_list)
            ]
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
        """Resolves, validates, and creates the target directory for plots."""
        if output_dir is not None:
            base_dir = Path(output_dir)
        else:
            first_dir = next(
                (m.experiment_dir for m in metrics_list if m.experiment_dir is not None),
                None,
            )
            if first_dir is None:
                raise ValueError("Could not find a destination path.")

            base_dir = Path(first_dir).parent if len(metrics_list) > 1 else Path(first_dir)

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
        """Renders and saves plot images from parsed metrics."""


class LinePlotter(Plotter):
    """Generates Seaborn line plots for training and evaluation metrics."""

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """Renders and saves line plots across epochs/steps."""
        if not self._is_plotting_available():
            raise ImportError(
                "Plotting dependencies are not available. "
                "Install them with `pip install hypertorch[plotting]`"
            )

        metrics_list = self._normalize_metrics(metrics)
        self._validate_metrics_list(metrics_list)
        plots_dir = self._resolve_plots_dir(metrics_list, output_dir, create_subfolder)
        targets = self._resolve_target_metrics(metrics_list, metric_names)

        is_multi = len(metrics_list) > 1
        x_col = metrics_list[0].x_col

        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        sns = importlib.import_module("seaborn")
        sns.set_theme(style="darkgrid")

        n_colors = max(len(metrics_list), 10)
        palette = sns.color_palette("tab10" if n_colors <= 10 else "tab20", n_colors=n_colors)
        saved_plots: list[Path] = []

        for var_name in targets:
            runs_with_metric = [
                (idx, m) for idx, m in enumerate(metrics_list) if var_name in m
            ]
            if not runs_with_metric:
                continue

            fig, ax = plt.subplots(figsize=(8.5, 5.2))

            # Dynamically identify all unique splits in order of appearance
            unique_splits: list[str] = []
            for _, m in runs_with_metric:
                for s in m.fetch(var_name)["split"].unique():
                    s_str = str(s)
                    if s_str not in unique_splits:
                        unique_splits.append(s_str)

            split_markers = {
                s: self.MARKERS[i % len(self.MARKERS)]
                for i, s in enumerate(unique_splits)
            }

            for run_idx, m in runs_with_metric:
                tidy_df = m.fetch(var_name)
                run_label = m.experiment_name or f"run_{run_idx}"
                run_color = palette[run_idx % len(palette)]
                base_zorder = 2 + (run_idx * 2)

                for split_name in unique_splits:
                    group = tidy_df[tidy_df["split"] == split_name]
                    if group.empty:
                        continue

                    marker = split_markers[split_name]

                    if is_multi:
                        color = run_color
                        label = f"{run_label} ({split_name})"
                    else:
                        s_idx = unique_splits.index(split_name)
                        color = palette[s_idx % len(palette)]
                        label = split_name

                    # Single-point evaluation (benchmark line)
                    if len(group) == 1:
                        val = float(group["value"].iloc[0])
                        ax.axhline(
                            y=val,
                            color=color,
                            linestyle=":",
                            linewidth=1.6,
                            alpha=0.75,
                            zorder=base_zorder,
                            label=f"{label}: {val:.4f}",
                        )
                    else:
                        sorted_group = group.sort_values(by=x_col)
                        # Spacing markers prevents a dense bead-string on 100+ epoch runs
                        markevery = max(1, len(sorted_group) // 10) if len(sorted_group) > 20 else 1

                        ax.plot(
                            sorted_group[x_col],
                            sorted_group["value"],
                            color=color,
                            linestyle="-",
                            marker=marker,
                            markersize=6,
                            markevery=markevery,
                            linewidth=2.0,
                            zorder=base_zorder + 1,
                            label=label,
                        )

            formatted_metric = var_name.replace("_", " ").capitalize()
            title_prefix = "Comparison" if is_multi else f"Experiment {metrics_list[0].experiment_name or '0'}"
            ax.set_title(f"{title_prefix} — {formatted_metric}", fontsize=12, pad=10)
            ax.set_xlabel(x_col.capitalize())
            ax.set_ylabel(formatted_metric)
            ax.legend(title="Runs & Splits" if is_multi else "Split", loc="best", frameon=True)

            clean_var = var_name.replace("/", "_")
            suffix = "comparison" if is_multi else (metrics_list[0].experiment_name or "0")
            output_path = plots_dir / f"LinePlot_{clean_var}_{suffix}.png"

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            saved_plots.append(output_path)

        return saved_plots


class ScatterPlotter(Plotter):
    """Generates pairwise Seaborn scatter plots comparing metrics against each other."""

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """Renders and saves pairwise metric scatter plots across runs."""
        if not self._is_plotting_available():
            raise ImportError(
                "Plotting dependencies are not available. "
                "Install them with `pip install hypertorch[plotting]`"
            )

        metrics_list = self._normalize_metrics(metrics)
        self._validate_metrics_list(metrics_list)
        targets = self._resolve_target_metrics(metrics_list, metric_names)

        if len(targets) < 2:
            raise ValueError(
                "ScatterPlotter requires at least two metrics to plot relationships. "
                f"Found: {targets}."
            )

        plots_dir = self._resolve_plots_dir(metrics_list, output_dir, create_subfolder)
        is_multi = len(metrics_list) > 1
        x_col = metrics_list[0].x_col

        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        sns = importlib.import_module("seaborn")
        sns.set_theme(style="darkgrid")

        n_colors = max(len(metrics_list), 10)
        palette = sns.color_palette("tab10" if n_colors <= 10 else "tab20", n_colors=n_colors)
        saved_plots: list[Path] = []

        for var_x, var_y in itertools.combinations(targets, 2):
            runs_with_both = [
                (idx, m) for idx, m in enumerate(metrics_list) if var_x in m and var_y in m
            ]
            if not runs_with_both:
                continue

            fig, ax = plt.subplots(figsize=(8.5, 5.2))

            unique_splits: list[str] = []
            for _, m in runs_with_both:
                for s in m.fetch(var_x)["split"].unique():
                    s_str = str(s)
                    if s_str not in unique_splits:
                        unique_splits.append(s_str)

            split_markers = {
                s: self.MARKERS[i % len(self.MARKERS)]
                for i, s in enumerate(unique_splits)
            }

            has_plotted_points = False

            for run_idx, m in runs_with_both:
                df_x = m.fetch(var_x)
                df_y = m.fetch(var_y)
                merged = df_x.merge(
                    df_y,
                    on=[x_col, "split"],
                    suffixes=(f"_{var_x}", f"_{var_y}"),
                )
                if merged.empty:
                    continue

                run_label = m.experiment_name or f"run_{run_idx}"
                run_color = palette[run_idx % len(palette)]
                base_zorder = 2 + (run_idx * 2)

                col_x = f"value_{var_x}"
                col_y = f"value_{var_y}"

                for split_name in unique_splits:
                    group = merged[merged["split"] == split_name]
                    if group.empty:
                        continue

                    has_plotted_points = True
                    marker = split_markers[split_name]

                    if is_multi:
                        color = run_color
                        label = f"{run_label} ({split_name})"
                    else:
                        s_idx = unique_splits.index(split_name)
                        color = palette[s_idx % len(palette)]
                        label = split_name

                    ax.scatter(
                        group[col_x],
                        group[col_y],
                        color=color,
                        marker=marker,
                        s=60,
                        alpha=0.85,
                        edgecolors="white",
                        linewidths=0.6,
                        zorder=base_zorder + 1,
                        label=label,
                    )

            if not has_plotted_points:
                plt.close(fig)
                continue

            label_x = var_x.replace("_", " ").capitalize()
            label_y = var_y.replace("_", " ").capitalize()
            title_prefix = "Comparison" if is_multi else f"Experiment {metrics_list[0].experiment_name or '0'}"
            ax.set_title(f"{title_prefix} — {label_y} vs {label_x}", fontsize=12, pad=10)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            ax.legend(title="Runs & Splits" if is_multi else "Split", loc="best", frameon=True)

            clean_x = var_x.replace("/", "_")
            clean_y = var_y.replace("/", "_")
            suffix = "comparison" if is_multi else (metrics_list[0].experiment_name or "0")
            output_path = plots_dir / f"ScatterPlot_{clean_y}_vs_{clean_x}_{suffix}.png"

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            saved_plots.append(output_path)

        return saved_plots