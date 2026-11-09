from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any
import warnings


from hypertorch.plot.plotter import Plotter
from hypertorch.types.parsed_metrics import ParsedMetrics


class ScatterPlotter(Plotter):
    """
    Generates composite Train vs. Val scatter plots across models with y = x reference lines.
    """

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """
        Render Train vs. Val generalization scatter subplots across models for each target metric.

        Args:
            metrics: Single ParsedMetrics or sequence of metrics to plot.
            metric_names: Optional subset of metric variables to plot.
            output_dir: Optional destination path for plot outputs.
            create_subfolder: Whether to save inside categorized subfolders.

        Returns:
            saved_plots: List of file paths to saved scatter plot PNG files.

        Raises:
            ImportError: If matplotlib or seaborn is not installed.
            ValueError: If metrics validation fails or no valid data is found.
        """
        plt, sns = self._init_plotting()
        metrics_list = self._normalize_metrics(metrics)
        self._validate_metrics_list(metrics_list)
        targets = self._resolve_target_metrics(metrics_list, metric_names)

        if not targets:
            return []

        metric_data = self._collect_generalization_data(metrics_list, targets)
        if not metric_data:
            return []

        plots_dir = self._resolve_plots_dir(metrics_list, output_dir, create_subfolder)
        palette = sns.color_palette("colorblind", n_colors=max(len(metrics_list), 10))

        comparison_metrics, single_run_metrics = self._partition_metrics(metric_data)
        saved_plots: list[Path] = []

        if comparison_metrics:
            saved_plots.extend(
                self._render_metric_group(
                    metrics_to_plot=comparison_metrics,
                    metric_data=metric_data,
                    title_prefix="Comparison",
                    file_stem="comparison",
                    plots_dir=plots_dir,
                    total_runs=len(metrics_list),
                    palette=palette,
                    plt=plt,
                    is_comparison=True,
                    create_subfolder=create_subfolder,
                )
            )

        for run_idx, r_metrics in single_run_metrics.items():
            run_label = metrics_list[run_idx].experiment_name or f"run_{run_idx}"
            saved_plots.extend(
                self._render_metric_group(
                    metrics_to_plot=r_metrics,
                    metric_data=metric_data,
                    title_prefix=f"Experiment {run_label}",
                    file_stem=run_label,
                    plots_dir=plots_dir,
                    total_runs=len(metrics_list),
                    palette=palette,
                    plt=plt,
                    is_comparison=False,
                    create_subfolder=create_subfolder,
                )
            )

        return saved_plots

    @staticmethod
    def _calculate_grid_dims(n_plots: int) -> tuple[int, int]:
        """
        Calculate balanced grid dimensions dynamically based on subplot count.

        Args:
            n_plots: Total number of subplots to place on the canvas.

        Returns:
            dims: Tuple of (n_cols, n_rows) defining the grid layout.
        """
        match n_plots:
            case 1:
                return 1, 1
            case 2:
                return 2, 1
            case 3:
                return 3, 1
            case 4:
                return 2, 2
            case _:
                return 3, math.ceil(n_plots / 3)

    @classmethod
    def _collect_generalization_data(
        cls,
        metrics_list: list[ParsedMetrics],
        targets: list[str],
    ) -> dict[str, dict[int, tuple[str, float, float]]]:
        """
        Extract final train and validation values for each metric across runs.

        Warns if an experiment run contains train but no val, or val but no train.

        Args:
            metrics_list: List of parsed experiment metrics.
            targets: Target metric variables to extract.

        Returns:
            metric_data: Mapping of metric name to run index and (label, train_val, val_val).
        """
        metric_data: dict[str, dict[int, tuple[str, float, float]]] = {}

        for var_name in targets:
            run_matches: dict[int, tuple[str, float, float]] = {}

            for run_idx, m in enumerate(metrics_list):
                if var_name not in m:
                    continue

                df = m.fetch(var_name)
                splits = [str(s) for s in df["split"].unique()]

                train_split = next((s for s in splits if s.lower() == "train"), None)
                if train_split is None:
                    train_split = next((s for s in splits if "train" in s.lower()), None)

                val_split = next(
                    (s for s in splits if s.lower() in {"val", "validation"}),
                    None,
                )
                if val_split is None:
                    val_split = next((s for s in splits if "val" in s.lower()), None)

                run_label = m.experiment_name or f"run_{run_idx}"

                if train_split is not None and val_split is None:
                    warnings.warn(
                        f"Experiment '{run_label}' has '{train_split}' split for metric "
                        f"'{var_name}' but no validation split. Skipping run for this metric.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                    continue

                if val_split is not None and train_split is None:
                    warnings.warn(
                        f"Experiment '{run_label}' has '{val_split}' split for metric "
                        f"'{var_name}' but no training split. Skipping run for this metric.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                    continue

                if train_split is None or val_split is None:
                    continue

                df_train = df[df["split"] == train_split]
                df_val = df[df["split"] == val_split]

                if df_train.empty or df_val.empty:
                    continue

                final_train = float(df_train["value"].iloc[-1])
                final_val = float(df_val["value"].iloc[-1])

                run_matches[run_idx] = (run_label, final_train, final_val)

            if run_matches:
                metric_data[var_name] = run_matches

        return metric_data

    @staticmethod
    def _partition_metrics(
        metric_data: dict[str, dict[int, tuple[str, float, float]]],
    ) -> tuple[list[str], dict[int, list[str]]]:
        """
        Partition metrics into multi-run comparisons and single-run metrics.

        Args:
            metric_data: Mapping of metric names to per-run evaluation points.

        Returns:
            partitions: Tuple of (comparison_metrics, single_run_metrics_by_run_index).
        """
        comparison_metrics: list[str] = []
        single_run_metrics: dict[int, list[str]] = {}

        for var_name, runs in metric_data.items():
            if len(runs) >= 2:
                comparison_metrics.append(var_name)
            else:
                run_idx = next(iter(runs.keys()))
                single_run_metrics.setdefault(run_idx, []).append(var_name)

        return comparison_metrics, single_run_metrics

    def _render_subplot(
        self,
        ax: Any,
        var_name: str,
        run_matches: dict[int, tuple[str, float, float]],
        palette: list[Any],
        show_label: bool,
    ) -> None:
        """
        Configure titles, labels, draw points and y=x diagonal reference line.

        Args:
            ax: Matplotlib axes to configure and populate.
            var_name: Metric name to plot.
            run_matches: Mapping of run index to (run_label, train_val, val_val).
            palette: Active color palette.
            show_label: Whether to expose labels to the figure legend.
        """
        all_train: list[float] = []
        all_val: list[float] = []

        for run_idx, (run_label, train_val, val_val) in run_matches.items():
            run_color = palette[run_idx % len(palette)]
            label = run_label if show_label else None
            ax.scatter(
                train_val,
                val_val,
                color=run_color,
                s=65,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
                label=label,
            )
            all_train.append(train_val)
            all_val.append(val_val)

        all_vals = all_train + all_val
        min_v = min(all_vals)
        max_v = max(all_vals)
        margin = (max_v - min_v) * 0.1 if max_v != min_v else 0.1
        lim_min = min_v - margin
        lim_max = max_v + margin

        ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            linestyle="--",
            color="gray",
            linewidth=1.2,
            alpha=0.7,
            zorder=2,
            label="y = x" if show_label else None,
        )

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        label_name = self._format_label(var_name)
        ax.set_title(f"{label_name}", fontsize=12, pad=10)
        ax.set_xlabel(f"Train {label_name}", fontsize=10)
        ax.set_ylabel(f"Val {label_name}", fontsize=10)
        ax.margins(x=0.08, y=0.08)

    def _render_metric_group(
        self,
        metrics_to_plot: list[str],
        metric_data: dict[str, dict[int, tuple[str, float, float]]],
        title_prefix: str,
        file_stem: str,
        plots_dir: Path,
        total_runs: int,
        palette: list[Any],
        plt: Any,
        is_comparison: bool,
        create_subfolder: bool,
    ) -> list[Path]:
        """
        Chunk metrics into groups of up to 6, render figure canvases, and save to disk.

        Args:
            metrics_to_plot: Metric names to plot.
            metric_data: Extracted train and val coordinates for each metric.
            title_prefix: Title context prefix ('Comparison' or experiment name).
            file_stem: Base identifier for output filenames.
            plots_dir: Root plot directory.
            total_runs: Total number of experiment runs in the batch.
            palette: Active color palette.
            plt: Matplotlib pyplot module.
            is_comparison: Whether rendering a multi-model comparison.
            create_subfolder: Whether to organize into categorized subdirectories.

        Returns:
            saved_plots: List of file paths to generated figure files.
        """
        dest_dir = self._resolve_dest_dir(
            plots_dir=plots_dir,
            is_multi=is_comparison,
            total_runs=total_runs,
            create_subfolder=create_subfolder,
        )

        max_plots_per_fig = 6
        total_chunks = math.ceil(len(metrics_to_plot) / max_plots_per_fig)
        saved_plots: list[Path] = []

        for chunk_idx in range(total_chunks):
            chunk = metrics_to_plot[
                chunk_idx * max_plots_per_fig : (chunk_idx + 1) * max_plots_per_fig
            ]
            n_plots = len(chunk)
            n_cols, n_rows = self._calculate_grid_dims(n_plots)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6.8 * n_cols, 5.8 * n_rows),
                squeeze=False,
            )
            flat_axes = axes.flatten()

            for idx, var_name in enumerate(chunk):
                self._render_subplot(
                    ax=flat_axes[idx],
                    var_name=var_name,
                    run_matches=metric_data[var_name],
                    palette=palette,
                    show_label=(idx == 0),
                )

            for empty_idx in range(n_plots, len(flat_axes)):
                flat_axes[empty_idx].set_visible(False)

            handles, labels = flat_axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    bbox_to_anchor=(1.02, 1.0),
                    loc="upper left",
                    borderaxespad=0.0,
                    frameon=True,
                    title=f"{title_prefix} (Runs)",
                )

            fig.suptitle(
                f"{title_prefix} — Train vs Val Generalization",
                fontsize=13,
                y=1.02,
            )

            part_suffix = f"_part{chunk_idx + 1}" if total_chunks > 1 else ""
            clean_file_stem = self._sanitize(file_stem)
            out_path = dest_dir / f"ScatterPlot_train_vs_val_{clean_file_stem}{part_suffix}.png"

            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_plots.append(out_path)

        return saved_plots
