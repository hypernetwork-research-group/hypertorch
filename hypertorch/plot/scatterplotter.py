from collections.abc import Sequence
import itertools
import math
from pathlib import Path
from typing import Any

import pandas as pd

from hypertorch.plot.plotter import Plotter
from hypertorch.types.parsed_metrics import ParsedMetrics


class ScatterPlotter(Plotter):
    """Generates composite scatter plots, dynamically sized and grouped by reference metric."""

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """
        Render pairwise metric subplots across models, separated by data split and anchor metric.

        Args:
            metrics: Single ParsedMetrics or sequence of metrics to plot.
            metric_names: Optional subset of metric variables to plot.
            output_dir: Optional destination path for plot outputs.
            create_subfolder: Whether to save inside categorized subfolders.

        Returns:
            saved_plots: List of file paths to saved scatter plot PNG files.

        Raises:
            ImportError: If matplotlib or seaborn is not installed.
            ValueError: If fewer than two target metrics are available or validation fails.
        """
        plt, sns = self._init_plotting()
        metrics_list = self._normalize_metrics(metrics)
        self._validate_metrics_list(metrics_list)
        targets = self._resolve_target_metrics(metrics_list, metric_names)

        if len(targets) < 2:
            raise ValueError(
                "ScatterPlotter requires at least two metrics to plot relationships. "
                f"Found: {targets}."
            )

        plots_dir = self._resolve_plots_dir(metrics_list, output_dir, create_subfolder)
        x_col = metrics_list[0].x_col
        palette = sns.color_palette("colorblind", n_colors=max(len(metrics_list), 10))

        unique_splits = self._get_unique_splits(metrics_list, targets)
        all_candidate_pairs = list(itertools.combinations(targets, 2))
        saved_plots: list[Path] = []

        for split_name in unique_splits:
            pair_data = self._collect_pair_data(
                metrics_list=metrics_list,
                candidate_pairs=all_candidate_pairs,
                split_name=split_name,
                x_col=x_col,
            )
            if not pair_data:
                continue

            comparison_pairs, single_run_pairs = self._partition_pairs(pair_data)

            for anchor, pairs in self._group_by_anchor(comparison_pairs).items():
                saved_plots.extend(
                    self._render_anchor_group(
                        pairs_to_plot=pairs,
                        pair_data=pair_data,
                        title_prefix="Comparison",
                        file_stem="comparison",
                        anchor_name=anchor,
                        split_name=split_name,
                        plots_dir=plots_dir,
                        total_runs=len(metrics_list),
                        palette=palette,
                        plt=plt,
                        is_comparison=True,
                        create_subfolder=create_subfolder,
                    )
                )

            for run_idx, r_pairs in single_run_pairs.items():
                run_label = metrics_list[run_idx].experiment_name or f"run_{run_idx}"
                for anchor, pairs in self._group_by_anchor(r_pairs).items():
                    saved_plots.extend(
                        self._render_anchor_group(
                            pairs_to_plot=pairs,
                            pair_data=pair_data,
                            title_prefix=f"Experiment {run_label}",
                            file_stem=run_label,
                            anchor_name=anchor,
                            split_name=split_name,
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

    @staticmethod
    def _get_unique_splits(metrics_list: list[ParsedMetrics], targets: list[str]) -> list[str]:
        """
        Identify all distinct data split names across target metrics.

        Args:
            metrics_list: List of parsed experiment metrics.
            targets: List of target metric names to search across.

        Returns:
            unique_splits: Ordered list of distinct split names discovered.
        """
        unique_splits: list[str] = []
        for m in metrics_list:
            for var in targets:
                if var in m:
                    for s in m.fetch(var)["split"].unique():
                        s_str = str(s)
                        if s_str not in unique_splits:
                            unique_splits.append(s_str)
        return unique_splits

    @staticmethod
    def _collect_pair_data(
        metrics_list: list[ParsedMetrics],
        candidate_pairs: list[tuple[str, str]],
        split_name: str,
        x_col: str,
    ) -> dict[tuple[str, str], dict[int, tuple[str, pd.DataFrame]]]:
        """
        Merge metric pairs and prune identical aliases or empty splits.

        Args:
            metrics_list: List of parsed experiment metrics.
            candidate_pairs: Candidate pairwise metric tuples.
            split_name: Evaluation split to filter by (e.g., 'train', 'val').
            x_col: Primary tracking column name.

        Returns:
            pair_data: Mapping of pairwise tuples to per-run merged DataFrames.
        """
        pair_data: dict[tuple[str, str], dict[int, tuple[str, pd.DataFrame]]] = {}

        for var_x, var_y in candidate_pairs:
            run_matches: dict[int, tuple[str, pd.DataFrame]] = {}
            col_x, col_y = f"value_{var_x}", f"value_{var_y}"
            is_redundant_alias = True

            for run_idx, m in enumerate(metrics_list):
                if var_x not in m or var_y not in m:
                    continue

                df_x_split = m.fetch(var_x).query(f"split == '{split_name}'")
                df_y_split = m.fetch(var_y).query(f"split == '{split_name}'")

                if df_x_split.empty or df_y_split.empty:
                    continue

                merged = df_x_split.merge(
                    df_y_split,
                    on=[x_col, "split"],
                    suffixes=(f"_{var_x}", f"_{var_y}"),
                ).sort_values(by=x_col)

                if len(merged) < 2:
                    continue

                if ((merged[col_x] - merged[col_y]).abs() > 1e-4).any():
                    is_redundant_alias = False

                run_label = m.experiment_name or f"run_{run_idx}"
                run_matches[run_idx] = (run_label, merged)

            if run_matches and not is_redundant_alias:
                pair_data[(var_x, var_y)] = run_matches

        return pair_data

    @staticmethod
    def _partition_pairs(
        pair_data: dict[tuple[str, str], dict[int, tuple[str, pd.DataFrame]]],
    ) -> tuple[list[tuple[str, str]], dict[int, list[tuple[str, str]]]]:
        """
        Partition pairwise metric groups into multi-run comparisons and single-run pairs.

        Args:
            pair_data: Mapping of metric pairs to per-run merged DataFrames.

        Returns:
            partitions: Tuple of (comparison_pairs, single_run_pairs_by_run_index).
        """
        comparison_pairs: list[tuple[str, str]] = []
        single_run_pairs: dict[int, list[tuple[str, str]]] = {}

        for pair, runs in pair_data.items():
            if len(runs) >= 2:
                comparison_pairs.append(pair)
            else:
                run_idx = next(iter(runs.keys()))
                single_run_pairs.setdefault(run_idx, []).append(pair)

        return comparison_pairs, single_run_pairs

    @staticmethod
    def _group_by_anchor(
        pairs: list[tuple[str, str]],
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Group pairwise metric combinations by their horizontal anchor variable.

        Args:
            pairs: List of pairwise metric name tuples (var_x, var_y).

        Returns:
            anchor_groups: Mapping of horizontal anchor metrics (var_x) to associated pairs.
        """
        anchor_groups: dict[str, list[tuple[str, str]]] = {}
        for var_x, var_y in pairs:
            anchor_groups.setdefault(var_x, []).append((var_x, var_y))
        return anchor_groups

    @staticmethod
    def _draw_run_scatter(
        ax: Any,
        merged: pd.DataFrame,
        col_x: str,
        col_y: str,
        run_color: Any,
        label: str | None,
    ) -> None:
        """
        Render points for an individual run's pairwise trajectory and terminal state.

        Args:
            ax: Matplotlib axes to draw on.
            merged: Merged DataFrame with aligned x and y metric coordinates.
            col_x: Column name for the horizontal metric.
            col_y: Column name for the vertical metric.
            run_color: Base color assigned to this run.
            label: Legend label or None to suppress duplicate legend entries.
        """
        ax.scatter(
            merged[col_x],
            merged[col_y],
            color=run_color,
            marker=".",
            s=32,
            alpha=0.8,
            zorder=2,
            label=label,
        )
        ax.scatter(
            merged[col_x].iloc[-1],
            merged[col_y].iloc[-1],
            color=run_color,
            s=60,
            marker="X",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    def _render_subplot(
        self,
        ax: Any,
        var_x: str,
        var_y: str,
        run_matches: dict[int, tuple[str, pd.DataFrame]],
        palette: list[Any],
        show_label: bool,
    ) -> None:
        """
        Configure titles, labels, padding margins, and draw model trajectories on a subplot.

        Args:
            ax: Matplotlib axes to configure and populate.
            var_x: Metric name on the horizontal axis.
            var_y: Metric name on the vertical axis.
            run_matches: Mapping of run index to run label and merged DataFrame.
            palette: Active color palette.
            show_label: Whether to expose labels to the figure legend.
        """
        col_x, col_y = f"value_{var_x}", f"value_{var_y}"

        for run_idx, (run_label, merged) in run_matches.items():
            run_color = palette[run_idx % len(palette)]
            label = run_label if show_label else None
            self._draw_run_scatter(
                ax=ax,
                merged=merged,
                col_x=col_x,
                col_y=col_y,
                run_color=run_color,
                label=label,
            )

        label_x = self._format_label(var_x)
        label_y = self._format_label(var_y)
        ax.set_title(f"{label_y} vs {label_x}", fontsize=12, pad=10)
        ax.set_xlabel(label_x, fontsize=10)
        ax.set_ylabel(label_y, fontsize=10)
        ax.margins(x=0.08, y=0.08)

    @staticmethod
    def _hide_unused_axes(axes: Sequence[Any], active_count: int) -> None:
        """
        Hide empty axes in non-full grid layouts.

        Args:
            axes: Flat sequence of axes from the figure grid.
            active_count: Number of subplots with actual data rendered.
        """
        for empty_idx in range(active_count, len(axes)):
            axes[empty_idx].set_visible(False)

    @staticmethod
    def _attach_chunk_legend(
        fig: Any,
        reference_ax: Any,
        title_prefix: str,
        split_name: str,
        is_comparison: bool,
    ) -> None:
        """
        Attach a unified legend to the figure based on the primary subplot handles.

        Args:
            fig: Matplotlib Figure instance.
            reference_ax: Subplot axis containing primary legend handles and labels.
            title_prefix: Legend header prefix context.
            split_name: Data split name for context label.
            is_comparison: Whether the figure compares multiple models.
        """
        handles, labels = reference_ax.get_legend_handles_labels()
        if not handles:
            return

        leg_title = (
            f"{title_prefix} ({split_name.capitalize()} — X = Final)"
            if is_comparison
            else f"{title_prefix} ({split_name.capitalize()})"
        )
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            title=leg_title,
        )

    def _render_anchor_group(
        self,
        pairs_to_plot: list[tuple[str, str]],
        pair_data: dict[tuple[str, str], dict[int, tuple[str, pd.DataFrame]]],
        title_prefix: str,
        file_stem: str,
        anchor_name: str,
        split_name: str,
        plots_dir: Path,
        total_runs: int,
        palette: list[Any],
        plt: Any,
        is_comparison: bool,
        create_subfolder: bool,
    ) -> list[Path]:
        """
        Chunk and render anchor groups into one or more figure files.

        Args:
            pairs_to_plot: Pairwise metric tuples sharing a common anchor.
            pair_data: Mapping of metric pairs to per-run merged DataFrames.
            title_prefix: Title context prefix ('Comparison' or experiment name).
            file_stem: Base identifier for output filenames.
            anchor_name: Metric name serving as horizontal reference.
            split_name: Evaluation split name.
            plots_dir: Root plot directory.
            total_runs: Total number of experiment runs in the batch.
            palette: Active color palette.
            plt: Matplotlib pyplot module.
            is_comparison: Whether rendering a multi-model comparison.
            create_subfolder: Whether to organize into categorized subdirectories.

        Returns:
            saved_chunk_plots: List of file paths to generated figure chunks.
        """
        dest_dir = self._resolve_dest_dir(
            plots_dir=plots_dir,
            is_multi=is_comparison,
            total_runs=total_runs,
            create_subfolder=create_subfolder,
        )

        max_plots_per_fig = 6
        total_chunks = math.ceil(len(pairs_to_plot) / max_plots_per_fig)
        saved_chunk_plots: list[Path] = []

        for chunk_idx in range(total_chunks):
            chunk = pairs_to_plot[
                chunk_idx * max_plots_per_fig : (chunk_idx + 1) * max_plots_per_fig
            ]
            out_path = self._render_chunk(
                chunk=chunk,
                pair_data=pair_data,
                dest_dir=dest_dir,
                title_prefix=title_prefix,
                file_stem=file_stem,
                anchor_name=anchor_name,
                split_name=split_name,
                chunk_idx=chunk_idx,
                total_chunks=total_chunks,
                palette=palette,
                plt=plt,
                is_comparison=is_comparison,
            )
            saved_chunk_plots.append(out_path)

        return saved_chunk_plots

    def _render_chunk(
        self,
        chunk: list[tuple[str, str]],
        pair_data: dict[tuple[str, str], dict[int, tuple[str, pd.DataFrame]]],
        dest_dir: Path,
        title_prefix: str,
        file_stem: str,
        anchor_name: str,
        split_name: str,
        chunk_idx: int,
        total_chunks: int,
        palette: list[Any],
        plt: Any,
        is_comparison: bool,
    ) -> Path:
        """
        Render an individual grid canvas containing up to six subplots.

        Args:
            chunk: Slice of pairwise tuples (at most 6) to render on this canvas.
            pair_data: Mapping of metric pairs to per-run merged DataFrames.
            dest_dir: Target output directory for image file.
            title_prefix: Title context prefix.
            file_stem: Base identifier for output filenames.
            anchor_name: Metric name serving as horizontal reference.
            split_name: Evaluation split name.
            chunk_idx: Zero-based index of the current chunk.
            total_chunks: Total number of chunks for this anchor group.
            palette: Active color palette.
            plt: Matplotlib pyplot module.
            is_comparison: Whether rendering a multi-model comparison.

        Returns:
            out_path: File path of the rendered and saved figure PNG.
        """
        n_plots = len(chunk)
        n_cols, n_rows = self._calculate_grid_dims(n_plots)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6.8 * n_cols, 5.8 * n_rows),
            squeeze=False,
        )
        flat_axes = axes.flatten()

        for idx, (var_x, var_y) in enumerate(chunk):
            self._render_subplot(
                ax=flat_axes[idx],
                var_x=var_x,
                var_y=var_y,
                run_matches=pair_data[(var_x, var_y)],
                palette=palette,
                show_label=(idx == 0),
            )

        self._hide_unused_axes(flat_axes, n_plots)
        self._attach_chunk_legend(fig, flat_axes[0], title_prefix, split_name, is_comparison)

        formatted_anchor = self._format_label(anchor_name)
        fig.suptitle(
            f"{title_prefix} — Relationships vs {formatted_anchor}",
            fontsize=13,
            y=1.02,
        )

        part_suffix = f"_part{chunk_idx + 1}" if total_chunks > 1 else ""
        clean_file_stem = self._sanitize(file_stem)
        clean_anchor = self._sanitize(anchor_name)
        out_path = (
            dest_dir
            / f"ScatterPlot_rel_{split_name}_{clean_file_stem}_vs_{clean_anchor}{part_suffix}.png"
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
