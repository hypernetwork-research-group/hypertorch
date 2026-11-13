from abc import ABC, abstractmethod
from collections.abc import Sequence
import colorsys
import importlib
import importlib.util
import itertools
import math
from pathlib import Path

import pandas as pd
from hypertorch.types import ParsedMetrics
from hypertorch.utils.file_utils import validate_base_dir


class Plotter(ABC):
    """Abstract Base Class (ABC) for all experiment plotters in HyperTorch.

    Establishes a common structure, inherited by classes that specialize
    in specific chart types (Line, Scatter, etc.).
    """

    @staticmethod
    def _is_plotting_available() -> bool:
        """Check whether matplotlib and seaborn are importable."""
        return (
            importlib.util.find_spec("matplotlib") is not None
            and importlib.util.find_spec("seaborn") is not None
        )

    @staticmethod
    def _get_shade(
        base_color: tuple[float, ...] | list[float],
        split_idx: int,
        num_splits: int,
    ) -> tuple[float, float, float]:
        """Generates a shade of base_color based on split index while keeping the hue."""
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

        x_col = metrics_list[0].x_col

        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        sns = importlib.import_module("seaborn")
        sns.set_theme(style="darkgrid")

        palette = sns.color_palette("colorblind")
        saved_plots: list[Path] = []

        for var_name in targets:
            runs_with_metric = [
                (idx, m) for idx, m in enumerate(metrics_list) if var_name in m
            ]
            if not runs_with_metric:
                continue

            is_multi = len(runs_with_metric) > 1

            # Resolve subfolder based on whether this metric belongs to multiple runs
            subfolder_name = "comparison" if is_multi else "unique"
            dest_dir = plots_dir / subfolder_name if create_subfolder else plots_dir
            dest_dir.mkdir(parents=True, exist_ok=True)

            unique_splits: list[str] = []
            for _, m in runs_with_metric:
                for s in m.fetch(var_name)["split"].unique():
                    s_str = str(s)
                    if s_str not in unique_splits:
                        unique_splits.append(s_str)

            total_curves = sum(
                1
                for _, m in runs_with_metric
                for split_name in unique_splits
                if not m.fetch(var_name)[m.fetch(var_name)["split"] == split_name].empty
            )

            is_busy = total_curves > 2
            fig_size = (11.2, 5.2) if is_busy else (8.5, 5.0)
            fig, ax = plt.subplots(figsize=fig_size)

            for run_idx, m in runs_with_metric:
                tidy_df = m.fetch(var_name)
                run_label = m.experiment_name or f"run_{run_idx}"
                run_color = palette[run_idx % len(palette)]
                base_zorder = 2 + (run_idx * 2)

                for split_name in unique_splits:
                    group = tidy_df[tidy_df["split"] == split_name]
                    if group.empty:
                        continue

                    s_idx = unique_splits.index(split_name)

                    if is_multi:
                        color = self._get_shade(run_color, s_idx, len(unique_splits))
                        label = f"{run_label} ({split_name})"
                    else:
                        color = palette[s_idx % len(palette)]
                        label = split_name

                    marker = None if is_busy else "."

                    if len(group) == 1:
                        val = float(group["value"].iloc[0])
                        ax.axhline(
                            y=val,
                            color=color,
                            linestyle=":",
                            linewidth=1.3,
                            alpha=0.8,
                            zorder=base_zorder,
                            label=f"{label}: {val:.4f}",
                        )
                    else:
                        sorted_group = group.sort_values(by=x_col)
                        ax.plot(
                            sorted_group[x_col],
                            sorted_group["value"],
                            color=color,
                            linestyle="-",
                            linewidth=1.4,
                            marker=marker,
                            markersize=5.5 if marker else None,
                            zorder=base_zorder + 1,
                            label=label,
                        )

            formatted_metric = var_name.replace("_", " ").capitalize()
            first_run_name = runs_with_metric[0][1].experiment_name or "0"
            title_prefix = "Comparison" if is_multi else f"Experiment {first_run_name}"
            suffix = "comparison" if is_multi else first_run_name

            ax.set_title(f"{title_prefix} — {formatted_metric}", fontsize=12, pad=10)
            ax.set_xlabel(x_col.capitalize())
            ax.set_ylabel(formatted_metric)

            if is_busy:
                ax.legend(
                    title="Runs & Splits" if is_multi else "Split",
                    bbox_to_anchor=(1.02, 1.0),
                    loc="upper left",
                    borderaxespad=0.0,
                    frameon=True,
                )
            else:
                ax.legend(
                    title="Runs & Splits" if is_multi else "Split",
                    loc="best",
                    framealpha=0.85,
                )

            clean_var = var_name.replace("/", "_")
            output_path = dest_dir / f"LinePlot_{clean_var}_{suffix}.png"

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_plots.append(output_path)

        return saved_plots


class ScatterPlotter(Plotter):
    """Generates composite pairwise scatter plots, dynamically sized and grouped by reference metric."""

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """Renders pairwise metric subplots across models, separated by data split and anchor metric."""
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
        x_col = metrics_list[0].x_col

        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        sns = importlib.import_module("seaborn")
        sns.set_theme(style="darkgrid")

        palette = sns.color_palette("colorblind", n_colors=max(len(metrics_list), 10))

        unique_splits: list[str] = []
        for m in metrics_list:
            for var in targets:
                if var in m:
                    for s in m.fetch(var)["split"].unique():
                        s_str = str(s)
                        if s_str not in unique_splits:
                            unique_splits.append(s_str)

        all_candidate_pairs = list(itertools.combinations(targets, 2))
        max_plots_per_fig = 6
        saved_plots: list[Path] = []

        for split_name in unique_splits:
            pair_data: dict[tuple[str, str], dict[int, tuple[str, pd.DataFrame]]] = {}

            for var_x, var_y in all_candidate_pairs:
                run_matches: dict[int, tuple[str, pd.DataFrame]] = {}
                col_x = f"value_{var_x}"
                col_y = f"value_{var_y}"
                is_redundant_alias = True

                for run_idx, m in enumerate(metrics_list):
                    if var_x not in m or var_y not in m:
                        continue

                    df_x = m.fetch(var_x)
                    df_y = m.fetch(var_y)
                    df_x_split = df_x[df_x["split"] == split_name]
                    df_y_split = df_y[df_y["split"] == split_name]

                    if df_x_split.empty or df_y_split.empty:
                        continue

                    merged = df_x_split.merge(
                        df_y_split,
                        on=[x_col, "split"],
                        suffixes=(f"_{var_x}", f"_{var_y}"),
                    ).sort_values(by=x_col)

                    if len(merged) < 2:
                        continue

                    diff = (merged[col_x] - merged[col_y]).abs()
                    if (diff > 1e-4).any():
                        is_redundant_alias = False

                    run_label = m.experiment_name or f"run_{run_idx}"
                    run_matches[run_idx] = (run_label, merged)

                if run_matches and not is_redundant_alias:
                    pair_data[(var_x, var_y)] = run_matches

            if not pair_data:
                continue

            # Separate multi-run comparisons from single-run metrics
            comparison_pairs: list[tuple[str, str]] = []
            single_run_pairs: dict[int, list[tuple[str, str]]] = {}

            for pair, runs in pair_data.items():
                if len(runs) >= 2:
                    comparison_pairs.append(pair)
                else:
                    run_idx = next(iter(runs.keys()))
                    single_run_pairs.setdefault(run_idx, []).append(pair)

            def _render_anchor_group(
                pairs_to_plot: list[tuple[str, str]],
                title_prefix: str,
                file_stem: str,
                anchor_name: str,
                is_comparison: bool,
            ) -> None:
                """Renders pairs sharing the same anchor into dynamically sized figures"""
                subfolder_name = "comparison" if is_comparison else "unique"
                dest_dir = plots_dir / subfolder_name if create_subfolder else plots_dir
                dest_dir.mkdir(parents=True, exist_ok=True)

                total_chunks = math.ceil(len(pairs_to_plot) / max_plots_per_fig)

                for chunk_idx in range(total_chunks):
                    chunk = pairs_to_plot[
                        chunk_idx * max_plots_per_fig : (chunk_idx + 1) * max_plots_per_fig
                    ]
                    n_plots = len(chunk)

                    # Dynamic geometry based on plot count
                    if n_plots == 1:
                        n_cols, n_rows = 1, 1
                    elif n_plots == 2:
                        n_cols, n_rows = 2, 1
                    elif n_plots == 3:
                        n_cols, n_rows = 3, 1
                    elif n_plots == 4:
                        n_cols, n_rows = 2, 2
                    else:  # 5 or 6 plots
                        n_cols, n_rows = 3, math.ceil(n_plots / 3)

                    fig, axes = plt.subplots(
                        n_rows,
                        n_cols,
                        figsize=(6.8 * n_cols, 5.8 * n_rows),
                        squeeze=False,
                    )
                    flat_axes = axes.flatten()

                    for idx, (var_x, var_y) in enumerate(chunk):
                        ax = flat_axes[idx]
                        col_x = f"value_{var_x}"
                        col_y = f"value_{var_y}"

                        for run_idx, (run_label, merged) in pair_data[(var_x, var_y)].items():
                            run_color = palette[run_idx % len(palette)]

                            # Trajectory points halved to s=32
                            ax.scatter(
                                merged[col_x],
                                merged[col_y],
                                color=run_color,
                                marker=".",
                                s=32,
                                alpha=0.8,
                                zorder=2,
                                label=run_label if idx == 0 else None,
                            )

                            # Final epoch marker halved to s=60
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

                        label_x = var_x.replace("_", " ").capitalize()
                        label_y = var_y.replace("_", " ").capitalize()
                        ax.set_title(f"{label_y} vs {label_x}", fontsize=12, pad=10)
                        ax.set_xlabel(label_x, fontsize=10)
                        ax.set_ylabel(label_y, fontsize=10)
                        ax.margins(x=0.08, y=0.08)

                    # Hide unused axes in non-full grids (e.g. 5 plots in a 3x2)
                    for empty_idx in range(n_plots, len(flat_axes)):
                        flat_axes[empty_idx].set_visible(False)

                    handles, labels = flat_axes[0].get_legend_handles_labels()
                    if handles:
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

                    formatted_anchor = anchor_name.replace("_", " ").capitalize()
                    fig.suptitle(
                        f"{title_prefix} — Relationships vs {formatted_anchor}",
                        fontsize=13,
                        y=1.02,
                    )

                    part_suffix = f"_part{chunk_idx + 1}" if total_chunks > 1 else ""
                    clean_file_stem = file_stem.replace("/", "_")
                    clean_anchor = anchor_name.replace("/", "_")
                    out_path = (
                        dest_dir
                        / f"ScatterPlot_relationships_{split_name}_{clean_file_stem}_vs_{clean_anchor}{part_suffix}.png"
                    )

                    plt.tight_layout()
                    plt.savefig(out_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    saved_plots.append(out_path)

            # Group multi-run comparisons by anchor 
            comparison_by_anchor: dict[str, list[tuple[str, str]]] = {}
            for var_x, var_y in comparison_pairs:
                comparison_by_anchor.setdefault(var_x, []).append((var_x, var_y))

            for anchor, pairs in comparison_by_anchor.items():
                _render_anchor_group(
                    pairs_to_plot=pairs,
                    title_prefix="Comparison",
                    file_stem="comparison",
                    anchor_name=anchor,
                    is_comparison=True,
                )

            # Group single-run metrics by run and anchor
            for run_idx, r_pairs in single_run_pairs.items():
                run_label = metrics_list[run_idx].experiment_name or f"run_{run_idx}"
                single_by_anchor: dict[str, list[tuple[str, str]]] = {}
                for var_x, var_y in r_pairs:
                    single_by_anchor.setdefault(var_x, []).append((var_x, var_y))

                for anchor, pairs in single_by_anchor.items():
                    _render_anchor_group(
                        pairs_to_plot=pairs,
                        title_prefix=f"Experiment {run_label}",
                        file_stem=run_label,
                        anchor_name=anchor,
                        is_comparison=False,
                    )

        return saved_plots