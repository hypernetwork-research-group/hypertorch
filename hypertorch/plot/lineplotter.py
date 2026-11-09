from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from hypertorch.plot.plotter import Plotter
from hypertorch.types.parsed_metrics import ParsedMetrics


class LinePlotter(Plotter):
    """Generates Seaborn line plots for training and evaluation metrics."""

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """
        Render and save line plots across epochs or training steps.

        Args:
            metrics: Single ParsedMetrics or sequence of metrics to plot.
            metric_names: Optional subset of metric variables to plot.
            output_dir: Optional destination path for plot outputs.
            create_subfolder: Whether to save inside a 'plots' subfolder.

        Returns:
            saved_plots: List of file paths to saved line plot PNG files.
        """
        plt, sns = self._init_plotting()
        metrics_list = self._normalize_metrics(metrics)
        self._validate_metrics_list(metrics_list)

        plots_dir = self._resolve_plots_dir(metrics_list, output_dir, create_subfolder)
        targets = self._resolve_target_metrics(metrics_list, metric_names)
        x_col = metrics_list[0].x_col
        palette = sns.color_palette("colorblind")

        saved_plots: list[Path] = []
        for var_name in targets:
            runs_with_metric = [(idx, m) for idx, m in enumerate(metrics_list) if var_name in m]
            if not runs_with_metric:
                continue

            plot_path = self._render_metric_plot(
                var_name=var_name,
                runs_with_metric=runs_with_metric,
                metrics_list=metrics_list,
                plots_dir=plots_dir,
                x_col=x_col,
                palette=palette,
                plt=plt,
                create_subfolder=create_subfolder,
            )
            saved_plots.append(plot_path)

        return saved_plots

    @staticmethod
    def _resolve_curve_styling(total_curves: int) -> tuple[float, float]:
        """Calculate dynamic linewidth and markersize scaled down by curve count."""
        if total_curves <= 1:
            return 1.6, 4.0
        linewidth = max(0.9, 1.6 - (total_curves - 1) * 0.06)
        markersize = max(2.2, 4.0 - (total_curves - 1) * 0.16)
        return round(linewidth, 2), round(markersize, 2)

    @staticmethod
    def _get_split_marker(split_name: str) -> str:
        """Resolve point marker character based on split category."""
        clean_split = split_name.lower()
        if "train" in clean_split:
            return "o"
        if "val" in clean_split:
            return "s"
        return "d"

    @staticmethod
    def _draw_singular(ax: Any, val: float, color: Any, zorder: int, label: str) -> None:
        """Render a horizontal reference line for single-point evaluations."""
        ax.axhline(
            y=val,
            color=color,
            linestyle=":",
            linewidth=1.3,
            alpha=0.8,
            zorder=zorder,
            label=f"{label}: {val:.4f}",
        )

    @staticmethod
    def _draw_curve(
        ax: Any,
        group: pd.DataFrame,
        x_col: str,
        color: Any,
        marker: str | None,
        zorder: int,
        label: str,
        linewidth: float,
        markersize: float,
    ) -> None:
        """Render a continuous line progression across training epochs or steps."""
        sorted_group = group.sort_values(by=x_col)
        ax.plot(
            sorted_group[x_col],
            sorted_group["value"],
            color=color,
            linestyle="-",
            linewidth=linewidth,
            marker=marker,
            markersize=markersize if marker else None,
            markeredgewidth=0.5 if marker else None,
            zorder=zorder,
            label=label,
        )

    def _render_metric_plot(
        self,
        var_name: str,
        runs_with_metric: list[tuple[int, ParsedMetrics]],
        metrics_list: list[ParsedMetrics],
        plots_dir: Path,
        x_col: str,
        palette: list[Any],
        plt: Any,
        create_subfolder: bool,
    ) -> Path:
        """Construct, render, and save an individual metric line plot to disk."""
        is_multi = len(runs_with_metric) > 1
        first_run_name = runs_with_metric[0][1].experiment_name or "0"
        title_prefix, suffix = self._resolve_run_naming(is_multi, first_run_name)

        dest_dir = self._resolve_dest_dir(
            plots_dir=plots_dir,
            is_multi=is_multi,
            total_runs=len(metrics_list),
            create_subfolder=create_subfolder,
        )

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

        linewidth, markersize = self._resolve_curve_styling(total_curves)

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

                if len(group) == 1:
                    val = float(group["value"].iloc[0])
                    self._draw_singular(
                        ax=ax,
                        val=val,
                        color=color,
                        zorder=base_zorder,
                        label=label,
                    )
                else:
                    marker = self._get_split_marker(split_name)
                    self._draw_curve(
                        ax=ax,
                        group=group,
                        x_col=x_col,
                        color=color,
                        marker=marker,
                        zorder=base_zorder + 1,
                        label=label,
                        linewidth=linewidth,
                        markersize=markersize,
                    )

        formatted_metric = self._format_label(var_name)
        ax.set_title(f"{title_prefix} — {formatted_metric}", fontsize=12, pad=10)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel(formatted_metric)

        legend_title = "Runs & Splits" if is_multi else "Split"
        if is_busy:
            ax.legend(
                title=legend_title,
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                borderaxespad=0.0,
                frameon=True,
            )
        else:
            ax.legend(title=legend_title, loc="best", framealpha=0.85)

        clean_var = self._sanitize(var_name)
        output_path = dest_dir / f"LinePlot_{clean_var}_{suffix}.png"

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path
