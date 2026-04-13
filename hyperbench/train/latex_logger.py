from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, TypedDict
from typing_extensions import NotRequired

from lightning.pytorch.loggers import Logger

from pathlib import Path
from typing import Mapping, Union

from torch_geometric import metrics

from hyperbench import train


class LaTexTableConfig(TypedDict):
    """
    Configuration for the LaTex table logger.

    Args:
        table_caption: Caption for the LaTex table.
        sort_by: Sorting criterion for the table rows.
        border: Whether to include borders in the LaTex table.
    """

    table_caption: NotRequired[str]
    sort_by: NotRequired[str]
    border: NotRequired[bool]


class LaTexTableLogger(Logger):
    # TODO
    # - settings has to be configurable in Trainer
    # - best results in tests
    # - scala colori

    """A Lightning Logger that accumulates metrics and writes a LaTex comparison table.

    Multiple instances (one per model) share a class-level store keyed by experiment_name.
    Every time finalize() is called (after fit() or test() for each model), the current
    state of all accumulated metrics is written to a LaTex file. The last model to
    finalize produces the most complete table.

    This means the file is progressively updated as models finish training/testing,
    so you can open it mid-run to see partial results.

    Args:
        save_dir: Base directory where the comparison/ subfolder will be created.
        model_name: The model's full name (e.g., "mlp:mean").
        experiment_name: Shared key that groups all models in the same experiment.
        precision: Decimal places for metric values in the table.
    """

    # Class-level shared store: {experiment_name: {model_name: {metric_name: value}}}
    __shared_stores: ClassVar[Dict[str, Dict[str, Dict[str, float]]]] = {}

    def __init__(
        self,
        save_dir: str | Path,
        model_name: str,
        experiment_name: str,
        precision: int = 4,
        options: LaTexTableConfig | None = None,
    ) -> None:
        super().__init__()
        self.__save_dir = save_dir
        self.__model_name = model_name
        self.__experiment_name = experiment_name
        self.__precision = precision
        d: LaTexTableConfig = {
            "table_caption": f"Results for Experiments",
            "sort_by": "asc",
            "border": True,
        }
        self.__options = options if options is not None else d
        if experiment_name not in LaTexTableLogger.__shared_stores:
            LaTexTableLogger.__shared_stores[experiment_name] = {}

    @property
    def name(self) -> str:
        return "LaTexTableLogger"

    @property
    def version(self) -> str | int:
        return self.__model_name

    @property
    def store(self) -> Dict[str, Dict[str, float]]:
        """Access the shared store for the current experiment."""
        return dict(LaTexTableLogger.__shared_stores.get(self.__experiment_name, {}))

    @property
    def save_dir(self) -> Union[str, Path]:
        return self.__save_dir

    @property
    def experiment_name(self) -> Union[str, Path]:
        return self.__experiment_name

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Accumulate metrics for this model. Called by Lightning on every log step.

        Keeps only the latest value for each metric name. For example, if
        "val_auc" is logged at step 10 and step 20, only the step 20 value is kept.
        """
        store = LaTexTableLogger.__shared_stores[self.__experiment_name]
        if self.__model_name not in store:
            store[self.__model_name] = {}
        store[self.__model_name].update(metrics)

    def log_hyperparams(self, params: Any) -> None:
        pass

    def finalize(self, status: str) -> None:
        """Write the LaTex comparison table with all accumulated metrics so far.

        Called by Lightning after fit() and after test() for each model. Since models
        train/test sequentially, each finalize() overwrites the file with all data
        accumulated up to that point. The file grows more complete over time.
        """
        test_results, train_results, val_results = self.__split_results()

        if not test_results and not train_results and not val_results:
            return

        comparison_dir = Path(self.__save_dir) / "comparison"
        self.__save_comparison_tables(
            test_results=test_results,
            save_dir=comparison_dir,
            train_results=train_results if train_results else None,
            val_results=val_results if val_results else None,
            precision=self.__precision,
            table_caption=self.__options.get("table_caption", "Results for Experiments"),
            sort_by=self.__options.get("sort_by", "asc"),
            border=self.__options.get("border", True),
        )
        self.__save_comparison_tables(
            test_results=test_results,
            save_dir=comparison_dir,
            train_results=None,
            val_results=None,
            precision=self.__precision,
            filename=f"test.tex",
            table_caption=self.__options.get("table_caption", "Results for Experiments"),
            sort_by=self.__options.get("sort_by", "asc"),
            border=self.__options.get("border", True),
        )

    def __split_results(
        self,
    ) -> Tuple[
        Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]
    ]:
        """Split all accumulated metrics into test vs train/val groups.

        Metrics are classified by their name prefix:
        - "test_*"  --> test_results
        - "train_*" --> train_results
        - "val_*" --> val_results
        - anything else (e.g., "epoch") --> ignored
        """
        store = self.__shared_stores.get(self.__experiment_name, {})
        test_results: Dict[str, Dict[str, float]] = {}
        train_results: Dict[str, Dict[str, float]] = {}
        val_results: Dict[str, Dict[str, float]] = {}

        for model_name, metrics in store.items():
            test_metrics: Dict[str, float] = {}
            train_metrics: Dict[str, float] = {}
            val_metrics: Dict[str, float] = {}

            for metric_name, value in metrics.items():
                if metric_name.startswith("test_"):
                    test_metrics[metric_name] = value
                elif metric_name.startswith("train_"):
                    train_metrics[metric_name] = value
                elif metric_name.startswith("val_"):
                    val_metrics[metric_name] = value

            if test_metrics:
                test_results[model_name] = test_metrics
            if train_metrics:
                train_results[model_name] = train_metrics
            if val_metrics:
                val_results[model_name] = val_metrics

        return test_results, train_results, val_results

    def clear(self, experiment_name: str) -> None:
        """Remove accumulated data for an experiment."""
        self.__shared_stores.pop(experiment_name, None)

    def __build_comparison_table(
        self,
        results: Mapping[str, Mapping[str, float]],
        precision: int = 4,
        title: str | None = None,
    ) -> str:
        if not results:
            return ""

        def esc(value: str) -> str:
            return (
                value.replace("\\", "\\textbackslash{}")
                .replace("&", "\\&")
                .replace("%", "\\%")
                .replace("$", "\\$")
                .replace("#", "\\#")
                .replace("_", "\\_")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("~", "\\textasciitilde{}")
                .replace("^", "\\textasciicircum{}")
            )

        all_metrics = sorted(
            {metric for model_metrics in results.values() for metric in model_metrics}
        )

        num_cols = 1 + len(all_metrics)
        col_spec = "l" + "c" * len(all_metrics)

        header_cells = "Model"
        if all_metrics:
            header_cells += " & " + " & ".join(esc(metric) for metric in all_metrics)
        header_line = header_cells + r" \\"

        rows = []
        for model_name in sorted(results):
            model_metrics = results[model_name]
            cells = [esc(model_name)]
            for metric in all_metrics:
                value = model_metrics.get(metric)
                if isinstance(value, (int, float)):
                    cells.append(f"{value:.{precision}f}")
                else:
                    cells.append("-")
            rows.append(" & ".join(cells) + r" \\")

        lines = [rf"\begin{{tabular}}{{{col_spec}}}", r"\hline"]

        if title:
            title_line = rf"\multicolumn{{{num_cols}}}{{c}}{{\textbf{{{esc(title)}}}}} \\"
            lines.extend([title_line, r"\hline"])

        lines.extend(
            [
                header_line,
                r"\hline",
                *rows,
                r"\hline",
                r"\end{tabular}",
            ]
        )

        return "\n".join(lines)

    def __save_comparison_tables(
        self,
        test_results: Mapping[str, Mapping[str, float]],
        save_dir: Union[str, Path],
        train_results: Mapping[str, Mapping[str, float]] | None = None,
        val_results: Mapping[str, Mapping[str, float]] | None = None,
        filename: str = "overall.tex",
        precision: int = 4,
        table_caption: str | None = None,
        sort_by: str | None = "asc",
        border: bool = True,
    ) -> Path:
        def esc(value: str) -> str:
            return (
                value.replace("\\", "\\textbackslash{}")
                .replace("&", "\\&")
                .replace("%", "\\%")
                .replace("$", "\\$")
                .replace("#", "\\#")
                .replace("_", "\\_")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("~", "\\textasciitilde{}")
                .replace("^", "\\textasciicircum{}")
            )

        def section_lines(
            title: str,
            results: Mapping[str, Mapping[str, float]],
            total_cols: int,
        ) -> list[str]:
            if not results:
                return []

            normalized_sort = sort_by.lower() if sort_by is not None else None

            if normalized_sort not in (None, "asc", "des"):
                raise ValueError(f"Invalid sort_by value: {sort_by}. Use 'asc', 'des', or None.")
            if normalized_sort is None:
                normalized_sort = "asc"

            metrics = sorted({m for mm in results.values() for m in mm})

            # Rank map per metric so coloring is column-wise and independent
            # from raw metric scale.
            metric_rank: dict[str, dict[float, float]] = {}

            for metric in metrics:
                vals = [
                    float(v)
                    for model_metrics in results.values()
                    for k, v in model_metrics.items()
                    if k == metric and isinstance(v, (int, float))
                ]
                if vals:
                    unique_vals = sorted(set(vals))
                    if len(unique_vals) == 1:
                        metric_rank[metric] = {unique_vals[0]: 0.5}
                    else:
                        denom = len(unique_vals) - 1
                        metric_rank[metric] = {
                            value: idx / denom for idx, value in enumerate(unique_vals)
                        }

            def colorize_value(metric: str, value: float, text: str) -> str:
                rank_map = metric_rank.get(metric)
                if rank_map is None:
                    return text

                t = rank_map.get(value, 0.5)
                if normalized_sort == "asc":
                    # Lower value is better: low rank => green, high rank => red.
                    green = int(round((1.0 - t) * 100))
                    red = int(round(t * 100))
                else:
                    # Higher value is better: high rank => green, low rank => red.
                    green = int(round(t * 100))
                    red = int(round((1.0 - t) * 100))

                # Use explicit RGB to avoid backend-dependent color mixing quirks.
                red_rgb = red / 100.0
                green_rgb = green / 100.0
                return rf"\cellcolor[rgb]{{{red_rgb:.3f},{green_rgb:.3f},0.000}}{text}"

            best_by_metric: dict[str, float] = {}

            for metric in metrics:
                vals = [
                    v
                    for model_metrics in results.values()
                    for k, v in model_metrics.items()
                    if k == metric and isinstance(v, (int, float))
                ]
                if vals:
                    best_by_metric[metric] = min(vals) if normalized_sort == "asc" else max(vals)

            header_cells = ["Model", *[esc(m) for m in metrics]]
            while len(header_cells) < total_cols:
                header_cells.append("")

            lines = [
                rf"\multicolumn{{{total_cols}}}{{c}}{{\textbf{{{esc(title)}}}}} \\",
                " & ".join(header_cells) + r" \\",
            ]

            for model_name in sorted(results):
                model_metrics = results[model_name]
                row = [esc(model_name)]

                for metric in metrics:
                    value = model_metrics.get(metric)
                    if isinstance(value, (int, float)):
                        formatted = f"{value:.{precision}f}"
                        best = best_by_metric.get(metric)
                        if best is not None and value == best:
                            formatted = rf"\underline{{{formatted}}}"

                        row.append(colorize_value(metric, float(value), formatted))
                    else:
                        row.append("-")

                while len(row) < total_cols:
                    row.append("")
                lines.append(" & ".join(row) + r" \\")

            lines.append(r"\hline" if border else r"\midrule")
            return lines

        sections_data: list[tuple[str, Mapping[str, Mapping[str, float]]]] = []
        if test_results:
            sections_data.append(("Test Results", test_results))
        if train_results:
            sections_data.append(("Train Results", train_results))
        if val_results:
            sections_data.append(("Val Results", val_results))

        if not sections_data:
            content = ""
        else:
            # One tabular must have fixed column count; use max needed across sections.
            max_metrics = max(len({m for mm in rs.values() for m in mm}) for _, rs in sections_data)
            total_cols = 1 + max_metrics
            if border:
                col_spec = "|" + "|".join(["l", *(["c"] * (total_cols - 1))]) + "|"
            else:
                col_spec = "l" + "c" * (total_cols - 1)

            lines: list[str] = [
                rf"\begin{{tabular}}{{{col_spec}}}",
                r"\hline" if border else r"\toprule",
            ]

            for title, results in sections_data:
                lines.extend(section_lines(title, results, total_cols))

            # Replace last section-ending \midrule with \bottomrule
            last_rule = r"\hline" if border else r"\midrule"
            final_rule = r"\hline" if border else r"\bottomrule"
            if lines[-1] == last_rule:
                lines[-1] = final_rule
            else:
                lines.append(final_rule)

            lines.append(r"\end{tabular}")
            table_lines: list[str] = [r"\begin{table}[htbp]", r"\centering"]

            if table_caption:
                table_lines.append(rf"\caption{{{esc(table_caption)}}}")

            table_lines.extend(lines)
            table_lines.append(r"\end{table}")

            content = "\n".join(table_lines) + "\n"

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        file_path = save_path / filename
        content = (
            "% Requires: \\usepackage{booktabs}\n"
            "% Requires: \\usepackage[table]{xcolor}\n" + content
        )
        file_path.write_text(content)
        return file_path
