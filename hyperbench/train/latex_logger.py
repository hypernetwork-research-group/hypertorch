from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, TypedDict, Mapping, Union
from typing_extensions import NotRequired

from lightning.pytorch.loggers import Logger
from torch_geometric import metrics

from hyperbench import train


def collect_metric_bounds(
    results: Mapping[str, Mapping[str, Any]],
    metric_names: list[str],
) -> dict[str, tuple[float, float]]:
    metric_bounds: dict[str, tuple[float, float]] = {}

    for metric in metric_names:
        values = [
            float(value)
            for model_metrics in results.values()
            for key, value in model_metrics.items()
            if key == metric and isinstance(value, (int, float))
        ]
        if values:
            metric_bounds[metric] = (min(values), max(values))

    return metric_bounds


def colorize_metric_value(
    metric: str,
    value: float,
    text: str,
    metric_bounds: Mapping[str, tuple[float, float]] | None,
    sort_order: str,
) -> str:
    bounds = None if metric_bounds is None else metric_bounds.get(metric)
    if bounds is None:
        return text

    v_min, v_max = bounds

    if v_max == v_min:
        quality = 1.0
    else:
        t = (value - v_min) / (v_max - v_min)  # 0..1, low->high
        quality = (1.0 - t) if sort_order.lower() == "asc" else t

    red = int(round((1.0 - quality) * 100))
    green = int(round(quality * 100))

    # Blend toward white to keep the gradient readable but less bright.
    soften = 0.35
    red_chan = int(round(((red / 100) * (1.0 - soften) + soften) * 255))
    green_chan = int(round(((green / 100) * (1.0 - soften) + soften) * 255))
    blue_chan = int(round(soften * 255))

    return rf"\cellcolor[HTML]{{{red_chan:02X}{green_chan:02X}{blue_chan:02X}}}{text}"


class LaTexTableConfig(TypedDict):
    """
    Configuration for the LaTex table logger.

    Args:
        table_caption: Caption for the LaTex table.
        sort_by: Per-column sorting criteria ("asc" or "des").
        border: Whether to include borders in the LaTex table.
    """

    table_caption: NotRequired[str]
    sort_by: NotRequired[list[str]]
    border: NotRequired[bool]


class LaTexTableLogger(Logger):
    # TODO
    # - settings has to be configurable in Trainer

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
    __shared_stores: ClassVar[Dict[str, Dict[str, Dict[str, Any]]]] = {}

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
        d: LaTexTableConfig = {}
        self.__options = options if options is not None else d
        if experiment_name not in self.__shared_stores:
            self.__shared_stores[experiment_name] = {}

    @property
    def name(self) -> str:
        return "LaTexTableLogger"

    @property
    def version(self) -> str | int:
        return self.__model_name

    @property
    def store(self) -> Dict[str, Dict[str, Any]]:
        """Access the shared store for the current experiment."""
        return dict(self.__shared_stores.get(self.__experiment_name, {}))

    @property
    def save_dir(self) -> Union[str, Path]:
        return self.__save_dir

    @property
    def experiment_name(self) -> Union[str, Path]:
        return self.__experiment_name

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Accumulate metrics for this model. Called by Lightning on every log step.

        Keeps only the latest value for each metric name. For example, if
        "val_auc" is logged at step 10 and step 20, only the step 20 value is kept.
        """
        store = self.__shared_stores[self.__experiment_name]
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
        table_caption_opt = self.__options.get("table_caption")
        sort_by_opt = self.__options.get("sort_by")
        border_opt = self.__options.get("border")

        table_caption = table_caption_opt if isinstance(table_caption_opt, str) else None
        sort_by = sort_by_opt if isinstance(sort_by_opt, list) and sort_by_opt else ["asc"]
        border = border_opt if isinstance(border_opt, bool) else True

        self.__save_comparison_tables(
            test_results=test_results,
            save_dir=comparison_dir,
            train_results=train_results if train_results else None,
            val_results=val_results if val_results else None,
            precision=self.__precision,
            table_caption=table_caption,
            sort_by=sort_by,
            border=border,
        )
        self.__save_comparison_tables(
            test_results=test_results,
            save_dir=comparison_dir,
            train_results=None,
            val_results=None,
            precision=self.__precision,
            filename=f"test.tex",
            table_caption=table_caption,
            sort_by=sort_by,
            border=border,
        )

    def __split_results(
        self,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Split all accumulated metrics into test vs train/val groups.

        Metrics are classified by their name prefix:
        - "test_*"  --> test_results
        - "train_*" --> train_results
        - "val_*" --> val_results
        - anything else (e.g., "epoch") --> ignored
        """
        store = self.__shared_stores.get(self.__experiment_name, {})
        test_results: Dict[str, Dict[str, Any]] = {}
        train_results: Dict[str, Dict[str, Any]] = {}
        val_results: Dict[str, Dict[str, Any]] = {}

        for model_name, metrics in store.items():
            test_metrics: Dict[str, Any] = {}
            train_metrics: Dict[str, Any] = {}
            val_metrics: Dict[str, Any] = {}

            for metric_name, value in metrics.items():
                if metric_name.startswith("test"):
                    test_metrics[metric_name] = value
                elif metric_name.startswith("train"):
                    train_metrics[metric_name] = value
                elif metric_name.startswith("val"):
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
        sections_data: list[tuple[str, Mapping[str, Mapping[str, Any]]]],
        precision: int = 4,
        table_caption: Optional[str] = None,
        sort_by: list[str] | None = None,
        border: bool = True,
    ) -> str:
        if not sections_data:
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

        def section_lines(
            title: str,
            results: Mapping[str, Mapping[str, Any]],
            total_cols: int,
        ) -> list[str]:
            metrics = sorted({m for mm in results.values() for m in mm})
            sort_orders = sort_by if sort_by else ["asc"]

            normalized_orders: list[str] = []
            for order in sort_orders:
                normalized = order.lower()
                if normalized not in ("asc", "des"):
                    raise ValueError(f"Invalid sort_by value: {order}. Use 'asc' or 'des'.")
                normalized_orders.append(normalized)

            metric_sort: dict[str, str] = {}
            for idx, metric in enumerate(metrics):
                metric_sort[metric] = (
                    normalized_orders[idx]
                    if idx < len(normalized_orders)
                    else normalized_orders[-1]
                )

            metric_bounds = collect_metric_bounds(results, metrics)

            best_by_metric: dict[str, float] = {}

            for metric in metrics:
                vals = [
                    v
                    for model_metrics in results.values()
                    for k, v in model_metrics.items()
                    if k == metric and isinstance(v, (int, float))
                ]
                if vals:
                    best_by_metric[metric] = (
                        min(vals) if metric_sort.get(metric, "asc") == "asc" else max(vals)
                    )

            header_cells = ["Model", *[esc(m) for m in metrics]]
            while len(header_cells) < total_cols:
                header_cells.append("")

            lines = [
                r"\addlinespace[3pt]",
                rf"\multicolumn{{{total_cols}}}{{c}}{{\textbf{{{esc(title)}}}}} \\",
                r"\midrule",
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

                        row.append(
                            colorize_metric_value(
                                metric=metric,
                                value=float(value),
                                text=formatted,
                                metric_bounds=metric_bounds,
                                sort_order=metric_sort.get(metric, "asc"),
                            )
                        )
                    else:
                        row.append("-")

                while len(row) < total_cols:
                    row.append("")
                lines.append(" & ".join(row) + r" \\")

            lines.append(r"\hline" if border else r"\midrule")
            return lines

        # One tabular must have fixed column count; use max needed across sections.
        max_metrics = max(len({m for mm in rs.values() for m in mm}) for _, rs in sections_data)
        total_cols = 1 + max_metrics
        if border:
            col_spec = "|".join(["l", *(["c"] * (total_cols - 1))])
        else:
            col_spec = "l" + "c" * (total_cols - 1)

        lines: list[str] = [
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline" if border else r"\toprule",
        ]

        for title, results in sections_data:
            lines.extend(section_lines(title, results, total_cols))

        # Replace the last section-ending rule with a final closing \hline.
        last_rule = r"\hline" if border else r"\midrule"
        final_rule = r"\hline"
        lines and lines[-1] == last_rule and lines.pop()
        lines and lines[-1] == final_rule or lines.append(final_rule)

        lines.append(r"\end{tabular}")
        table_lines: list[str] = [r"\begin{table}[htbp]", r"\centering"]

        if table_caption:
            table_lines.append(rf"\caption{{{esc(table_caption)}}}")

        table_lines.extend(lines)
        table_lines.append(r"\end{table}")
        return "\n".join(table_lines) + "\n"

    def __save_comparison_tables(
        self,
        test_results: Mapping[str, Mapping[str, Any]],
        save_dir: Union[str, Path],
        train_results: Mapping[str, Mapping[str, Any]] | None = None,
        val_results: Mapping[str, Mapping[str, Any]] | None = None,
        filename: str = "overall.tex",
        precision: int = 4,
        table_caption: str | None = None,
        sort_by: list[str] | None = None,
        border: bool = True,
    ) -> Path:
        sections_data: list[tuple[str, Mapping[str, Mapping[str, Any]]]] = []
        if test_results:
            sections_data.append(("Test Results", test_results))
        if train_results:
            sections_data.append(("Train Results", train_results))
        if val_results:
            sections_data.append(("Val Results", val_results))

        content = self.__build_comparison_table(
            sections_data=sections_data,
            precision=precision,
            border=border,
            table_caption=table_caption,
            sort_by=sort_by,
        )

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        file_path = save_path / filename
        if content != "":
            content = (
                "% Requires: \\usepackage{booktabs}\n"
                "% Requires: \\usepackage[table]{xcolor}\n" + content
            )
        file_path.write_text(content)
        return file_path
