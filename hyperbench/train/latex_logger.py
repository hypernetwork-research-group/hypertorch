from pathlib import Path
from typing import Any, ClassVar, TypedDict
from collections.abc import Mapping
from typing_extensions import NotRequired
from hyperbench.utils import LATEX_CHARACTER_ESCAPE_TABLE, escape, validate_is_non_negative
from lightning.pytorch.loggers import Logger


def collect_metric_bounds(
    results: Mapping[str, Mapping[str, Any]],
    metric_names: list[str],
) -> dict[str, tuple[float, float]]:
    """
    Collect minimum and maximum values for numeric metrics.

    Args:
        results: Mapping from model names to metric dictionaries.
        metric_names: Metric names to inspect.

    Returns:
        metric_bounds: Mapping from metric name to ``(min, max)`` bounds.
    """
    metric_bounds: dict[str, tuple[float, float]] = {}

    for metric_name in metric_names:
        values = [
            float(orther_metric_value)
            for model_metrics in results.values()
            for other_metric_name, orther_metric_value in model_metrics.items()
            if other_metric_name == metric_name and isinstance(orther_metric_value, (int, float))
        ]
        if values:
            metric_bounds[metric_name] = (min(values), max(values))

    return metric_bounds


def colorize_metric_value(
    metric: str,
    value: float,
    text: str,
    metric_bounds: Mapping[str, tuple[float, float]] | None,
    sort_order: str,
) -> str:
    """
    Wrap a formatted metric value in a LaTex cell color command.

    Args:
        metric: Metric name.
        value: Numeric metric value.
        text: Already formatted metric text.
        metric_bounds: Optional metric bounds used for color scaling.
        sort_order: ``"asc"`` when lower is better, or ``"des"`` when higher is better.

    Returns:
        text: Original or colorized LaTex cell text.

    Raises:
        ValueError: If ``sort_order`` is unsupported.
    """
    bounds = None if metric_bounds is None else metric_bounds.get(metric)
    if bounds is None:
        return text

    normalized_sort_order = sort_order.lower()
    if normalized_sort_order not in ("asc", "des"):
        raise ValueError(f"'sort_order' must be 'asc' or 'des', got {sort_order!r}.")

    min_metric_value, max_metric_value = bounds
    if max_metric_value == min_metric_value:
        quality = 1.0
    else:
        normalized_metric_value = (value - min_metric_value) / (
            max_metric_value - min_metric_value
        )  # 0..1, low->high
        quality = (
            (1.0 - normalized_metric_value)
            if normalized_sort_order == "asc"
            else normalized_metric_value
        )

    red = round((1.0 - quality) * 100)
    green = round(quality * 100)

    # Blend toward white to keep the gradient readable but less bright.
    soften = 0.35
    red_chan = round(((red / 100) * (1.0 - soften) + soften) * 255)
    green_chan = round(((green / 100) * (1.0 - soften) + soften) * 255)
    blue_chan = round(soften * 255)

    return rf"\cellcolor[HTML]{{{red_chan:02X}{green_chan:02X}{blue_chan:02X}}}{text}"


class LaTexTableConfig(TypedDict):
    """
    Configuration for the LaTex table logger.

    Attributes:
        table_caption: Caption for the LaTex table.
        sort_by: Per-column sorting criteria ("asc" or "des").
        border: Whether to include borders in the LaTex table.
    """

    table_caption: NotRequired[str]
    sort_by: NotRequired[list[str]]
    border: NotRequired[bool]


class LaTexTableLogger(Logger):
    """A Lightning Logger that accumulates metrics and writes a LaTex comparison table.

    Multiple instances (one per model) share a class-level store keyed by experiment_name.
    Every time finalize() is called (after fit() or test() for each model), the current
    state of all accumulated metrics is written to a LaTex file. The last model to
    finalize produces the most complete table.

    This means the file is progressively updated as models finish training/testing,
    so you can open it mid-run to see partial results.

    Attributes:
        __save_dir: Base directory where the comparison subfolder will be created.
        __model_name: The model's full name.
        __experiment_name: Shared key grouping models in the same experiment.
        __precision: Decimal places for metric values in the table.
        __options: Table rendering options.
    """

    # Class-level shared store: {experiment_name: {model_name: {metric_name: value}}}
    __shared_stores: ClassVar[dict[str, dict[str, dict[str, Any]]]] = {}

    def __init__(
        self,
        save_dir: str | Path,
        model_name: str,
        experiment_name: str,
        precision: int = 4,
        options: LaTexTableConfig | None = None,
    ) -> None:
        """
        Initialize the LaTex table logger.

        Args:
            save_dir: Base directory where comparison files are written.
            model_name: Full model name to use as the table row label.
            experiment_name: Shared key grouping models in the same experiment.
            precision: Decimal places for metric values.
            options: Optional table rendering configuration.
                Defaults to an empty dict configuration, if not provided.
        """
        super().__init__()
        validate_is_non_negative("precision", precision)

        self.__save_dir = save_dir
        self.__model_name = model_name
        self.__experiment_name = experiment_name
        self.__precision = precision

        default_empty_options: LaTexTableConfig = {}
        self.__options = options if options is not None else default_empty_options

        if experiment_name not in self.__shared_stores:
            self.__shared_stores[experiment_name] = {}

    @property
    def name(self) -> str:
        """
        Return the logger name.

        Returns:
            name: Logger name.
        """
        return "LaTexTableLogger"

    @property
    def version(self) -> str | int:
        """
        Return the logger version.

        Returns:
            version: Model name used as the logger version.
        """
        return self.__model_name

    @property
    def store(self) -> dict[str, dict[str, Any]]:
        """Access the shared store for the current experiment."""
        return dict(self.__shared_stores.get(self.__experiment_name, {}))

    @property
    def save_dir(self) -> str | Path:
        """
        Return the logger save directory.

        Returns:
            save_dir: Base directory for comparison files.
        """
        return self.__save_dir

    @property
    def experiment_name(self) -> str | Path:
        """
        Return the experiment name.

        Returns:
            experiment_name: Shared experiment key.
        """
        return self.__experiment_name

    def clear(self, experiment_name: str) -> None:
        """
        Remove accumulated data for an experiment.

        Args:
            experiment_name: The experiment name whose data should be cleared.
        """
        self.__shared_stores.pop(experiment_name, None)

    def log_hyperparams(self, params: Any) -> None:
        """
        Accept hyperparameter logging calls from Lightning.

        Args:
            params: Hyperparameters provided by Lightning, unused.
        """
        pass

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Accumulate metrics for this model. Called by Lightning on every log step.

        Keeps only the latest value for each metric name. For example, if
        "val_auc" is logged at step 10 and step 20, only the step 20 value is kept.
        """
        store = self.__shared_stores[self.__experiment_name]
        if self.__model_name not in store:
            store[self.__model_name] = {}
        store[self.__model_name].update(metrics)

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
            train_results=train_results or None,
            val_results=val_results or None,
            precision=self.__precision,
            table_caption=table_caption,
            sort_by=sort_by,
            border=border,
        )
        # test
        self.__save_comparison_tables(
            test_results=test_results,
            save_dir=comparison_dir,
            train_results=None,
            val_results=None,
            precision=self.__precision,
            filename="test.tex",
            table_caption=table_caption,
            sort_by=sort_by,
            border=border,
        )
        # train
        self.__save_comparison_tables(
            test_results={},
            save_dir=comparison_dir,
            train_results=train_results or None,
            val_results=None,
            precision=self.__precision,
            filename="train.tex",
            table_caption=table_caption,
            sort_by=sort_by,
            border=border,
        )
        # val
        self.__save_comparison_tables(
            test_results={},
            save_dir=comparison_dir,
            train_results=None,
            val_results=val_results or None,
            precision=self.__precision,
            filename="val.tex",
            table_caption=table_caption,
            sort_by=sort_by,
            border=border,
        )

    def __build_comparison_table(
        self,
        sections_data: list[tuple[str, Mapping[str, Mapping[str, Any]]]],
        precision: int = 4,
        table_caption: str | None = None,
        sort_by: list[str] | None = None,
        border: bool = True,
    ) -> str:
        """
        Build a LaTex comparison table from grouped result sections.

        Args:
            sections_data: Section titles and per-model metric mappings.
            precision: Decimal places for metric values. Defaults to ``4``.
            table_caption: Optional table caption. Defaults to ``None``.
            sort_by: Optional per-metric sort directions. Defaults to ``None``.
            border: Whether to use bordered table rules. Defaults to ``True``.

        Returns:
            table: LaTex table content.
        """
        if not sections_data:
            return ""

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
            lines.extend(
                self.__get_section_lines(title, results, total_cols, precision, sort_by, border)
            )

        # Replace the last section-ending rule with a final closing \hline.
        last_rule = r"\hline" if border else r"\midrule"
        final_rule = r"\hline"
        lines and lines[-1] == last_rule and lines.pop()
        (lines and lines[-1] == final_rule) or lines.append(final_rule)

        lines.append(r"\end{tabular}")
        table_lines: list[str] = [r"\begin{table}[htbp]", r"\centering"]

        if table_caption:
            escaped_caption = escape(table_caption, LATEX_CHARACTER_ESCAPE_TABLE)
            table_lines.append(rf"\caption{{{escaped_caption}}}")

        table_lines.extend(lines)
        table_lines.append(r"\end{table}")
        return "\n".join(table_lines) + "\n"

    def __get_section_lines(
        self,
        title: str,
        results: Mapping[str, Mapping[str, Any]],
        total_cols: int,
        precision: int,
        sort_by: list[str] | None,
        border: bool,
    ) -> list[str]:
        """
        Build LaTex table rows for a result section.

        Args:
            title: Section title.
            results: Mapping from model names to metric dictionaries.
            total_cols: Total number of columns in the shared table.
            precision: Decimal places for metric values.
            sort_by: Optional per-metric sort directions.
            border: Whether to use bordered table rules.

        Returns:
            lines: LaTex rows for the section.

        Raises:
            ValueError: If any sort direction is unsupported.
        """
        metrics = sorted({m for mm in results.values() for m in mm})
        sort_orders = sort_by or ["asc"]

        normalized_orders: list[str] = []
        for order in sort_orders:
            normalized = order.lower()
            if normalized not in ("asc", "des"):
                raise ValueError(f"Invalid 'sort_by' value: {order}. Use 'asc' or 'des'.")
            normalized_orders.append(normalized)

        metric_sort: dict[str, str] = {}
        for idx, metric in enumerate(metrics):
            metric_sort[metric] = (
                normalized_orders[idx] if idx < len(normalized_orders) else normalized_orders[-1]
            )

        metric_bounds = collect_metric_bounds(results, metrics)

        best_by_metric: dict[str, float] = {}

        for metric in metrics:
            vals = [
                metric_value
                for model_metrics in results.values()
                for metric_name, metric_value in model_metrics.items()
                if metric_name == metric and isinstance(metric_value, (int, float))
            ]
            if vals:
                best_by_metric[metric] = (
                    min(vals) if metric_sort.get(metric, "asc") == "asc" else max(vals)
                )

        header_cells = [
            "Model",
            *[escape(metric, LATEX_CHARACTER_ESCAPE_TABLE) for metric in metrics],
        ]
        while len(header_cells) < total_cols:
            header_cells.append("")

        escaped_title = escape(title, LATEX_CHARACTER_ESCAPE_TABLE)
        lines = [
            r"\addlinespace[3pt]",
            rf"\multicolumn{{{total_cols}}}{{c}}{{\textbf{{{escaped_title}}}}} \\",
            r"\midrule",
            " & ".join(header_cells) + r" \\",
        ]

        for model_name in sorted(results):
            model_metrics = results[model_name]
            row = [escape(model_name, LATEX_CHARACTER_ESCAPE_TABLE)]

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

    def __save_comparison_tables(
        self,
        test_results: Mapping[str, Mapping[str, Any]],
        save_dir: str | Path,
        train_results: Mapping[str, Mapping[str, Any]] | None = None,
        val_results: Mapping[str, Mapping[str, Any]] | None = None,
        filename: str = "overall.tex",
        precision: int = 4,
        table_caption: str | None = None,
        sort_by: list[str] | None = None,
        border: bool = True,
    ) -> Path:
        """
        Build and save LaTex comparison tables to a file.

        Args:
            test_results: Mapping from model names to test metrics.
            save_dir: Directory where the LaTex file will be written.
            train_results: Optional mapping from model names to train metrics.
                Defaults to ``None``.
            val_results: Optional mapping from model names to validation metrics.
                Defaults to ``None``.
            filename: Name of the output file. Defaults to ``"overall.tex"``.
            precision: Decimal places for metric values. Defaults to ``4``.
            table_caption: Optional table caption. Defaults to ``None``.
            sort_by: Optional per-metric sort directions. Defaults to ``None``.
            border: Whether to use bordered table rules. Defaults to ``True``.

        Returns:
            path: Path to the written file.
        """
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

    def __split_results(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """
        Split all accumulated metrics into test vs train/val groups.

        Metrics are classified by their name prefix:
        - "test/*"  -> test_results
        - "train/*" -> train_results
        - "val/*"   -> val_results
        - anything else (e.g., "epoch") -> ignored
        """
        store = self.__shared_stores.get(self.__experiment_name, {})
        test_results: dict[str, dict[str, Any]] = {}
        train_results: dict[str, dict[str, Any]] = {}
        val_results: dict[str, dict[str, Any]] = {}

        for model_name, metrics in store.items():
            test_metrics: dict[str, Any] = {}
            train_metrics: dict[str, Any] = {}
            val_metrics: dict[str, Any] = {}

            for metric_name, value in metrics.items():
                if metric_name.startswith("test/"):
                    test_metrics[metric_name] = value
                elif metric_name.startswith("train/"):
                    train_metrics[metric_name] = value
                elif metric_name.startswith("val/"):
                    val_metrics[metric_name] = value

            if test_metrics:
                test_results[model_name] = test_metrics
            if train_metrics:
                train_results[model_name] = train_metrics
            if val_metrics:
                val_results[model_name] = val_metrics

        return test_results, train_results, val_results
