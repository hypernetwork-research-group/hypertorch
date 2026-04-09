from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

from lightning.pytorch.loggers import Logger

from pathlib import Path
from typing import Mapping, Union

from hyperbench import train


class LaTexTableLogger(Logger):
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
    ) -> None:
        super().__init__()
        self.__save_dir = save_dir
        self.__model_name = model_name
        self.__experiment_name = experiment_name
        self.__precision = precision

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

        lines = [
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline",
            header_line,
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
        ]
        return "\n".join(lines)

    def __save_comparison_tables(
        self,
        test_results: Mapping[str, Mapping[str, float]],
        save_dir: Union[str, Path],
        train_results: Mapping[str, Mapping[str, float]] | None = None,
        val_results: Mapping[str, Mapping[str, float]] | None = None,
        filename: str = "results.tex",
        precision: int = 4,
    ) -> Path:
        sections = []

        test_table = self.__build_comparison_table(test_results, precision)
        if test_table:
            sections.append("\\section*{Test Results}\n" + test_table)

        if train_results or val_results:
            if train_results:
                train_table = self.__build_comparison_table(train_results, precision)
                if train_table:
                    sections.append("\\section*{Train Results}\n" + train_table)
            if val_results:
                val_table = self.__build_comparison_table(val_results, precision)
                if val_table:
                    sections.append("\\section*{Val Results}\n" + val_table)

        content = "\n\n".join(sections) + "\n" if sections else ""

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        file_path = save_path / filename
        file_path.write_text(content)

        return file_path
