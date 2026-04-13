from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple
import copy

from lightning.pytorch.loggers import Logger

from pathlib import Path
from typing import Mapping, Union

from hyperbench import train


class MarkdownTableLogger(Logger):
    """A Lightning Logger that accumulates metrics and writes a markdown comparison table.

    Multiple instances (one per model) share a class-level store keyed by experiment_name.
    Every time finalize() is called (after fit() or test() for each model), the current
    state of all accumulated metrics is written to a markdown file. The last model to
    finalize produces the most complete table.

    This means the file is progressively updated as models finish training/testing,
    so partial results are available while running.

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

        if experiment_name not in self.__shared_stores:
            self.__shared_stores[experiment_name] = {}

    @property
    def name(self) -> str:
        return "MarkdownTableLogger"

    @property
    def version(self) -> str | int:
        return self.__model_name

    @property
    def store(self) -> Dict[str, Dict[str, float]]:
        """Access the shared store for the current experiment."""
        return copy.deepcopy(self.__shared_stores.get(self.__experiment_name, {}))

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
        store = self.__shared_stores[self.__experiment_name]
        if self.__model_name not in store:
            store[self.__model_name] = {}
        store[self.__model_name].update(metrics)

    def log_hyperparams(self, params: Any) -> None:
        pass

    def finalize(self, status: str) -> None:
        """Write the markdown comparison table with all accumulated metrics so far.

        Called by Lightning after fit() and after test() for each model. Since models
        train/test sequentially, each finalize() overwrites the file with all data
        accumulated up to that point. The file grows more complete over time.

        Args:
            status: The stage that just completed, e.g., "fit" or "test".
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
        - "test*"  --> test_results
        - "train*" --> train_results
        - "val*" --> val_results
        - anything else (e.g., "epoch") --> ignored

        Returns:
            Tuple of (test_results, train_results, val_results), where each is a dict
            mapping model names to their respective metric dicts. Models with no metrics
            in a category are excluded from that category's dict.
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
        """Remove accumulated data for an experiment.
        Args:
            experiment_name: The experiment name whose data should be cleared.

        """
        self.__shared_stores.pop(experiment_name, None)

    def __build_comparison_table(
        self,
        results: Mapping[str, Mapping[str, float]],
        precision: int = 4,
    ) -> str:
        """Build a markdown comparison table from model results.

        Example:
            Input:

            ```python
            {
            "mlp:mean": {"test_auc": 0.85, "test_loss": 0.32},
            "gat:default": {"test_auc": 0.82},
            }
            ```

            Output:

            ```md
            | Model | test_auc | test_loss |
            | --- | --- | --- |
            | gat:default | 0.8200 | - |
            | mlp:mean | 0.8500 | 0.3200 |
            ```

        Args:
            results: Mapping of model names to metric dictionaries.
            precision: Number of decimal places for numeric metric values.

        Returns:
            Markdown table string. Returns an empty string if ``results`` is empty.


        """
        if not results:
            return ""

        # Collect all unique metric names across all models, sorted for determinism
        all_metrics = sorted(
            {metric for model_metrics in results.values() for metric in model_metrics}
        )

        # Build header row
        header = "| Model | " + " | ".join(all_metrics) + " |"
        separator = "| --- | " + " | ".join("---" for _ in all_metrics) + " |"

        # Build one row per model, sorted by model name for determinism
        rows = []
        for model_name in sorted(results):
            model_metrics = results[model_name]
            cells = []
            for metric in all_metrics:
                value = model_metrics.get(metric)
                if isinstance(value, (int, float)):
                    cells.append(f"{value:.{precision}f}")
                else:
                    cells.append("-")
            rows.append(f"| {model_name} | " + " | ".join(cells) + " |")

        return "\n".join([header, separator] + rows)

    def __save_comparison_tables(
        self,
        test_results: Mapping[str, Mapping[str, float]],
        save_dir: Union[str, Path],
        train_results: Mapping[str, Mapping[str, float]] | None = None,
        val_results: Mapping[str, Mapping[str, float]] | None = None,
        filename: str = "results.md",
        precision: int = 4,
    ) -> Path:
        """Build and save markdown comparison tables to a file.

        Writes two sections:
        - "## Test Results" with the test metrics table
        - "## Train/Val Results" with the train/val metrics table (if provided)

        Args:
            test_results: Dict from test_all(), mapping model names to test metric dicts.
            save_dir: Directory where the markdown file will be written.
            train_results: Optional dict mapping model names to train metric dicts.
            val_results: Optional dict mapping model names to val metric dicts.
            filename: Name of the output file.
            precision: Decimal places for metric values.

        Returns:
            Path to the written file.
        """
        sections = []

        # Test results table
        test_table = self.__build_comparison_table(test_results, precision)
        if test_table:
            sections.append(f"## Test Results\n\n{test_table}")

        # Train/val results table
        if train_results or val_results:
            if train_results:
                train_table = self.__build_comparison_table(train_results, precision)
                sections.append(f"## Train Results\n\n{train_table}")
            if val_results:
                val_table = self.__build_comparison_table(val_results, precision)
                sections.append(f"## Val Results\n\n{val_table}")

        content = "\n\n".join(sections) + "\n" if sections else ""

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        file_path = save_path / filename
        file_path.write_text(content)

        return file_path
