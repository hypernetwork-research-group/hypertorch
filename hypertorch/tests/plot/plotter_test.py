from collections.abc import Sequence
from pathlib import Path
import pytest
import pandas as pd

from hypertorch.plot import Plotter
from hypertorch.types import ParsedMetrics


class DummyPlotter(Plotter):
    """Implementation of Plotter for unit testing."""

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        return []


""" ABC Enforcement & Dependency Initialization Tests """


def test_plotter_is_plotting_available_returns_bool() -> None:
    available = Plotter._is_plotting_available()
    assert isinstance(available, bool)


def test_plotter_is_plotting_available_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
    assert Plotter._is_plotting_available() is False


def test_plotter_init_plotting_success() -> None:
    plt, sns = Plotter._init_plotting()
    assert plt is not None
    assert sns is not None


def test_plotter_init_plotting_missing_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Plotter, "_is_plotting_available", staticmethod(lambda: False))
    with pytest.raises(ImportError, match="Plotting dependencies are not available"):
        Plotter._init_plotting()


""" String Formatting & Naming Helper Tests """


def test_plotter_format_label() -> None:
    assert Plotter._format_label("loss") == "Loss"
    assert Plotter._format_label("local_loss") == "Local loss"
    assert Plotter._format_label("avg_precision") == "Avg precision"


def test_plotter_sanitize() -> None:
    assert Plotter._sanitize("train/loss") == "train_loss"
    assert Plotter._sanitize("checkpoint/model/v1") == "checkpoint_model_v1"
    assert Plotter._sanitize("standard_metric") == "standard_metric"


def test_plotter_resolve_run_naming_single_run() -> None:
    prefix, stem = Plotter._resolve_run_naming(is_multi=False, run_name="exp_1")
    assert prefix == "Experiment exp_1"
    assert stem == "exp_1"


def test_plotter_resolve_run_naming_multi_run() -> None:
    prefix, stem = Plotter._resolve_run_naming(is_multi=True, run_name="exp_1")
    assert prefix == "Comparison"
    assert stem == "comparison"


""" Color & Shading Helper Tests """


def test_plotter_get_shade_single_split() -> None:
    base_color = (0.2, 0.4, 0.6)
    shade = Plotter._get_shade(base_color=base_color, split_idx=0, num_splits=1)
    assert shade == base_color


def test_plotter_get_shade_multi_split_progression() -> None:
    base_color = (0.2, 0.4, 0.6)
    num_splits = 3
    shade_0 = Plotter._get_shade(base_color, split_idx=0, num_splits=num_splits)
    shade_1 = Plotter._get_shade(base_color, split_idx=1, num_splits=num_splits)
    shade_2 = Plotter._get_shade(base_color, split_idx=2, num_splits=num_splits)

    assert shade_0 != shade_1 != shade_2
    assert all(0.0 <= channel <= 1.0 for shade in (shade_0, shade_1, shade_2) for channel in shade)


""" Destination Directory Resolution Tests """


def test_plotter_resolve_dest_dir_single_run(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    dest = Plotter._resolve_dest_dir(
        plots_dir=plots_dir,
        is_multi=False,
        total_runs=1,
        create_subfolder=True,
    )
    assert dest == plots_dir
    assert dest.exists()


def test_plotter_resolve_dest_dir_multi_run_comparison_subfolder(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    dest = Plotter._resolve_dest_dir(
        plots_dir=plots_dir,
        is_multi=True,
        total_runs=2,
        create_subfolder=True,
    )
    assert dest == plots_dir / "comparison"
    assert dest.exists()


def test_plotter_resolve_dest_dir_multi_run_unique_subfolder(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    dest = Plotter._resolve_dest_dir(
        plots_dir=plots_dir,
        is_multi=False,
        total_runs=2,
        create_subfolder=True,
    )
    assert dest == plots_dir / "unique"
    assert dest.exists()


def test_plotter_resolve_dest_dir_multi_run_without_subfolder(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    dest = Plotter._resolve_dest_dir(
        plots_dir=plots_dir,
        is_multi=True,
        total_runs=2,
        create_subfolder=False,
    )
    assert dest == plots_dir
    assert dest.exists()


""" Metrics Normalization & Validation Tests """


def test_plotter_normalize_metrics_single_instance() -> None:
    m = ParsedMetrics(x_col="epoch")
    normalized = Plotter._normalize_metrics(m)
    assert isinstance(normalized, list)
    assert len(normalized) == 1
    assert normalized[0] is m


def test_plotter_normalize_metrics_sequence() -> None:
    m1 = ParsedMetrics(x_col="epoch")
    m2 = ParsedMetrics(x_col="epoch")
    normalized = Plotter._normalize_metrics([m1, m2])
    assert isinstance(normalized, list)
    assert len(normalized) == 2


def test_plotter_normalize_metrics_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="No ParsedMetrics provided to plot"):
        Plotter._normalize_metrics([])


def test_plotter_validate_metrics_valid() -> None:
    m = ParsedMetrics(x_col="epoch")
    m.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))
    Plotter._validate_metrics(m)


def test_plotter_validate_metrics_missing_x_col_raises_error() -> None:
    m = ParsedMetrics(x_col="")
    with pytest.raises(ValueError, match="contains no primary column"):
        Plotter._validate_metrics(m)


def test_plotter_validate_metrics_empty_metrics_raises_error() -> None:
    m = ParsedMetrics(x_col="epoch")
    with pytest.raises(ValueError, match="contains no metrics to plot"):
        Plotter._validate_metrics(m)


def test_plotter_validate_metrics_list_valid() -> None:
    m1 = ParsedMetrics(x_col="epoch")
    m1.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))
    m2 = ParsedMetrics(x_col="epoch")
    m2.add("acc", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.8]}))

    Plotter._validate_metrics_list([m1, m2])


def test_plotter_validate_metrics_list_empty_raises_error() -> None:
    with pytest.raises(ValueError, match="ParsedMetrics list cannot be empty"):
        Plotter._validate_metrics_list([])


def test_plotter_validate_metrics_list_mismatched_x_col_raises_error() -> None:
    m1 = ParsedMetrics(x_col="epoch")
    m1.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))
    m2 = ParsedMetrics(x_col="step")
    m2.add("loss", pd.DataFrame({"step": [0], "split": ["train"], "value": [0.5]}))

    with pytest.raises(ValueError, match="Mismatched primary column 'step' found"):
        Plotter._validate_metrics_list([m1, m2])


""" Target Metric Resolution Tests """


def test_plotter_resolve_target_metrics_defaults_preserves_order() -> None:
    m1 = ParsedMetrics(x_col="epoch")
    m1.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))
    m1.add("auc", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.8]}))

    m2 = ParsedMetrics(x_col="epoch")
    m2.add("auc", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.85]}))
    m2.add("accuracy", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.9]}))

    targets = Plotter._resolve_target_metrics([m1, m2], metric_names=None)
    assert targets == ["auc", "loss", "accuracy"]


def test_plotter_resolve_target_metrics_filtered_subset() -> None:
    m1 = ParsedMetrics(x_col="epoch")
    m1.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))
    m1.add("auc", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.8]}))

    targets = Plotter._resolve_target_metrics([m1], metric_names=["auc", "non_existent"])
    assert targets == ["auc"]


""" Plots Directory Resolution Tests """


def test_plotter_resolve_plots_dir_explicit_output_dir(tmp_path: Path) -> None:
    m = ParsedMetrics(x_col="epoch")
    out_dir = tmp_path / "custom_output"
    out_dir.mkdir()

    resolved_with_subfolder = Plotter._resolve_plots_dir(
        metrics_list=[m],
        output_dir=out_dir,
        create_subfolder=True,
    )
    assert resolved_with_subfolder == out_dir / "plots"
    assert resolved_with_subfolder.exists()

    resolved_no_subfolder = Plotter._resolve_plots_dir(
        metrics_list=[m],
        output_dir=out_dir,
        create_subfolder=False,
    )
    assert resolved_no_subfolder == out_dir


def test_plotter_resolve_plots_dir_from_single_experiment_dir(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_1"
    exp_dir.mkdir()
    m = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir)

    resolved = Plotter._resolve_plots_dir(
        metrics_list=[m],
        output_dir=None,
        create_subfolder=True,
    )
    assert resolved == exp_dir / "plots"
    assert resolved.exists()


def test_plotter_resolve_plots_dir_from_multi_experiment_dir(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    exp_1 = logs_root / "exp_1"
    exp_2 = logs_root / "exp_2"
    exp_1.mkdir(parents=True)
    exp_2.mkdir(parents=True)

    m1 = ParsedMetrics(x_col="epoch", experiment_dir=exp_1)
    m2 = ParsedMetrics(x_col="epoch", experiment_dir=exp_2)

    resolved = Plotter._resolve_plots_dir(
        metrics_list=[m1, m2],
        output_dir=None,
        create_subfolder=True,
    )
    assert resolved == logs_root / "plots"
    assert resolved.exists()


def test_plotter_resolve_plots_dir_missing_destination_raises_error() -> None:
    m = ParsedMetrics(x_col="epoch", experiment_dir=None)
    with pytest.raises(ValueError, match="Could not find a destination path"):
        Plotter._resolve_plots_dir(metrics_list=[m], output_dir=None, create_subfolder=True)


def test_plotter_resolve_plots_dir_nonexistent_directory_raises_error(tmp_path: Path) -> None:
    non_existent = tmp_path / "ghost_dir"
    m = ParsedMetrics(x_col="epoch", experiment_dir=non_existent)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Plotter._resolve_plots_dir(metrics_list=[m], output_dir=None, create_subfolder=True)
