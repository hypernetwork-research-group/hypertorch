from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from hypertorch.train import LinePlotter, Plotter
from hypertorch.types import ParsedMetrics


@pytest.fixture(autouse=True)
def mock_plotting_backend(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mocks matplotlib and seaborn in sys.modules and importlib find_spec."""
    mock_matplotlib = MagicMock()
    mock_plt = MagicMock()
    mock_sns = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()

    # Configure subplots unpacking and legend handles
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_ax.get_legend_handles_labels.return_value = ([MagicMock()], ["train"])

    # Simulate saving plot images to disk so path.exists() checks pass
    mock_plt.savefig.side_effect = lambda path, *args, **kwargs: Path(path).touch()

    mock_matplotlib.pyplot = mock_plt

    # Make _is_plotting_available() evaluate to True by default
    monkeypatch.setattr(
        "hypertorch.train.plotter.importlib.util.find_spec",
        lambda name: MagicMock() if name in {"matplotlib", "seaborn"} else None,
    )
    monkeypatch.setitem(sys.modules, "matplotlib", mock_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mock_plt)
    monkeypatch.setitem(sys.modules, "seaborn", mock_sns)

    return {
        "matplotlib": mock_matplotlib,
        "plt": mock_plt,
        "sns": mock_sns,
        "fig": mock_fig,
        "ax": mock_ax,
    }


def test_plotter_abstract_instantiation() -> None:
    """Verifies that Plotter cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Plotter()  # type: ignore[abstract]


def test_is_plotting_available_true() -> None:
    """Verifies _is_plotting_available returns True when dependencies are found."""
    assert Plotter._is_plotting_available() is True


def test_is_plotting_available_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies _is_plotting_available returns False when dependencies are absent."""
    monkeypatch.setattr(
        "hypertorch.train.plotter.importlib.util.find_spec",
        lambda _name: None,
    )
    assert Plotter._is_plotting_available() is False


def test_is_plotting_available_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies _is_plotting_available returns False when only one dependency is present."""
    monkeypatch.setattr(
        "hypertorch.train.plotter.importlib.util.find_spec",
        lambda name: MagicMock() if name == "matplotlib" else None,
    )
    assert Plotter._is_plotting_available() is False


@patch("hypertorch.train.plotter.Plotter._is_plotting_available", return_value=False)
def test_line_plotter_raises_when_plotting_unavailable(
    _mock_available: MagicMock, tmp_path: Path
) -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch", experiment_dir=tmp_path)
    metrics.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))

    with pytest.raises(
        ImportError,
        match=r"Plotting dependencies are not available\.",
    ):
        plotter.plot(metrics)


def test_line_plotter_validation_missing_primary_column(tmp_path: Path) -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="")  # Empty primary column
    metrics.add("loss", pd.DataFrame({"val": [1.0]}))

    with pytest.raises(ValueError, match="ParsedMetrics contains no primary column"):
        plotter.plot(metrics, output_dir=tmp_path)


def test_line_plotter_validation_no_metrics(tmp_path: Path) -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch")  # No metrics added

    with pytest.raises(ValueError, match="ParsedMetrics contains no metrics to plot"):
        plotter.plot(metrics, output_dir=tmp_path)


def test_line_plotter_validation_missing_destination_path() -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch", experiment_dir=None)
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}),
    )

    with pytest.raises(ValueError, match="Could not find a destination path"):
        plotter.plot(metrics, output_dir=None)


def test_line_plotter_validation_destination_does_not_exist(tmp_path: Path) -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch")
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}),
    )

    non_existent = tmp_path / "ghost_folder"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        plotter.plot(metrics, output_dir=non_existent)


def test_line_plotter_validation_destination_not_a_directory(tmp_path: Path) -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch")
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}),
    )

    file_path = tmp_path / "regular_file.txt"
    file_path.write_text("dummy")

    with pytest.raises(NotADirectoryError, match="is not a directory"):
        plotter.plot(metrics, output_dir=file_path)


def test_line_plotter_default_destination_and_subfolder(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_42"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="42")
    metrics.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 0],
                "split": ["train", "train", "test", "val_single"],
                "value": [0.8, 0.4, 0.35, 0.5],
            }
        ),
    )

    plotter = LinePlotter()
    created_plots = plotter.plot(metrics)

    assert len(created_plots) == 1
    expected_path = exp_dir / "plots" / "LinePlot_loss_42.png"
    assert created_plots[0] == expected_path
    assert expected_path.exists()


def test_line_plotter_custom_output_dir_without_subfolder(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_destination"
    custom_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_name="0")
    metrics.add(
        "accuracy",
        pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.8, 0.9]}),
    )

    plotter = LinePlotter()
    created_plots = plotter.plot(metrics, output_dir=custom_dir, create_subfolder=False)

    assert len(created_plots) == 1
    expected_path = custom_dir / "LinePlot_accuracy_0.png"
    assert created_plots[0] == expected_path
    assert expected_path.exists()
    assert not (custom_dir / "plots").exists()


def test_line_plotter_only_single_point_evaluations(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_10"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="10")
    metrics.add(
        "test_acc",
        pd.DataFrame({"epoch": [0], "split": ["test"], "value": [0.92]}),
    )

    plotter = LinePlotter()
    created_plots = plotter.plot(metrics)

    assert len(created_plots) == 1
    assert created_plots[0].exists()


def test_line_plotter_filter_metrics_and_empty_legend(
    tmp_path: Path, mock_plotting_backend: dict[str, Any]
) -> None:
    exp_dir = tmp_path / "experiment_0"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir)
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]}),
    )
    metrics.add(
        "f1",
        pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.7, 0.8]}),
    )

    mock_plotting_backend["ax"].get_legend_handles_labels.return_value = ([], [])

    plotter = LinePlotter()
    created_plots = plotter.plot(metrics, metric_names=["loss", "non_existent_metric"])

    assert len(created_plots) == 1
    assert created_plots[0].name == "LinePlot_loss_0.png"
