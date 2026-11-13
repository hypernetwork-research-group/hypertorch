from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock
import pandas as pd
import pytest

from hypertorch.plot import LinePlotter, Plotter
from hypertorch.types import ParsedMetrics


@pytest.fixture(autouse=True)
def mock_plotting_backend(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Mock matplotlib and seaborn in sys.modules and importlib to run headless.
    """
    mock_matplotlib = MagicMock()
    mock_plt = MagicMock()
    mock_sns = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()

    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_ax.get_legend_handles_labels.return_value = ([MagicMock()], ["train"])

    def fake_savefig(path: Path | str, *args: Any, **kwargs: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    mock_plt.savefig.side_effect = fake_savefig
    mock_matplotlib.pyplot = mock_plt

    mock_sns.color_palette.return_value = [
        (0.12, 0.47, 0.71),
        (1.0, 0.5, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.4, 0.74),
        (0.55, 0.34, 0.29),
    ]

    monkeypatch.setattr(
        "hypertorch.plot.plotter.importlib.util.find_spec",
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


""" Marker & Split Dispatch Tests """


@pytest.mark.parametrize(
    "split_name, expected_marker",
    [
        ("train", "o"),
        ("val", "s"),
        ("test", "d"),
    ],
)
def test_lineplotter_get_split_marker(split_name: str, expected_marker: str) -> None:
    assert LinePlotter._get_split_marker(split_name) == expected_marker


def test_lineplotter_draw_singular() -> None:
    mock_ax = MagicMock()
    LinePlotter._draw_singular(
        ax=mock_ax,
        val=0.42,
        color="#1f77b4",
        zorder=3,
        label="test (0.4200)",
    )

    mock_ax.axhline.assert_called_once_with(
        y=0.42,
        color="#1f77b4",
        linestyle=":",
        linewidth=1.3,
        alpha=0.8,
        zorder=3,
        label="test (0.4200): 0.4200",
    )


def test_lineplotter_draw_curve() -> None:
    mock_ax = MagicMock()
    group = pd.DataFrame(
        {
            "epoch": [2, 0, 1],
            "value": [0.3, 0.7, 0.5],
        }
    )

    LinePlotter._draw_curve(
        ax=mock_ax,
        group=group,
        x_col="epoch",
        color="#2ca02c",
        marker="s",
        zorder=4,
        label="val",
    )

    mock_ax.plot.assert_called_once()
    args, kwargs = mock_ax.plot.call_args
    assert list(args[0]) == [0, 1, 2]
    assert list(args[1]) == [0.7, 0.5, 0.3]
    assert kwargs["color"] == "#2ca02c"
    assert kwargs["linestyle"] == "-"
    assert kwargs["linewidth"] == 1.4
    assert kwargs["marker"] == "s"
    assert kwargs["markersize"] == 5.0
    assert kwargs["markeredgewidth"] == 0.5
    assert kwargs["zorder"] == 4
    assert kwargs["label"] == "val"


def test_lineplotter_draw_split_dispatches_singular_vs_curve() -> None:
    plotter = LinePlotter()
    mock_ax = MagicMock()

    single_df = pd.DataFrame({"epoch": [0], "value": [0.85]})
    plotter._draw_split(
        ax=mock_ax,
        group=single_df,
        split_name="test",
        x_col="epoch",
        color="gray",
        base_zorder=2,
        label="test",
    )
    mock_ax.axhline.assert_called_once()
    mock_ax.plot.assert_not_called()

    mock_ax.reset_mock()

    curve_df = pd.DataFrame({"epoch": [0, 1], "value": [0.6, 0.4]})
    plotter._draw_split(
        ax=mock_ax,
        group=curve_df,
        split_name="train",
        x_col="epoch",
        color="blue",
        base_zorder=2,
        label="train",
    )
    mock_ax.plot.assert_called_once()
    _, kwargs = mock_ax.plot.call_args
    assert kwargs["marker"] == "o"
    mock_ax.axhline.assert_not_called()


""" Dependency & Validation Tests """


def test_lineplotter_missing_dependencies_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hypertorch.plot.plotter.Plotter._is_plotting_available",
        staticmethod(lambda: False),
    )
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch")

    with pytest.raises(ImportError, match="Plotting dependencies are not available"):
        plotter.plot(metrics)


def test_lineplotter_validation_missing_primary_column(tmp_path: Path) -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="")
    metrics.add("loss", pd.DataFrame({"val": [1.0]}))

    with pytest.raises(ValueError, match="ParsedMetrics contains no primary column"):
        plotter.plot(metrics, output_dir=tmp_path)


def test_lineplotter_validation_no_metrics(tmp_path: Path) -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch")

    with pytest.raises(ValueError, match="ParsedMetrics contains no metrics to plot"):
        plotter.plot(metrics, output_dir=tmp_path)


def test_lineplotter_validation_missing_destination_path() -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch", experiment_dir=None)
    metrics.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))

    with pytest.raises(ValueError, match="Could not find a destination path"):
        plotter.plot(metrics, output_dir=None)


def test_lineplotter_validation_destination_does_not_exist(tmp_path: Path) -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch")
    metrics.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))

    non_existent = tmp_path / "ghost_folder"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        plotter.plot(metrics, output_dir=non_existent)


def test_lineplotter_validation_destination_not_a_directory(tmp_path: Path) -> None:
    plotter = LinePlotter()
    metrics = ParsedMetrics(x_col="epoch")
    metrics.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))

    file_path = tmp_path / "regular_file.txt"
    file_path.write_text("dummy")

    with pytest.raises(NotADirectoryError, match="is not a directory"):
        plotter.plot(metrics, output_dir=file_path)


def test_lineplotter_mismatched_x_col_raises_value_error(tmp_path: Path) -> None:
    plotter = LinePlotter()
    exp_dir = tmp_path / "experiment_0"
    exp_dir.mkdir()

    m1 = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir)
    m1.add(
        "loss", pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.4]})
    )

    m2 = ParsedMetrics(x_col="step", experiment_dir=exp_dir)
    m2.add("loss", pd.DataFrame({"step": [0, 1], "split": ["train", "train"], "value": [0.5, 0.4]}))

    with pytest.raises(ValueError, match="Mismatched primary column 'step' found"):
        plotter.plot([m1, m2])


""" Single-Run Plot Generation Tests """


def test_lineplotter_plot_single_run_defaults(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_42"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="42")
    metrics.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1],
                "split": ["train", "train"],
                "value": [0.8, 0.4],
            }
        ),
    )

    plotter = LinePlotter()
    created_plots = plotter.plot(metrics)

    assert len(created_plots) == 1
    expected_path = exp_dir / "plots" / "LinePlot_loss_42.png"
    assert created_plots[0] == expected_path
    assert expected_path.exists()


def test_lineplotter_plot_single_run_with_mixed_splits(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_1"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="1")
    metrics.add(
        "accuracy",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1, 0],
                "split": ["train", "train", "val", "val", "test"],
                "value": [0.6, 0.8, 0.55, 0.75, 0.72],
            }
        ),
    )

    plotter = LinePlotter()
    created_plots = plotter.plot(metrics)

    expected_file = exp_dir / "plots" / "LinePlot_accuracy_1.png"
    assert created_plots == [expected_file]
    assert expected_file.exists()


def test_lineplotter_plot_custom_output_dir_without_subfolder(tmp_path: Path) -> None:
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


def test_lineplotter_only_single_point_evaluations(tmp_path: Path) -> None:
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


# 50
def test_lineplotter_plot_skips_when_no_runs_contain_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp_dir = tmp_path / "exp_empty_targets"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="0")
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}),
    )

    monkeypatch.setattr(
        Plotter,
        "_resolve_target_metrics",
        staticmethod(lambda metrics_list, metric_names: ["ghost_metric"]),
    )

    plotter = LinePlotter()
    saved_plots = plotter.plot(metrics)
    assert saved_plots == []


""" Multi-Run & Layout Behavior Tests """


# 215
def test_lineplotter_plot_multi_run_comparison_routing(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    exp_1 = logs_root / "experiment_1"
    exp_2 = logs_root / "experiment_2"
    exp_1.mkdir(parents=True)
    exp_2.mkdir(parents=True)

    m1 = ParsedMetrics(x_col="epoch", experiment_dir=exp_1, experiment_name="1")
    m1.add(
        "loss", pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]})
    )

    m2 = ParsedMetrics(x_col="epoch", experiment_dir=exp_2, experiment_name="2")
    m2.add(
        "loss", pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.6, 0.35]})
    )

    plotter = LinePlotter()
    saved_plots = plotter.plot([m1, m2])

    expected_path = logs_root / "plots" / "comparison" / "LinePlot_loss_comparison.png"
    assert saved_plots == [expected_path]
    assert expected_path.is_file()


def test_lineplotter_plot_multi_run_unique_metric_routing(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    exp_1 = logs_root / "experiment_1"
    exp_2 = logs_root / "experiment_2"
    exp_1.mkdir(parents=True)
    exp_2.mkdir(parents=True)

    m1 = ParsedMetrics(x_col="epoch", experiment_dir=exp_1, experiment_name="1")
    m1.add(
        "loss", pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]})
    )
    m1.add("auc", pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.7, 0.9]}))

    m2 = ParsedMetrics(x_col="epoch", experiment_dir=exp_2, experiment_name="2")
    m2.add(
        "loss", pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.6, 0.4]})
    )

    plotter = LinePlotter()
    saved_plots = plotter.plot([m1, m2])

    expected_comp = logs_root / "plots" / "comparison" / "LinePlot_loss_comparison.png"
    expected_unique = logs_root / "plots" / "unique" / "LinePlot_auc_1.png"

    assert set(saved_plots) == {expected_comp, expected_unique}
    assert expected_comp.is_file()
    assert expected_unique.is_file()


# 211; 273->271
def test_lineplotter_render_run_curves_skips_empty_split_group(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    exp_1 = logs_root / "experiment_1"
    exp_2 = logs_root / "experiment_2"
    exp_1.mkdir(parents=True)
    exp_2.mkdir(parents=True)

    # Run 1 has 'train' and 'val'
    m1 = ParsedMetrics(x_col="epoch", experiment_dir=exp_1, experiment_name="1")
    m1.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1],
                "split": ["train", "train", "val", "val"],
                "value": [0.6, 0.4, 0.5, 0.3],
            }
        ),
    )

    # Run 2 only has 'train' (its 'val' group is completely empty)
    m2 = ParsedMetrics(x_col="epoch", experiment_dir=exp_2, experiment_name="2")
    m2.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1],
                "split": ["train", "train"],
                "value": [0.7, 0.5],
            }
        ),
    )

    plotter = LinePlotter()
    saved_plots = plotter.plot([m1, m2])

    expected_path = logs_root / "plots" / "comparison" / "LinePlot_loss_comparison.png"
    assert saved_plots == [expected_path]
    assert expected_path.is_file()


def test_lineplotter_plot_busy_layout_legend(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_busy"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="busy")
    metrics.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1, 0],
                "split": ["train", "train", "val", "val", "test"],
                "value": [0.9, 0.4, 0.85, 0.45, 0.8],
            }
        ),
    )

    plotter = LinePlotter()
    saved_plots = plotter.plot(metrics)

    expected_path = exp_dir / "plots" / "LinePlot_loss_busy.png"
    assert saved_plots == [expected_path]
    assert expected_path.exists()


def test_lineplotter_plot_filter_metrics_and_empty_subset(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_filter"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="filtered")
    metrics.add(
        "loss", pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]})
    )
    metrics.add(
        "auc", pd.DataFrame({"epoch": [0, 1], "split": ["val", "val"], "value": [0.75, 0.85]})
    )

    plotter = LinePlotter()

    saved_plots = plotter.plot(metrics, metric_names=["auc"])
    assert len(saved_plots) == 1
    assert saved_plots[0].name == "LinePlot_auc_filtered.png"

    empty_plots = plotter.plot(metrics, metric_names=["non_existent_metric"])
    assert empty_plots == []
