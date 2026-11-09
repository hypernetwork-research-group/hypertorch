from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock
import warnings

import pandas as pd
import pytest

from hypertorch.plot import ScatterPlotter
from hypertorch.types import ParsedMetrics


@pytest.fixture(autouse=True)
def mock_plotting_backend(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Mock matplotlib and seaborn in sys.modules and importlib to run headless.
    """
    mock_matplotlib = MagicMock()
    mock_plt = MagicMock()
    mock_sns = MagicMock()

    def fake_subplots(*args: Any, **kwargs: Any) -> tuple[MagicMock, MagicMock]:
        fig = MagicMock()
        axes_list = [MagicMock() for _ in range(12)]
        for ax in axes_list:
            ax.get_legend_handles_labels.return_value = ([MagicMock()], ["run_0"])
        ax_container = MagicMock()
        ax_container.flatten.return_value = axes_list
        return fig, ax_container

    mock_plt.subplots.side_effect = fake_subplots

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
    }


""" Layout & Grid Dimension Tests """


@pytest.mark.parametrize(
    "n_plots, expected_dims",
    [
        (1, (1, 1)),
        (2, (2, 1)),
        (3, (3, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
        (6, (3, 2)),
        (7, (3, 3)),
        (9, (3, 3)),
    ],
)
def test_scatterplotter_calculate_grid_dims(
    n_plots: int, expected_dims: tuple[int, int]
) -> None:
    assert ScatterPlotter._calculate_grid_dims(n_plots) == expected_dims


""" Data Extraction & Warning Tests """


def test_scatterplotter_collect_generalization_data_success() -> None:
    m = ParsedMetrics(x_col="epoch", experiment_name="run_0")
    m.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1],
                "split": ["train", "train", "val", "val"],
                "value": [0.8, 0.4, 0.9, 0.5],
            }
        ),
    )

    data = ScatterPlotter._collect_generalization_data([m], targets=["loss"])
    assert "loss" in data
    assert 0 in data["loss"]
    run_label, final_train, final_val = data["loss"][0]
    assert run_label == "run_0"
    assert final_train == 0.4
    assert final_val == 0.5


def test_scatterplotter_collect_generalization_data_warning_missing_val() -> None:
    m = ParsedMetrics(x_col="epoch", experiment_name="run_no_val")
    m.add(
        "accuracy",
        pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.6, 0.8]}),
    )

    with pytest.warns(UserWarning, match="no validation split"):
        data = ScatterPlotter._collect_generalization_data([m], targets=["accuracy"])
    assert data == {}


def test_scatterplotter_collect_generalization_data_warning_missing_train() -> None:
    m = ParsedMetrics(x_col="epoch", experiment_name="run_no_train")
    m.add(
        "accuracy",
        pd.DataFrame({"epoch": [0, 1], "split": ["val", "val"], "value": [0.55, 0.75]}),
    )

    with pytest.warns(UserWarning, match="no training split"):
        data = ScatterPlotter._collect_generalization_data([m], targets=["accuracy"])
    assert data == {}


def test_scatterplotter_collect_generalization_data_skips_neither_train_nor_val() -> None:
    m = ParsedMetrics(x_col="epoch", experiment_name="run_test_only")
    m.add("loss", pd.DataFrame({"epoch": [0], "split": ["test"], "value": [0.5]}))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = ScatterPlotter._collect_generalization_data([m], targets=["loss"])
    assert data == {}


def test_scatterplotter_collect_generalization_data_skips_empty_split_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = ParsedMetrics(x_col="epoch", experiment_name="run_empty_split")
    m.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))

    mock_df = MagicMock()
    mock_df.__getitem__.side_effect = lambda key: (
        pd.Series(["train", "val"])
        if isinstance(key, str) and key == "split"
        else pd.DataFrame()
    )
    monkeypatch.setattr(m, "fetch", lambda var: mock_df)

    data = ScatterPlotter._collect_generalization_data([m], targets=["loss"])
    assert data == {}


def test_scatterplotter_partition_metrics() -> None:
    metric_data = {
        "loss": {0: ("run_0", 0.3, 0.4), 1: ("run_1", 0.25, 0.35)},
        "accuracy": {0: ("run_0", 0.85, 0.80)},
    }

    comparison_metrics, single_run_metrics = ScatterPlotter._partition_metrics(metric_data)
    assert comparison_metrics == ["loss"]
    assert single_run_metrics == {0: ["accuracy"]}


""" Subplot & Reference Diagonal Tests """


def test_scatterplotter_render_subplot() -> None:
    mock_ax = MagicMock()
    plotter = ScatterPlotter()
    run_matches = {0: ("model_a", 0.2, 0.3), 1: ("model_b", 0.15, 0.22)}
    palette = ["#1f77b4", "#ff7f0e"]

    plotter._render_subplot(
        ax=mock_ax,
        var_name="loss",
        run_matches=run_matches,
        palette=palette,
        show_label=True,
    )

    assert mock_ax.scatter.call_count == 2
    mock_ax.plot.assert_called_once()
    _, plot_kwargs = mock_ax.plot.call_args
    assert plot_kwargs["linestyle"] == "--"
    assert plot_kwargs["label"] == "y = x"

    mock_ax.set_title.assert_called_once_with("Loss", fontsize=12, pad=10)
    mock_ax.set_xlabel.assert_called_once_with("Train Loss", fontsize=10)
    mock_ax.set_ylabel.assert_called_once_with("Val Loss", fontsize=10)


""" Dependency & Validation Tests """


def test_scatterplotter_missing_dependencies_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hypertorch.plot.plotter.Plotter._is_plotting_available",
        staticmethod(lambda: False),
    )
    plotter = ScatterPlotter()
    metrics = ParsedMetrics(x_col="epoch")

    with pytest.raises(ImportError, match="Plotting dependencies are not available"):
        plotter.plot(metrics)


def test_scatterplotter_plot_returns_empty_when_no_targets(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_empty"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir)
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}),
    )

    plotter = ScatterPlotter()
    saved_plots = plotter.plot(metrics, metric_names=["non_existent_metric"])
    assert saved_plots == []


""" Plot Generation & Subfolder Routing Tests """


def test_scatterplotter_plot_single_run_defaults(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_42"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="42")
    metrics.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1],
                "split": ["train", "train", "val", "val"],
                "value": [0.8, 0.4, 0.85, 0.45],
            }
        ),
    )

    plotter = ScatterPlotter()
    saved_plots = plotter.plot(metrics)

    expected_path = exp_dir / "plots" / "ScatterPlot_train_vs_val_42.png"
    assert saved_plots == [expected_path]
    assert expected_path.is_file()


def test_scatterplotter_plot_multi_run_comparison_routing(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    exp_1 = logs_root / "exp_1"
    exp_2 = logs_root / "exp_2"
    exp_1.mkdir(parents=True)
    exp_2.mkdir(parents=True)

    m1 = ParsedMetrics(x_col="epoch", experiment_dir=exp_1, experiment_name="1")
    m1.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1],
                "split": ["train", "train", "val", "val"],
                "value": [0.7, 0.3, 0.75, 0.38],
            }
        ),
    )
    m1.add(
        "auc",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1],
                "split": ["train", "train", "val", "val"],
                "value": [0.6, 0.9, 0.55, 0.88],
            }
        ),
    )

    m2 = ParsedMetrics(x_col="epoch", experiment_dir=exp_2, experiment_name="2")
    m2.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1],
                "split": ["train", "train", "val", "val"],
                "value": [0.65, 0.28, 0.7, 0.32],
            }
        ),
    )

    plotter = ScatterPlotter()
    saved_plots = plotter.plot([m1, m2])

    expected_comp = (
        logs_root / "plots" / "comparison" / "ScatterPlot_train_vs_val_comparison.png"
    )
    expected_unique = logs_root / "plots" / "unique" / "ScatterPlot_train_vs_val_1.png"

    assert set(saved_plots) == {expected_comp, expected_unique}
    assert expected_comp.is_file()
    assert expected_unique.is_file()


def test_scatterplotter_plot_custom_output_dir_without_subfolder(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_scatter_dest"
    custom_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_name="custom_run")
    metrics.add(
        "accuracy",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1],
                "split": ["train", "train", "val", "val"],
                "value": [0.7, 0.9, 0.65, 0.85],
            }
        ),
    )

    plotter = ScatterPlotter()
    saved_plots = plotter.plot(
        metrics=metrics,
        output_dir=custom_dir,
        create_subfolder=False,
    )

    expected_path = custom_dir / "ScatterPlot_train_vs_val_custom_run.png"
    assert saved_plots == [expected_path]
    assert expected_path.is_file()
    assert not (custom_dir / "plots").exists()


def test_scatterplotter_plot_chunking_exceeding_max_plots(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_chunking"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="chunk_exp")
    for i in range(7):
        metrics.add(
            f"metric_{i}",
            pd.DataFrame(
                {
                    "epoch": [0, 1, 0, 1],
                    "split": ["train", "train", "val", "val"],
                    "value": [0.1 + i, 0.2 + i, 0.15 + i, 0.25 + i],
                }
            ),
        )

    plotter = ScatterPlotter()
    saved_plots = plotter.plot(metrics)

    part_files = [p.name for p in saved_plots]
    assert len(part_files) == 2
    assert "ScatterPlot_train_vs_val_chunk_exp_part1.png" in part_files
    assert "ScatterPlot_train_vs_val_chunk_exp_part2.png" in part_files


def test_scatterplotter_plot_skips_when_generalization_data_empty(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_no_pairs"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="0")
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]}),
    )

    plotter = ScatterPlotter()
    with pytest.warns(UserWarning, match="no validation split"):
        saved_plots = plotter.plot(metrics)

    assert saved_plots == []


def test_scatterplotter_plot_skips_legend_when_handles_empty(
    tmp_path: Path, mock_plotting_backend: dict[str, Any]
) -> None:
    exp_dir = tmp_path / "exp_no_handles"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="no_handles")
    metrics.add(
        "loss",
        pd.DataFrame(
            {
                "epoch": [0, 1, 0, 1],
                "split": ["train", "train", "val", "val"],
                "value": [0.5, 0.3, 0.6, 0.4],
            }
        ),
    )

    def fake_subplots_empty(*args: Any, **kwargs: Any) -> tuple[MagicMock, MagicMock]:
        fig = MagicMock()
        axes_list = [MagicMock() for _ in range(12)]
        for ax in axes_list:
            ax.get_legend_handles_labels.return_value = ([], [])
        ax_container = MagicMock()
        ax_container.flatten.return_value = axes_list
        return fig, ax_container

    mock_plotting_backend["plt"].subplots.side_effect = fake_subplots_empty

    plotter = ScatterPlotter()
    saved_plots = plotter.plot(metrics)

    assert len(saved_plots) == 1
    assert saved_plots[0].is_file()