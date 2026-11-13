from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock
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
def test_scatterplotter_calculate_grid_dims(n_plots: int, expected_dims: tuple[int, int]) -> None:
    assert ScatterPlotter._calculate_grid_dims(n_plots) == expected_dims


def test_scatterplotter_hide_unused_axes() -> None:
    mock_axes = [MagicMock() for _ in range(6)]
    ScatterPlotter._hide_unused_axes(axes=mock_axes, active_count=4)

    for i in range(4):
        mock_axes[i].set_visible.assert_not_called()
    for i in range(4, 6):
        mock_axes[i].set_visible.assert_called_once_with(False)


""" Data Splitting, Pairing, & Pruning Tests """


def test_scatterplotter_get_unique_splits() -> None:
    m1 = ParsedMetrics(x_col="epoch")
    m1.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))
    m1.add("auc", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.8]}))

    m2 = ParsedMetrics(x_col="epoch")
    m2.add("loss", pd.DataFrame({"epoch": [0], "split": ["test"], "value": [0.4]}))

    splits = ScatterPlotter._get_unique_splits([m1, m2], targets=["loss", "auc"])
    assert splits == ["train", "val", "test"]


def test_scatterplotter_collect_pair_data_prunes_identical_aliases() -> None:
    m = ParsedMetrics(x_col="epoch", experiment_name="run_0")
    df_data = {"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]}
    m.add("loss", pd.DataFrame(df_data))
    m.add("loss_alias", pd.DataFrame(df_data))

    pair_data = ScatterPlotter._collect_pair_data(
        metrics_list=[m],
        candidate_pairs=[("loss", "loss_alias")],
        split_name="train",
        x_col="epoch",
    )

    assert ("loss", "loss_alias") not in pair_data


def test_scatterplotter_collect_pair_data_skips_insufficient_or_missing_points() -> None:
    m = ParsedMetrics(x_col="epoch", experiment_name="run_0")
    m.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}))
    m.add("auc", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.8]}))

    f1_df = pd.DataFrame({"epoch": [0, 1], "split": ["test", "test"], "value": [0.7, 0.9]})
    m.add("f1", f1_df)

    pair_data_len1 = ScatterPlotter._collect_pair_data(
        metrics_list=[m],
        candidate_pairs=[("loss", "auc")],
        split_name="train",
        x_col="epoch",
    )
    assert pair_data_len1 == {}

    pair_data_missing_split = ScatterPlotter._collect_pair_data(
        metrics_list=[m],
        candidate_pairs=[("loss", "f1")],
        split_name="val",
        x_col="epoch",
    )
    assert pair_data_missing_split == {}


def test_scatterplotter_partition_pairs_and_group_by_anchor() -> None:
    dummy_df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "split": ["train", "train"],
            "value_a": [1, 2],
            "value_b": [3, 4],
        }
    )

    pair_data: dict[tuple[str, str], dict[int, tuple[str, pd.DataFrame]]] = {
        ("loss", "auc"): {
            0: ("run_0", dummy_df),
            1: ("run_1", dummy_df),
        },
        ("loss", "f1"): {
            0: ("run_0", dummy_df),
        },
    }

    comparison_pairs, single_run_pairs = ScatterPlotter._partition_pairs(pair_data)
    assert comparison_pairs == [("loss", "auc")]
    assert single_run_pairs == {0: [("loss", "f1")]}

    anchor_groups = ScatterPlotter._group_by_anchor(comparison_pairs)
    assert anchor_groups == {"loss": [("loss", "auc")]}


""" Subplot & Legend Drawing Tests """


def test_scatterplotter_draw_run_scatter() -> None:
    mock_ax = MagicMock()
    merged = pd.DataFrame(
        {
            "value_loss": [0.5, 0.3],
            "value_auc": [0.7, 0.9],
        }
    )

    ScatterPlotter._draw_run_scatter(
        ax=mock_ax,
        merged=merged,
        col_x="value_loss",
        col_y="value_auc",
        run_color="#1f77b4",
        label="model_a",
    )

    assert mock_ax.scatter.call_count == 2

    args1, kwargs1 = mock_ax.scatter.call_args_list[0]
    assert list(args1[0]) == [0.5, 0.3]
    assert list(args1[1]) == [0.7, 0.9]
    assert kwargs1["marker"] == "."
    assert kwargs1["s"] == 32
    assert kwargs1["label"] == "model_a"

    args2, kwargs2 = mock_ax.scatter.call_args_list[1]
    assert args2[0] == 0.3
    assert args2[1] == 0.9
    assert kwargs2["marker"] == "X"
    assert kwargs2["s"] == 60


def test_scatterplotter_attach_chunk_legend_with_and_without_handles() -> None:
    mock_fig = MagicMock()
    mock_ax = MagicMock()

    mock_ax.get_legend_handles_labels.return_value = ([MagicMock()], ["run_0"])
    ScatterPlotter._attach_chunk_legend(
        fig=mock_fig,
        reference_ax=mock_ax,
        title_prefix="Comparison",
        split_name="train",
        is_comparison=True,
    )
    mock_fig.legend.assert_called_once()

    mock_fig.reset_mock()

    mock_ax.get_legend_handles_labels.return_value = ([], [])
    ScatterPlotter._attach_chunk_legend(
        fig=mock_fig,
        reference_ax=mock_ax,
        title_prefix="Experiment 0",
        split_name="val",
        is_comparison=False,
    )
    mock_fig.legend.assert_not_called()


""" Validation & Error Handling Tests """


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


def test_scatterplotter_fewer_than_two_metrics_raises_value_error(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_0"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir)
    df_data = {"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]}
    metrics.add("loss", pd.DataFrame(df_data))

    plotter = ScatterPlotter()
    with pytest.raises(ValueError, match="requires at least two metrics to plot relationships"):
        plotter.plot(metrics)


""" Plot Generation & Subfolder Routing Tests """


def test_scatterplotter_plot_single_run_defaults(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_10"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="10")
    loss_df = pd.DataFrame({"epoch": [0, 1, 2], "split": ["train"] * 3, "value": [0.8, 0.5, 0.2]})
    auc_df = pd.DataFrame({"epoch": [0, 1, 2], "split": ["train"] * 3, "value": [0.6, 0.75, 0.9]})
    metrics.add("loss", loss_df)
    metrics.add("auc", auc_df)

    plotter = ScatterPlotter()
    saved_plots = plotter.plot(metrics)

    expected_path = exp_dir / "plots" / "ScatterPlot_rel_train_10_vs_auc.png"
    assert saved_plots == [expected_path]
    assert expected_path.is_file()


def test_scatterplotter_plot_multi_run_comparison_routing(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    exp_1 = logs_root / "exp_1"
    exp_2 = logs_root / "exp_2"
    exp_1.mkdir(parents=True)
    exp_2.mkdir(parents=True)

    m1 = ParsedMetrics(x_col="epoch", experiment_dir=exp_1, experiment_name="1")
    m1.add("loss", pd.DataFrame({"epoch": [0, 1], "split": ["train"] * 2, "value": [0.8, 0.4]}))
    m1.add("auc", pd.DataFrame({"epoch": [0, 1], "split": ["train"] * 2, "value": [0.6, 0.9]}))
    m1.add("f1", pd.DataFrame({"epoch": [0, 1], "split": ["train"] * 2, "value": [0.5, 0.85]}))

    m2 = ParsedMetrics(x_col="epoch", experiment_dir=exp_2, experiment_name="2")
    m2.add("loss", pd.DataFrame({"epoch": [0, 1], "split": ["train"] * 2, "value": [0.7, 0.3]}))
    m2.add("auc", pd.DataFrame({"epoch": [0, 1], "split": ["train"] * 2, "value": [0.65, 0.92]}))

    plotter = ScatterPlotter()
    saved_plots = plotter.plot([m1, m2])

    expected_comp = (
        logs_root / "plots" / "comparison" / "ScatterPlot_rel_train_comparison_vs_auc.png"
    )
    expected_unique_auc = logs_root / "plots" / "unique" / "ScatterPlot_rel_train_1_vs_auc.png"
    expected_unique_f1 = logs_root / "plots" / "unique" / "ScatterPlot_rel_train_1_vs_f1.png"

    assert set(saved_plots) == {expected_comp, expected_unique_auc, expected_unique_f1}
    assert expected_comp.is_file()
    assert expected_unique_auc.is_file()
    assert expected_unique_f1.is_file()


def test_scatterplotter_plot_custom_output_dir_without_subfolder(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_scatter_dest"
    custom_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_name="custom_run")
    acc_df = pd.DataFrame({"epoch": [0, 1], "split": ["val"] * 2, "value": [0.7, 0.9]})
    loss_df = pd.DataFrame({"epoch": [0, 1], "split": ["val"] * 2, "value": [0.6, 0.2]})
    metrics.add("accuracy", acc_df)
    metrics.add("loss", loss_df)

    plotter = ScatterPlotter()
    saved_plots = plotter.plot(
        metrics=metrics,
        output_dir=custom_dir,
        create_subfolder=False,
    )

    expected_path = custom_dir / "ScatterPlot_rel_val_custom_run_vs_accuracy.png"
    assert saved_plots == [expected_path]
    assert expected_path.is_file()
    assert not (custom_dir / "plots").exists()


def test_scatterplotter_plot_chunking_exceeding_max_plots(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_chunking"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="chunk_exp")
    anchor_df = pd.DataFrame({"epoch": [0, 1], "split": ["train"] * 2, "value": [0.1, 0.2]})
    metrics.add("anchor", anchor_df)
    for i in range(7):
        m_df = pd.DataFrame({"epoch": [0, 1], "split": ["train"] * 2, "value": [0.3 + i, 0.4 + i]})
        metrics.add(f"metric_{i}", m_df)

    plotter = ScatterPlotter()
    metric_names = ["anchor"] + [f"metric_{i}" for i in range(7)]
    saved_plots = plotter.plot(metrics, metric_names=metric_names)

    part_files = [p.name for p in saved_plots if "vs_anchor" in p.name]
    assert len(part_files) == 2
    assert "ScatterPlot_rel_train_chunk_exp_vs_anchor_part1.png" in part_files
    assert "ScatterPlot_rel_train_chunk_exp_vs_anchor_part2.png" in part_files


def test_scatterplotter_plot_skips_when_pair_data_empty(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_no_data"
    exp_dir.mkdir()

    metrics = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="0")
    df_data = {"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]}
    metrics.add("loss", pd.DataFrame(df_data))
    metrics.add("loss_dup", pd.DataFrame(df_data))

    plotter = ScatterPlotter()
    saved_plots = plotter.plot(metrics)

    assert saved_plots == []
