from pathlib import Path
import matplotlib.axes._axes as maxes
import pandas as pd
import pytest
from hypertorch.train import LinePlotter, ParsedMetrics


def test_line_plotter_initialization(tmp_path: Path) -> None:
    # Matches experiment_<id> regex
    exp_dir = tmp_path / "experiment_48"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)
    assert plotter.num_exp == "48"
    assert (exp_dir / "plots").exists()

    # Fallback to "0" when no regex match
    custom_dir = tmp_path / "custom_run"
    custom_dir.mkdir()
    custom_plotter = LinePlotter(custom_dir)
    assert custom_plotter.num_exp == "0"


def test_line_plotter_generates_plots(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_1"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)

    # Build ParsedMetrics container with both multi-point and single-point evaluations
    metrics = ParsedMetrics(x_col="epoch")

    # Multi-point train/val curve + single-point "test" (blue axhline) + "baseline" (gray axhline)
    df_loss = pd.DataFrame(
        {
            "epoch": [0, 1, 0, 1, 0, 0],
            "split": ["train", "train", "val", "val", "test", "baseline"],
            "value": [0.8, 0.4, 0.9, 0.5, 0.35, 1.2],
        }
    )
    metrics.add("loss", df_loss)

    # Single-point only (only axhline, no lineplot curves)
    df_acc = pd.DataFrame(
        {
            "epoch": [0],
            "split": ["test"],
            "value": [0.92],
        }
    )
    metrics.add("accuracy", df_acc)

    # Plot all metrics
    created_plots = plotter.plot(metrics)
    assert len(created_plots) == 2
    for plot_path in created_plots:
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


def test_line_plotter_filter_metric_names(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_0"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)

    metrics = ParsedMetrics(x_col="epoch")
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]}),
    )
    metrics.add(
        "f1",
        pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.7, 0.8]}),
    )

    # Plot only 'loss', ignore non-existent 'unknown_metric'
    created_plots = plotter.plot(metrics, metric_names=["loss", "unknown_metric"])
    assert len(created_plots) == 1
    assert created_plots[0].name == "LinePlot_loss_0.png"


def test_line_plotter_no_legend_handles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp_dir = tmp_path / "experiment_0"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)

    metrics = ParsedMetrics(x_col="epoch")
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]}),
    )

    # Force legend handles to return empty lists to cover the 'if handles: False' branch
    monkeypatch.setattr(maxes.Axes, "get_legend_handles_labels", lambda self: ([], []))

    created_plots = plotter.plot(metrics)
    assert len(created_plots) == 1