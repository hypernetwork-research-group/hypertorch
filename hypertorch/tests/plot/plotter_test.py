from pathlib import Path
import matplotlib.axes._axes as maxes
import pandas as pd
import pytest
from hypertorch.train.plotter import LinePlotter


def test_line_plotter_initialization(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_48"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)
    assert plotter.num_exp == "48"
    assert (exp_dir / "plots").exists()


def test_line_plotter_generates_png(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_0"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)

    df = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train/loss": [0.8, 0.5, 0.3],
            "test/accuracy": [0.85, None, None],  # split == "test" -> True for blue color
            "baseline/loss": [0.99, None, None],  # split == "test" -> False for gray color
            "unslashed_metric": [1.0, 2.0, 3.0],  # no slash -> False for Split lambda ternary
        }
    )

    created_plots = plotter.plot(df, exp_dir / "metrics.csv")
    assert len(created_plots) > 0


def test_line_plotter_x_col_and_tracking_collision(tmp_path: Path) -> None:
    """Hits deeply nested x_col fallbacks and 'clean_var in tracking_cols' False branch."""
    exp_dir = tmp_path / "experiment_0"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)

    df_step = pd.DataFrame({"step": [1, 2], "train/loss": [0.5, 0.4]})
    assert len(plotter.plot(df_step, exp_dir / "metrics.csv")) == 1

    df_custom = pd.DataFrame({"custom_idx": [1, 2], "train/epoch": [10, 20]})
    assert len(plotter.plot(df_custom, exp_dir / "metrics.csv")) == 0


def test_line_plotter_branches(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_0"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)

    df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "loss": [0.5, 0.3],  # c == var_name (True)
            "train/loss": [
                0.6,
                0.4,
            ],  # c == var_name (False), "/" in c (True), split == var_name (True)
            "accuracy": [0.7, 0.9],  # c == var_name (False), "/" in c (False)
            "test/accuracy": [
                0.8,
                0.8,
            ],  # c == var_name (False), "/" in c (True), split == var_name (False)
        }
    )

    created_plots = plotter.plot(df, exp_dir / "metrics.csv", metrics=["loss"])
    assert len(created_plots) == 1


def test_line_plotter_empty_melted(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment_0"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)
    df = pd.DataFrame({"epoch": [0, 1], "train/loss": [None, None]})
    plots = plotter.plot(df, exp_dir / "metrics.csv")
    assert len(plots) == 0


def test_line_plotter_no_legend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp_dir = tmp_path / "experiment_0"
    exp_dir.mkdir()
    plotter = LinePlotter(exp_dir)
    df = pd.DataFrame({"epoch": [0, 1], "train/loss": [0.5, 0.3]})
    monkeypatch.setattr(maxes.Axes, "get_legend_handles_labels", lambda self: ([], []))
    plots = plotter.plot(df, exp_dir / "metrics.csv")
    assert len(plots) == 1
