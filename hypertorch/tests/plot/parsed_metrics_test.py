from pathlib import Path
import pandas as pd
import pytest
from hypertorch.types import ParsedMetrics


def test_parsed_metrics_initialization_defaults() -> None:
    metrics = ParsedMetrics(x_col="epoch")

    assert metrics.x_col == "epoch"
    assert metrics.csv_path is None
    assert metrics.experiment_dir is None
    assert metrics.experiment_name == "0"
    assert len(metrics) == 0
    assert metrics.names() == []
    assert metrics.all() == {}


def test_parsed_metrics_initialization_with_values(tmp_path: Path) -> None:
    csv_file = tmp_path / "metrics.csv"
    exp_dir = tmp_path / "experiment_3"

    metrics = ParsedMetrics(
        x_col="step",
        csv_path=csv_file,
        experiment_dir=exp_dir,
        experiment_name="3",
    )

    assert metrics.x_col == "step"
    assert metrics.csv_path == csv_file
    assert metrics.experiment_dir == exp_dir
    assert metrics.experiment_name == "3"


def test_parsed_metrics_add_fetch_and_getitem() -> None:
    metrics = ParsedMetrics(x_col="epoch")
    loss_df = pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]})
    acc_df = pd.DataFrame({"epoch": [0, 1], "split": ["val", "val"], "value": [0.8, 0.9]})

    metrics.add("loss", loss_df)
    metrics.add("accuracy", acc_df)

    assert len(metrics) == 2
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "f1" not in metrics

    pd.testing.assert_frame_equal(metrics.fetch("loss"), loss_df)
    pd.testing.assert_frame_equal(metrics["accuracy"], acc_df)


def test_parsed_metrics_fetch_missing_raises_keyerror() -> None:
    metrics = ParsedMetrics(x_col="epoch")
    metrics.add(
        "loss",
        pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]}),
    )
    metrics.add(
        "accuracy",
        pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.8]}),
    )

    with pytest.raises(KeyError) as exc_info:
        metrics.fetch("missing_metric")

    assert "Metric 'missing_metric' not found." in str(exc_info.value)
    assert "Available metrics: [accuracy, loss]" in str(exc_info.value)


def test_parsed_metrics_all_returns_shallow_copy() -> None:
    metrics = ParsedMetrics(x_col="epoch")
    df = pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.5]})
    metrics.add("loss", df)

    all_dict = metrics.all()
    assert "loss" in all_dict

    all_dict["ghost_metric"] = df
    assert "ghost_metric" not in metrics


def test_parsed_metrics_names_sorting_and_iter() -> None:
    metrics = ParsedMetrics(x_col="epoch")
    df = pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.1]})

    metrics.add("zebra", df)
    metrics.add("alpha", df)
    metrics.add("beta", df)

    assert metrics.names() == ["alpha", "beta", "zebra"]
    assert list(iter(metrics)) == ["alpha", "beta", "zebra"]


def test_parsed_metrics_repr() -> None:
    metrics = ParsedMetrics(x_col="step")
    metrics.add(
        "loss",
        pd.DataFrame({"step": [0], "split": ["train"], "value": [0.5]}),
    )

    assert repr(metrics) == "ParsedMetrics(x_col='step', metrics=['loss'])"


def test_parsed_metrics_average_reduces_duplicates(tmp_path: Path) -> None:
    csv_file = tmp_path / "metrics.csv"
    exp_dir = tmp_path / "experiment_1"

    metrics = ParsedMetrics(
        x_col="epoch",
        csv_path=csv_file,
        experiment_dir=exp_dir,
        experiment_name="1",
    )

    # Epoch 0 has two train values (10.0 and 20.0), averaging to 15.0
    loss_df = pd.DataFrame(
        {
            "epoch": [0, 0, 1],
            "split": ["train", "train", "train"],
            "value": [10.0, 20.0, 5.0],
        }
    )
    metrics.add("loss", loss_df)

    averaged = metrics.average()

    # Metadata must be preserved on the new instance
    assert averaged.x_col == "epoch"
    assert averaged.csv_path == csv_file
    assert averaged.experiment_dir == exp_dir
    assert averaged.experiment_name == "1"

    # Original instance must not be modified
    assert len(metrics.fetch("loss")) == 3

    # Averaged instance must collapse the duplicate epoch entries
    avg_df = averaged.fetch("loss")
    assert len(avg_df) == 2
    expected_df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "split": ["train", "train"],
            "value": [15.0, 5.0],
        }
    )
    pd.testing.assert_frame_equal(avg_df, expected_df)


def test_parsed_metrics_average_without_duplicates() -> None:
    metrics = ParsedMetrics(x_col="epoch")
    df = pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.4, 0.2]})
    metrics.add("loss", df)

    averaged = metrics.average()
    pd.testing.assert_frame_equal(averaged.fetch("loss"), df)


def test_parsed_metrics_average_missing_x_col_in_dataframe() -> None:
    metrics = ParsedMetrics(x_col="epoch")
    # DataFrame without the primary x_col column
    df = pd.DataFrame({"step": [10, 20], "split": ["train", "train"], "value": [0.5, 0.3]})
    metrics.add("loss", df)

    averaged = metrics.average()
    pd.testing.assert_frame_equal(averaged.fetch("loss"), df)
