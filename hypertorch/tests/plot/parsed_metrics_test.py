from pathlib import Path
import pandas as pd
import pytest
from hypertorch.train import ParsedMetrics


def test_parsed_metrics_initialization() -> None:
    csv_file = Path("dummy/path/metrics.csv")
    metrics = ParsedMetrics(x_col="epoch", csv_path=csv_file)

    assert metrics.x_col == "epoch"
    assert metrics.csv_path == csv_file
    assert len(metrics) == 0
    assert metrics.names() == []
    assert metrics.all() == {}


def test_parsed_metrics_add_and_fetch() -> None:
    metrics = ParsedMetrics(x_col="epoch")
    df_loss = pd.DataFrame({"epoch": [0, 1], "split": ["train", "train"], "value": [0.5, 0.3]})
    df_acc = pd.DataFrame({"epoch": [0, 1], "split": ["val", "val"], "value": [0.8, 0.9]})

    metrics.add("loss", df_loss)
    metrics.add("accuracy", df_acc)

    assert len(metrics) == 2
    assert metrics.names() == ["accuracy", "loss"]
    assert "loss" in metrics
    assert "f1" not in metrics

    # Test fetch and __getitem__
    pd.testing.assert_frame_equal(metrics.fetch("loss"), df_loss)
    pd.testing.assert_frame_equal(metrics["accuracy"], df_acc)


def test_parsed_metrics_fetch_missing_raises_key_error() -> None:
    metrics = ParsedMetrics(x_col="step")
    metrics.add("loss", pd.DataFrame())

    with pytest.raises(KeyError, match="Metric 'unknown' not found"):
        metrics.fetch("unknown")

    with pytest.raises(KeyError, match="Metric 'missing' not found"):
        _ = metrics["missing"]


def test_parsed_metrics_all_returns_shallow_copy() -> None:
    metrics = ParsedMetrics(x_col="epoch")
    df = pd.DataFrame({"value": [1.0]})
    metrics.add("loss", df)

    all_dict = metrics.all()
    assert "loss" in all_dict

    # Mutating returned dictionary must not mutate internal state
    all_dict["injected"] = pd.DataFrame()
    assert "injected" not in metrics


def test_parsed_metrics_iteration_and_repr() -> None:
    metrics = ParsedMetrics(x_col="epoch")
    metrics.add("loss", pd.DataFrame())
    metrics.add("f1", pd.DataFrame())

    # Iteration yields sorted names
    iterated_names = list(metrics)
    assert iterated_names == ["f1", "loss"]

    # Repr formatting
    repr_str = repr(metrics)
    assert "x_col='epoch'" in repr_str
    assert "metrics=['f1', 'loss']" in repr_str