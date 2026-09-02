from pathlib import Path
import pandas as pd
import pytest

from hypertorch.train.logparser import LogParser


def test_logparser_find_and_load(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0" / "model" / "version_0"
    exp_dir.mkdir(parents=True)

    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text("epoch,step,train/loss\n0,1,0.5\n1,2,0.3\n")

    parser = LogParser(logs_dir)
    latest_exp = parser.find_latest_experiment_dir()
    assert latest_exp == logs_dir / "experiment_0"

    df, loaded_csv = parser.load_metrics()
    assert loaded_csv == csv_path
    assert isinstance(df, pd.DataFrame)
    assert "train/loss" in df.columns


def test_logparser_missing_dir_raises_error(tmp_path: Path) -> None:
    parser = LogParser(tmp_path / "non_existent_dir")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.find_latest_experiment_dir()


def test_logparser_empty_dir_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    (logs_dir / "experiment_0").mkdir(parents=True)

    parser = LogParser(logs_dir)
    with pytest.raises(FileNotFoundError, match="No CSV metric files found"):
        parser.find_latest_metrics_csv()


def test_logparser_no_subdirectories(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()  # Directory exists, but has no subdirectories
    parser = LogParser(logs_dir)
    with pytest.raises(FileNotFoundError, match="No experiment folders found"):
        parser.find_latest_experiment_dir()
