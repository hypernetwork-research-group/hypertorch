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

    df, loaded_csv = parser.load_latest_metrics()
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
    logs_dir.mkdir()
    parser = LogParser(logs_dir)
    with pytest.raises(FileNotFoundError, match="No experiment folders found"):
        parser.find_latest_experiment_dir()


def test_logparser_load_csv_relative_and_absolute(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)

    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text("epoch,loss\n0,0.4\n")

    parser = LogParser(logs_dir)

    # Relative path as string
    df_rel_str, resolved_rel_str = parser.load_csv("experiment_0/metrics.csv")
    assert resolved_rel_str == csv_path
    assert not df_rel_str.empty

    # Relative path as Path object
    df_rel_path, resolved_rel_path = parser.load_csv(Path("experiment_0/metrics.csv"))
    assert resolved_rel_path == csv_path
    assert not df_rel_path.empty

    # Absolute path
    df_abs, resolved_abs = parser.load_csv(csv_path)
    assert resolved_abs == csv_path
    assert not df_abs.empty


def test_logparser_load_csv_invalid_extension(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir(parents=True)
    txt_file = logs_dir / "metrics.txt"
    txt_file.write_text("dummy")

    parser = LogParser(logs_dir)
    with pytest.raises(ValueError, match="is not a CSV file"):
        parser.load_csv("metrics.txt")


def test_logparser_load_csv_missing_file_or_directory(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir(parents=True)

    parser = LogParser(logs_dir)

    # File does not exist
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.load_csv("non_existent.csv")

    # Path exists but is a directory ending with .csv
    dir_named_csv = logs_dir / "folder.csv"
    dir_named_csv.mkdir()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.load_csv("folder.csv")
