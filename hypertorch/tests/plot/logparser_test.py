from pathlib import Path
import pytest
from hypertorch.train import LogParser, ParsedMetrics


def test_logparser_find_and_parse_latest(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0" / "model" / "version_0"
    exp_dir.mkdir(parents=True)

    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text(
        "epoch,step,train/loss,val/loss,test/acc\n"
        "0,1,0.5,0.6,0.85\n"
        "1,2,0.3,0.4,\n"
    )

    parser = LogParser(logs_dir)
    assert parser.find_latest_experiment_dir() == logs_dir / "experiment_0"
    assert parser.find_latest_metrics_csv() == csv_path

    parsed = parser.parse()
    assert isinstance(parsed, ParsedMetrics)
    assert parsed.x_col == "epoch"
    assert parsed.csv_path == csv_path
    assert parsed.names() == ["acc", "loss"]

    loss_df = parsed.fetch("loss")
    assert set(loss_df["split"].unique()) == {"train", "val"}
    assert len(loss_df) == 4  # 2 train points + 2 val points


def test_logparser_parse_specific_paths(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_1"
    exp_dir.mkdir(parents=True)

    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text("step,train/loss\n1,0.4\n")

    parser = LogParser(logs_dir)

    # Relative string
    parsed_rel_str = parser.parse("experiment_1/metrics.csv")
    assert parsed_rel_str.x_col == "step"
    assert parsed_rel_str.csv_path == csv_path

    # Relative Path
    parsed_rel_path = parser.parse(Path("experiment_1/metrics.csv"))
    assert parsed_rel_path.csv_path == csv_path

    # Absolute Path
    parsed_abs = parser.parse(csv_path)
    assert parsed_abs.csv_path == csv_path


def test_logparser_fallback_tracking_columns(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)

    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text(
        "custom_index,unslashed_loss,train/epoch,all_nan\n"
        "0,0.5,10,\n"
        "1,0.4,20,\n"
    )

    parser = LogParser(logs_dir)
    parsed = parser.parse(csv_path)

    assert parsed.x_col == "custom_index"
    assert "unslashed_loss" in parsed
    assert "epoch" not in parsed
    assert "all_nan" not in parsed


def test_logparser_missing_base_dir_raises_error(tmp_path: Path) -> None:
    parser = LogParser(tmp_path / "non_existent_dir")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.find_latest_experiment_dir()


def test_logparser_empty_base_dir_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)
    with pytest.raises(FileNotFoundError, match="No experiment folders found"):
        parser.find_latest_experiment_dir()


def test_logparser_no_csv_in_experiment_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    (logs_dir / "experiment_0").mkdir(parents=True)
    parser = LogParser(logs_dir)
    with pytest.raises(FileNotFoundError, match="No CSV metric files found"):
        parser.find_latest_metrics_csv()


def test_logparser_invalid_extension(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir(parents=True)
    txt_file = logs_dir / "metrics.txt"
    txt_file.write_text("dummy")

    parser = LogParser(logs_dir)
    with pytest.raises(ValueError, match="is not a CSV file"):
        parser.parse("metrics.txt")


def test_logparser_empty_csv_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)
    parser = LogParser(logs_dir)

    empty_csv = exp_dir / "empty.csv"
    empty_csv.write_text("")
    with pytest.raises(ValueError, match="contains no data or columns"):
        parser.parse(empty_csv)

    header_only_csv = exp_dir / "header_only.csv"
    header_only_csv.write_text("epoch,loss\n")
    with pytest.raises(ValueError, match="contains no data or columns"):
        parser.parse(header_only_csv)


def test_logparser_all_nan_metric_is_skipped(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)

    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text(
        "epoch,loss,empty_metric\n"
        "0,0.5,\n"
        "1,0.3,\n"
    )

    parser = LogParser(logs_dir)
    parsed = parser.parse(csv_path)

    assert "loss" in parsed
    assert "empty_metric" not in parsed


def test_logparser_parse_path_already_relative_to_base_logs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    logs_dir = Path("hypertorch_logs")
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)

    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text("epoch,loss\n0,0.5\n")

    parser = LogParser(logs_dir)
    relative_nested = Path("hypertorch_logs/experiment_0/metrics.csv")

    parsed = parser.parse(relative_nested)
    assert "loss" in parsed


def test_logparser_missing_csv_file_raises_filenotfound(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir(parents=True)
    parser = LogParser(logs_dir)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.parse("non_existent_run.csv")

    fake_csv_dir = logs_dir / "not_a_file.csv"
    fake_csv_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.parse("not_a_file.csv")