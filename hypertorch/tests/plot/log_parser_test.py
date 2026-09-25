from pathlib import Path
import pytest
from hypertorch.train import LogParser
from hypertorch.types import ParsedMetrics


def test_logparser_lazy_properties_and_fallback(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    # Older experiment WITH a valid CSV
    exp_0 = logs_dir / "experiment_0"
    exp_0.mkdir()
    csv_0 = exp_0 / "metrics.csv"
    csv_0.write_text("epoch,train/loss\n0,0.5\n")

    # Newer experiment WITHOUT a CSV (crashed run)
    exp_1 = logs_dir / "experiment_1"
    exp_1.mkdir()

    parser = LogParser(logs_dir)

    # State before property access
    assert parser._latest_experiment_dir is None
    assert parser._latest_csv_file is None
    assert parser.base_logs_dir == logs_dir

    # Properties discover exp_0, correctly skipping exp_1
    assert parser.latest_experiment_dir == exp_0
    assert parser.latest_csv_file == csv_0

    # Ensure cached values are retained
    assert parser._latest_experiment_dir == exp_0
    assert parser._latest_csv_file == csv_0


def test_logparser_missing_base_logs_dir_raises_error(tmp_path: Path) -> None:
    parser = LogParser(tmp_path / "non_existent_folder")

    with pytest.raises(FileNotFoundError, match=r"Logs root directory .* does not exist"):
        _ = parser.latest_experiment_dir


def test_logparser_empty_base_logs_dir_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    parser = LogParser(logs_dir)

    with pytest.raises(FileNotFoundError, match="No experiment folders found inside"):
        _ = parser.latest_experiment_dir


def test_logparser_no_csv_in_any_experiment_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    (logs_dir / "experiment_0").mkdir(parents=True)
    (logs_dir / "experiment_1").mkdir(parents=True)

    parser = LogParser(logs_dir)

    with pytest.raises(
        FileNotFoundError, match="No CSV metric files found inside any experiment folder"
    ):
        _ = parser.latest_csv_file


def test_logparser_refresh_resets_caches_and_updates_dir(tmp_path: Path) -> None:
    dir_a = tmp_path / "logs_a"
    (dir_a / "experiment_0").mkdir(parents=True)
    csv_a = dir_a / "experiment_0" / "metrics.csv"
    csv_a.write_text("epoch,loss\n0,0.5\n")

    dir_b = tmp_path / "logs_b"
    (dir_b / "experiment_1").mkdir(parents=True)
    csv_b = dir_b / "experiment_1" / "metrics.csv"
    csv_b.write_text("epoch,loss\n0,0.2\n")

    parser = LogParser(dir_a)
    assert parser.latest_experiment_dir.name == "experiment_0"

    # Refresh with a new folder
    parser.refresh(dir_b)
    assert parser._latest_experiment_dir is None
    assert parser._latest_csv_file is None
    assert parser.base_logs_dir == dir_b
    assert parser.latest_experiment_dir.name == "experiment_1"
    assert parser.latest_csv_file == csv_b


def test_logparser_resolve_and_validate_path_errors(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    # Wrong extension
    txt_file = logs_dir / "metrics.txt"
    txt_file.write_text("dummy")
    with pytest.raises(ValueError, match="is not a CSV file"):
        parser.parse(txt_file)

    # File does not exist
    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.parse(logs_dir / "ghost.csv")


def test_logparser_empty_csv_file_states(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)
    parser = LogParser(logs_dir)

    # 1. 0-byte file (EmptyDataError)
    empty_csv = exp_dir / "empty.csv"
    empty_csv.write_text("")
    with pytest.raises(ValueError, match="is completely empty"):
        parser.parse(empty_csv)

    # 2. Header only, 0 data rows (Warning)
    header_only_csv = exp_dir / "header_only.csv"
    header_only_csv.write_text("epoch,loss\n")
    with pytest.warns(UserWarning, match="contains headers but no data rows"):
        parsed = parser.parse(header_only_csv)

    assert isinstance(parsed, ParsedMetrics)
    assert len(parsed) == 0


def test_logparser_parse_default_latest_run(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_5"
    exp_dir.mkdir(parents=True)

    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text("epoch,train/loss,val/loss\n0,0.8,0.7\n1,0.4,0.35\n")

    parser = LogParser(logs_dir)
    parsed = parser.parse()

    assert isinstance(parsed, ParsedMetrics)
    assert parsed.x_col == "epoch"
    assert parsed.csv_path == csv_path
    assert parsed.experiment_dir == exp_dir
    assert parsed.experiment_name == "5"
    assert "loss" in parsed

    loss_df = parsed.fetch("loss")
    assert set(loss_df["split"].unique()) == {"train", "val"}
    assert len(loss_df) == 4


def test_logparser_parse_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    logs_dir = Path("hypertorch_logs")
    exp_dir = logs_dir / "experiment_2"
    exp_dir.mkdir(parents=True)

    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text("step,train/acc\n100,0.75\n200,0.85\n")

    parser = LogParser(logs_dir)

    # Path relative to base_logs_dir
    parsed_rel = parser.parse("experiment_2/metrics.csv")
    assert parsed_rel.x_col == "step"
    assert parsed_rel.experiment_name == "2"
    assert parsed_rel.experiment_dir == exp_dir

    # Path already prefixed with base_logs_dir
    parsed_full_rel = parser.parse("hypertorch_logs/experiment_2/metrics.csv")
    assert parsed_full_rel.experiment_name == "2"
    assert parsed_full_rel.experiment_dir == exp_dir


def test_logparser_parse_external_path_with_experiment_ancestor(tmp_path: Path) -> None:
    external_dir = tmp_path / "external_runs" / "experiment_88" / "subfolder"
    external_dir.mkdir(parents=True)
    csv_file = external_dir / "run.csv"
    csv_file.write_text("epoch,train/loss\n0,0.5\n")

    parser = LogParser(tmp_path / "hypertorch_logs")
    parsed = parser.parse(csv_file)

    assert parsed.experiment_name == "88"
    assert parsed.experiment_dir == tmp_path / "external_runs" / "experiment_88"


def test_logparser_parse_external_path_without_experiment_ancestor(tmp_path: Path) -> None:
    external_dir = tmp_path / "custom_run"
    external_dir.mkdir(parents=True)
    csv_file = external_dir / "run.csv"
    csv_file.write_text("epoch,train/loss\n0,0.5\n")

    parser = LogParser(tmp_path / "hypertorch_logs")
    parsed = parser.parse(csv_file)

    assert parsed.experiment_name == "0"
    assert parsed.experiment_dir == external_dir


def test_logparser_tracking_column_fallback_and_nan_skipping(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)

    # 'idx' is used as custom x_col; 'all_nan' has no valid values and must be dropped
    csv_file = exp_dir / "metrics.csv"
    csv_file.write_text("idx,train/loss,all_nan\n0,0.5,\n1,0.3,\n")

    parser = LogParser(logs_dir)
    parsed = parser.parse(csv_file)

    assert parsed.x_col == "idx"
    assert "loss" in parsed
    assert "all_nan" not in parsed
    assert "idx" not in parsed  # x_col should not melt into metrics


def test_logparser_refresh_without_arguments(tmp_path: Path) -> None:
    """Covers line 48->50 where refresh() is called with updated_base_folder=None."""
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)
    csv_file = exp_dir / "metrics.csv"
    csv_file.write_text("epoch,loss\n0,0.5\n")

    parser = LogParser(logs_dir)
    assert parser.latest_experiment_dir == exp_dir

    # Refresh without providing an updated base folder
    parser.refresh()
    assert parser._latest_experiment_dir is None
    assert parser._latest_csv_file is None
    assert parser.base_logs_dir == logs_dir


def test_logparser_skips_metric_columns_matching_tracking_cols(tmp_path: Path) -> None:
    """Covers line 101->99 where a metric column resolves to a tracking column name."""
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)

    # 'val/epoch' strips to 'epoch', which is in tracking_cols and must be skipped
    csv_file = exp_dir / "metrics.csv"
    csv_file.write_text("epoch,loss,val/epoch\n0,0.5,0\n1,0.3,1\n")

    parser = LogParser(logs_dir)
    parsed = parser.parse(csv_file)

    assert "loss" in parsed
    assert "epoch" not in parsed
    assert "val/epoch" not in parsed
