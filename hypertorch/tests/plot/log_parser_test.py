import os
from pathlib import Path
import time
from typing import Any
import pytest

from hypertorch.train import LogParser
from hypertorch.types import ParsedMetrics


""" Base Directory Validation & Lifecycle Tests """


def test_logparser_init_validates_base_dir(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    parser = LogParser(logs_dir)
    assert parser.base_logs_dir == logs_dir.resolve()
    assert parser.directory_paths == []
    assert parser.dir_to_csvs == {}


def test_logparser_missing_base_logs_dir_raises_error(tmp_path: Path) -> None:
    non_existent = tmp_path / "ghost_logs"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        LogParser(non_existent)


def test_logparser_base_dir_not_a_directory_raises_error(tmp_path: Path) -> None:
    dummy_file = tmp_path / "not_a_dir.txt"
    dummy_file.write_text("hello")
    with pytest.raises(NotADirectoryError, match="is not a directory"):
        LogParser(dummy_file)


def test_logparser_move_base_dir_retains_queued_paths(tmp_path: Path) -> None:
    dir_a = tmp_path / "logs_a"
    dir_a.mkdir()
    exp_0 = dir_a / "experiment_0"
    exp_0.mkdir()
    csv_0 = exp_0 / "metrics.csv"
    csv_0.write_text("epoch,loss\n0,0.5\n")

    dir_b = tmp_path / "logs_b"
    dir_b.mkdir()

    parser = LogParser(dir_a)
    parser.discover_from_latest_dir(num=1)
    assert len(parser.directory_paths) == 1

    parser.move_base_dir(dir_b)
    assert parser.base_logs_dir == dir_b.resolve()
    assert len(parser.directory_paths) == 1
    assert parser.directory_paths[0] == exp_0.resolve()


def test_logparser_move_base_dir_validates_target(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.move_base_dir(tmp_path / "missing_folder")

    dummy_file = tmp_path / "file.txt"
    dummy_file.write_text("x")
    with pytest.raises(NotADirectoryError, match="is not a directory"):
        parser.move_base_dir(dummy_file)


""" Discovery & Low-Level Traversal Tests """


def test_logparser_safe_mtime_oserror() -> None:
    class MockDirEntry:
        def stat(self) -> None:
            raise OSError("Read error")

    mock_entry: Any = MockDirEntry()
    assert LogParser._safe_mtime(mock_entry) == 0.0


def test_logparser_iter_csvs_max_depth(tmp_path: Path) -> None:
    deep_dir = tmp_path / "a" / "b" / "c" / "d" / "e"
    deep_dir.mkdir(parents=True)
    csv_file = deep_dir / "metrics.csv"
    csv_file.write_text("epoch,loss\n0,0.1\n")

    # max_depth=2 stops search before descending to depth 5
    found = list(LogParser._iter_csvs(tmp_path, max_depth=2))
    assert found == []


def test_logparser_iter_csvs_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_scandir(_: str) -> None:
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "scandir", mock_scandir)
    found = list(LogParser._iter_csvs(tmp_path))
    assert found == []


def test_logparser_register_csv_branches(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)
    csv_1 = exp_dir / "metrics_1.csv"
    csv_2 = exp_dir / "metrics_2.csv"
    csv_1.write_text("epoch,loss\n0,0.1\n")
    csv_2.write_text("epoch,loss\n0,0.2\n")

    parser = LogParser(logs_dir)
    parser._register_csv(csv_1)
    parser._register_csv(csv_2)
    parser._register_csv(csv_1)

    assert len(parser.dir_to_csvs[exp_dir.resolve()]) == 2


def test_logparser_discover_from_latest_dir_num_validation(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    with pytest.raises(ValueError, match="requires num >= 1"):
        parser.discover_from_latest_dir(num=0)


def test_logparser_discover_from_latest_dir_empty_root_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    with pytest.raises(FileNotFoundError, match="No experiment folders found inside"):
        parser.discover_from_latest_dir(num=1)


def test_logparser_discover_from_latest_dir_no_csvs_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    (logs_dir / "experiment_0").mkdir(parents=True)
    parser = LogParser(logs_dir)

    with pytest.raises(
        FileNotFoundError,
        match="No CSV metric files found inside any experiment folder",
    ):
        parser.discover_from_latest_dir(num=1)


def test_logparser_discover_from_latest_dir_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    def mock_scandir(_: Path) -> None:
        raise OSError("Filesystem unreadable")

    monkeypatch.setattr(os, "scandir", mock_scandir)
    with pytest.raises(OSError, match="Failed to scan directory"):
        parser.discover_from_latest_dir(num=1)


def test_logparser_discover_from_latest_dir_orders_and_includes_all_models(
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    # Exp 0 (Older)
    exp_0 = logs_dir / "experiment_0" / "mlp" / "version_0"
    exp_0.mkdir(parents=True)
    csv_0 = exp_0 / "metrics.csv"
    csv_0.write_text("epoch,loss\n0,0.5\n")

    time.sleep(0.05)

    # Exp 1 (Newer) with two models and adjacent checkpoints
    exp_1 = logs_dir / "experiment_1"
    run_mlp = exp_1 / "mlp" / "version_0"
    run_gat = exp_1 / "gat" / "version_0"
    run_mlp.mkdir(parents=True)
    run_gat.mkdir(parents=True)

    csv_mlp = run_mlp / "metrics.csv"
    csv_gat = run_gat / "metrics.csv"
    csv_mlp.write_text("epoch,loss\n0,0.2\n")
    csv_gat.write_text("epoch,loss\n0,0.15\n")

    ckpt_dir = run_mlp / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "ignored.csv").write_text("epoch,loss\n0,999.0\n")

    parser = LogParser(logs_dir)
    parser.discover_from_latest_dir(num=1)

    # Both models within the latest folder must be discovered
    assert len(parser.directory_paths) == 2
    assert set(parser.directory_paths) == {run_mlp.resolve(), run_gat.resolve()}
    assert exp_0.resolve() not in parser.directory_paths


def test_logparser_discover_from_latest_dir_multiple_folders(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    exp_0 = logs_dir / "experiment_0"
    exp_0.mkdir()
    (exp_0 / "metrics.csv").write_text("epoch,loss\n0,0.5\n")

    time.sleep(0.05)

    exp_1 = logs_dir / "experiment_1"
    exp_1.mkdir()
    (exp_1 / "metrics.csv").write_text("epoch,loss\n0,0.2\n")

    parser = LogParser(logs_dir)
    parser.discover_from_latest_dir(num=2)

    assert len(parser.directory_paths) == 2
    assert parser.directory_paths[0] == exp_1.resolve()
    assert parser.directory_paths[1] == exp_0.resolve()


def test_logparser_discover_latest_metrics_num_validation(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    with pytest.raises(ValueError, match="requires num >= 1"):
        parser.discover_latest_metrics(num=0)


def test_logparser_discover_latest_metrics_empty_root_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    with pytest.raises(FileNotFoundError, match="No experiment folders found inside"):
        parser.discover_latest_metrics(num=1)


def test_logparser_discover_latest_metrics_no_csvs_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    (logs_dir / "experiment_0").mkdir(parents=True)
    parser = LogParser(logs_dir)

    with pytest.raises(
        FileNotFoundError,
        match="No CSV metric files found inside any experiment folder",
    ):
        parser.discover_latest_metrics(num=1)


def test_logparser_discover_latest_metrics_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    def mock_scandir(_: Path) -> None:
        raise OSError("Filesystem unreadable")

    monkeypatch.setattr(os, "scandir", mock_scandir)
    with pytest.raises(OSError, match="Failed to scan directory"):
        parser.discover_latest_metrics(num=1)


def test_logparser_discover_latest_metrics_halts_at_num_csvs(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    exp_1 = logs_dir / "experiment_1"
    run_a = exp_1 / "mlp" / "version_0"
    run_b = exp_1 / "gat" / "version_0"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    csv_a = run_a / "metrics.csv"
    csv_b = run_b / "metrics.csv"
    csv_a.write_text("epoch,loss\n0,0.1\n")
    csv_b.write_text("epoch,loss\n0,0.2\n")

    parser = LogParser(logs_dir)

    # Asking for num=1 halts immediately after the first CSV is found
    parser.discover_latest_metrics(num=1)
    all_csvs = [csv for csvs in parser.dir_to_csvs.values() for csv in csvs]
    assert len(all_csvs) == 1


def test_logparser_discover_latest_metrics_fewer_csvs_than_requested(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    exp_1 = logs_dir / "experiment_1"
    run_a = exp_1 / "mlp" / "version_0"
    run_b = exp_1 / "gat" / "version_0"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    csv_a = run_a / "metrics.csv"
    csv_b = run_b / "metrics.csv"
    csv_a.write_text("epoch,loss\n0,0.1\n")
    csv_b.write_text("epoch,loss\n0,0.2\n")

    parser = LogParser(logs_dir)

    # Asking for num=5 when only 2 exist exercises both the continuation and clean exit branches
    parser.discover_latest_metrics(num=5)
    all_csvs = [csv for csvs in parser.dir_to_csvs.values() for csv in csvs]
    assert len(all_csvs) == 2


def test_logparser_discover_directory_by_name_and_relative_path(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    exp_dir = logs_dir / "experiment_3"
    run_a = exp_dir / "model_a" / "version_0"
    run_b = exp_dir / "model_b" / "version_0"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    csv_a = run_a / "metrics.csv"
    csv_b = run_b / "metrics.csv"
    csv_a.write_text("epoch,loss\n0,0.1\n")
    csv_b.write_text("epoch,loss\n0,0.2\n")

    parser = LogParser(logs_dir)

    parser.discover_directory("experiment_3")
    assert len(parser.directory_paths) == 2
    assert set(parser.directory_paths) == {run_a.resolve(), run_b.resolve()}


def test_logparser_discover_directory_error_handling(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.discover_directory("ghost_dir")

    file_path = logs_dir / "test.txt"
    file_path.write_text("data")
    with pytest.raises(NotADirectoryError, match="is not a directory"):
        parser.discover_directory(file_path)

    empty_exp = logs_dir / "empty_exp"
    empty_exp.mkdir()
    with pytest.raises(FileNotFoundError, match="No CSV files found inside"):
        parser.discover_directory(empty_exp)


""" Path Resolution & Caching Tests """


def test_logparser_get_csvs_for_ancestor_and_leaf_directory(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_4"
    leaf_dir = exp_dir / "mlp" / "version_test"
    leaf_dir.mkdir(parents=True)

    csv_path = leaf_dir / "metrics.csv"
    csv_path.write_text("epoch,loss\n0,0.3\n")

    parser = LogParser(logs_dir)
    parser.discover_directory("experiment_4")

    assert parser.get_csvs_for_directory("experiment_4") == [csv_path.resolve()]
    assert parser.get_csvs_for_directory(leaf_dir) == [csv_path.resolve()]


def test_logparser_get_csvs_for_directory_on_demand_cache_miss(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_7"
    exp_dir.mkdir(parents=True)
    csv_path = exp_dir / "metrics.csv"
    csv_path.write_text("epoch,loss\n0,0.1\n")

    parser = LogParser(logs_dir)
    assert parser.dir_to_csvs == {}

    csvs = parser.get_csvs_for_directory("experiment_7")
    assert csvs == [csv_path.resolve()]
    assert exp_dir.resolve() in parser.dir_to_csvs


def test_logparser_get_csvs_for_directory_errors(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.get_csvs_for_directory("non_existent_dir")

    not_a_dir = logs_dir / "sample.txt"
    not_a_dir.write_text("text")
    with pytest.raises(NotADirectoryError, match="is not a directory"):
        parser.get_csvs_for_directory(not_a_dir)

    empty_dir = logs_dir / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No CSV files found inside directory"):
        parser.get_csvs_for_directory(empty_dir)


""" Parsing Functionality Tests """


def test_logparser_parse_all_and_parse_queue(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_10"
    exp_dir.mkdir(parents=True)
    csv_file = exp_dir / "metrics.csv"
    csv_file.write_text("epoch,train/loss,val/loss\n0,0.8,0.7\n1,0.4,0.35\n")

    parser = LogParser(logs_dir)
    parser.discover_directory("experiment_10")

    parsed_list = parser.parse_all()
    assert len(parsed_list) == 1
    metrics = parsed_list[0]

    assert isinstance(metrics, ParsedMetrics)
    assert metrics.x_col == "epoch"
    assert metrics.experiment_name == "10"
    assert "loss" in metrics


def test_logparser_parse_all_empty_queue_raises_error(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()
    parser = LogParser(logs_dir)

    with pytest.raises(ValueError, match="No directories queued in LogParser"):
        parser.parse_all()


def test_logparser_parse_variadic_directories(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_a = logs_dir / "experiment_1"
    exp_b = logs_dir / "experiment_2"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)

    (exp_a / "metrics.csv").write_text("epoch,loss\n0,0.5\n")
    (exp_b / "metrics.csv").write_text("epoch,loss\n0,0.2\n")

    parser = LogParser(logs_dir)

    with pytest.raises(ValueError, match="No directory paths provided to parse"):
        parser.parse()

    parsed_list = parser.parse("experiment_1", "experiment_2")
    assert len(parsed_list) == 2
    assert parsed_list[0].experiment_name == "1"
    assert parsed_list[1].experiment_name == "2"


def test_logparser_parse_from_path_file_and_directory(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_8"
    exp_dir.mkdir(parents=True)
    csv_file = exp_dir / "metrics.csv"
    csv_file.write_text("step,train/loss\n100,0.5\n")

    parser = LogParser(logs_dir)

    parsed_file = parser.parse_from_path(csv_file)
    assert parsed_file.x_col == "step"

    parsed_dir = parser.parse_from_path(exp_dir)
    assert parsed_dir.x_col == "step"


def test_logparser_parse_from_path_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    logs_dir = Path("hypertorch_logs")
    exp_dir = logs_dir / "experiment_9"
    exp_dir.mkdir(parents=True)
    csv_file = exp_dir / "metrics.csv"
    csv_file.write_text("epoch,train/loss\n0,0.4\n")

    parser = LogParser(logs_dir)

    parsed_dir = parser.parse_from_path("experiment_9")
    assert parsed_dir.experiment_name == "9"

    parsed_file = parser.parse_from_path("experiment_9/metrics.csv")
    assert parsed_file.experiment_name == "9"


def test_logparser_parse_flat_experiment_csv(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_12"
    exp_dir.mkdir(parents=True)
    csv_file = exp_dir / "metrics.csv"
    csv_file.write_text("epoch,loss\n0,0.1\n")

    parser = LogParser(logs_dir)
    parsed = parser.parse_from_path(csv_file)

    assert parsed.experiment_name == "12"


def test_logparser_parse_single_version_subfolder(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    version_dir = logs_dir / "experiment_6" / "version_0"
    version_dir.mkdir(parents=True)
    csv_file = version_dir / "metrics.csv"
    csv_file.write_text("epoch,loss\n0,0.1\n")

    parser = LogParser(logs_dir)
    parsed = parser.parse_from_path(csv_file)
    assert parsed.experiment_name == "6_0"


def test_logparser_parse_external_path_with_experiment_ancestor(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    external_dir = tmp_path / "external_logs" / "experiment_42" / "model_x" / "version_1"
    external_dir.mkdir(parents=True)
    csv_file = external_dir / "metrics.csv"
    csv_file.write_text("epoch,train/loss\n0,0.5\n")

    parser = LogParser(logs_dir)
    parsed = parser.parse_from_path(csv_file)

    assert parsed.experiment_name == "42_model_x_1"
    assert parsed.experiment_dir == tmp_path / "external_logs" / "experiment_42"


def test_logparser_parse_external_path_without_experiment_ancestor(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    logs_dir.mkdir()

    external_dir = tmp_path / "custom_run_folder"
    external_dir.mkdir(parents=True)
    csv_file = external_dir / "metrics.csv"
    csv_file.write_text("epoch,train/loss\n0,0.5\n")

    parser = LogParser(logs_dir)
    parsed = parser.parse_from_path(csv_file)

    assert parsed.experiment_name == "custom_run_folder"
    assert parsed.experiment_dir == external_dir


def test_logparser_run_identifier_sanitization_and_complex_naming(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    run_dir = logs_dir / "checkpoint-predict" / "mlp" / "version_checkpoint-predict"
    run_dir.mkdir(parents=True)

    csv_file = run_dir / "eval_metrics.csv"
    csv_file.write_text("epoch,train/acc\n0,0.95\n")

    parser = LogParser(logs_dir)
    metrics = parser.parse_from_path(csv_file)

    assert ":" not in metrics.experiment_name
    assert "/" not in metrics.experiment_name
    expected_name = "checkpoint-predict_mlp_checkpoint-predict_eval_metrics"
    assert metrics.experiment_name == expected_name


def test_logparser_csv_validation_errors(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)
    parser = LogParser(logs_dir)

    txt_file = exp_dir / "metrics.txt"
    txt_file.write_text("dummy")
    with pytest.raises(ValueError, match="is not a CSV file"):
        parser.parse_from_path(txt_file)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        parser.parse_from_path(exp_dir / "ghost.csv")

    empty_csv = exp_dir / "empty.csv"
    empty_csv.write_text("")
    with pytest.raises(ValueError, match="is completely empty"):
        parser.parse_from_path(empty_csv)

    header_only_csv = exp_dir / "header_only.csv"
    header_only_csv.write_text("epoch,loss\n")
    with pytest.warns(UserWarning, match="contains headers but no data rows"):
        parsed = parser.parse_from_path(header_only_csv)

    assert isinstance(parsed, ParsedMetrics)
    assert len(parsed) == 0


def test_logparser_tracking_column_fallback_and_nan_skipping(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)

    csv_file = exp_dir / "metrics.csv"
    csv_file.write_text("idx,train/loss,all_nan\n0,0.5,\n1,0.3,\n")

    parser = LogParser(logs_dir)
    parsed = parser.parse_from_path(csv_file)

    assert parsed.x_col == "idx"
    assert "loss" in parsed
    assert "all_nan" not in parsed
    assert "idx" not in parsed


def test_logparser_skips_metric_columns_matching_tracking_cols(tmp_path: Path) -> None:
    logs_dir = tmp_path / "hypertorch_logs"
    exp_dir = logs_dir / "experiment_0"
    exp_dir.mkdir(parents=True)

    csv_file = exp_dir / "metrics.csv"
    csv_file.write_text("epoch,loss,val/epoch\n0,0.5,0\n1,0.3,1\n")

    parser = LogParser(logs_dir)
    parsed = parser.parse_from_path(csv_file)

    assert "loss" in parsed
    assert "epoch" not in parsed
    assert "val/epoch" not in parsed