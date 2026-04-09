from pathlib import Path
from unittest.mock import patch
from griffe import logger

import hyperbench
from hyperbench import train
from hyperbench.train import negative_sampler
from hyperbench.train.markdown_logger import MarkdownTableLogger


def test_markdown_table_logger_basic_functions():
    experiment_name = "exp1"

    logger = MarkdownTableLogger(
        save_dir="dummy_dir",
        model_name="model_a",
        experiment_name=experiment_name,
        precision=2,
    )
    logger.clear(experiment_name)

    assert logger.name == "MarkdownTableLogger"
    assert logger.version == "model_a"
    assert logger.save_dir == "dummy_dir"
    assert logger.experiment_name == experiment_name


def test_log_hyperparams_is_noop():
    experiment_name = "exp_hparams"

    logger = MarkdownTableLogger(
        save_dir="dummy_dir",
        model_name="model_a",
        experiment_name=experiment_name,
    )
    logger.clear(experiment_name)

    assert logger.log_hyperparams({"lr": 0.001}) is None


def test_markdown_table_logger_log_metrics_accumulates_metrics():
    logger = MarkdownTableLogger(
        save_dir="dummy_dir",
        model_name="model_a",
        experiment_name="exp2",
    )

    logger.log_metrics({"test_auc": 0.80, "train_loss": 0.50})
    logger.log_metrics({"val_loss": 0.40})

    logger.finalize("success")
    store = logger.store
    assert store == {"model_a": {"test_auc": 0.80, "train_loss": 0.50, "val_loss": 0.40}}


def test_markdown_table_logger_finalize_calls_save_comparison_tables(tmp_path):
    experiment_name = "exp_comp"

    logger_a = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="mlp:mean",
        experiment_name="exp_comp",
        precision=3,
    )
    logger_b = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="gat:default",
        experiment_name="exp_comp",
        precision=3,
    )

    logger_a.log_metrics({"test_auc": 0.85, "train_loss": 0.42})
    logger_b.log_metrics({"test_auc": 0.82, "val_loss": 0.38})

    with patch.object(
        MarkdownTableLogger,
        "_MarkdownTableLogger__save_comparison_tables",
    ) as mock_save:
        logger_a.finalize("success")

    mock_save.assert_called_once()

    _, kwargs = mock_save.call_args
    assert kwargs["save_dir"] == Path(str(tmp_path)) / "comparison"
    assert kwargs["precision"] == 3

    assert kwargs["test_results"] == {
        "gat:default": {"test_auc": 0.82},
        "mlp:mean": {"test_auc": 0.85},
    }
    assert kwargs["train_results"] == {
        "mlp:mean": {"train_loss": 0.42},
    }
    assert kwargs["val_results"] == {
        "gat:default": {"val_loss": 0.38},
    }


def test_markdown_table_logger_finalize_does_not_save_when_no_results():
    experiment_name = "exp3"

    logger = MarkdownTableLogger(
        save_dir="dummy_dir",
        model_name="model_a",
        experiment_name=experiment_name,
    )

    with patch.object(
        MarkdownTableLogger,
        "_MarkdownTableLogger__save_comparison_tables",
    ) as mock_save:
        logger.finalize("success")

    mock_save.assert_not_called()


def test_markdown_table_logger_only_val_metrics():
    logger = MarkdownTableLogger(
        save_dir="dummy_dir",
        model_name="model_a",
        experiment_name="exp_val_only",
    )
    logger.log_metrics({"val_f1": 0.88, "epoch": 1})
    with patch.object(
        MarkdownTableLogger,
        "_MarkdownTableLogger__save_comparison_tables",
    ) as mock_save:
        logger.finalize("success")

    mock_save.assert_called_once()


def test_build_comparison_table_none_result():
    logger = MarkdownTableLogger(
        save_dir="dummy_dir",
        model_name="model_a",
        experiment_name="exp_table_none",
        precision=4,
    )

    with patch.object(
        MarkdownTableLogger, "_MarkdownTableLogger__build_comparison_table", return_value=""
    ) as mock_build:
        mock_build(results={}, precision=4)
        mock_build.assert_called_once_with(results={}, precision=4)


def test_save_comparison_tables_no_test_results(tmp_path):
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_save_no_test",
        precision=3,
    )

    logger.log_metrics({"train_loss": 0.254, "val_loss": 0.312})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "results.md").read_text()
    assert "## Test Results" not in content
    assert "## Train Results" in content
    assert "## Val Results" in content
    assert "| model_a | 0.312 |" in content
    assert "| model_a | 0.254 |" in content


def test_save_comparison_tables_no_train_results(tmp_path):
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_save_no_train",
        precision=3,
    )

    logger.log_metrics({"test_auc": 0.254, "val_loss": 0.312})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "results.md").read_text()
    assert "## Test Results" in content
    assert "## Train Results" not in content
    assert "## Val Results" in content


def test_save_comparison_tables_no_train_results(tmp_path):
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_save_no_train",
        precision=3,
    )

    # logger.log_metrics({"test_auc": 0.254})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "results.md").read_text()
    assert "## Test Results" in content
    assert "## Train Results" not in content
    assert "## Val Results" not in content


def test_build_comparison_table_correct_trail():
    logger = MarkdownTableLogger(
        save_dir="dummy_dir",
        model_name="model_a",
        experiment_name="exp_table_trail",
        precision=2,
    )

    results = {
        "model_a": {"test_auc": 0.9123, "train_loss": 0.254},
        "model_b": {"test_auc": 0.8821, "val_f1": 0.88},
    }

    real_build = getattr(
        logger,
        "_MarkdownTableLogger__build_comparison_table",
    )  # type: ignore[attr-defined]

    with patch.object(
        MarkdownTableLogger,
        "_MarkdownTableLogger__build_comparison_table",
        wraps=real_build,
    ) as mock_build:
        table = mock_build(results, precision=2)

    mock_build.assert_called_once_with(results, precision=2)

    expected_table = (
        "| Model | test_auc | train_loss | val_f1 |\n"
        "| --- | --- | --- | --- |\n"
        "| model_a | 0.91 | 0.25 | - |\n"
        "| model_b | 0.88 | - | 0.88 |"
    )

    assert table == expected_table
