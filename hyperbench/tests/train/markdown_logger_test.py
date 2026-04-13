from pathlib import Path
from textwrap import dedent
from griffe import logger
import pytest

import hyperbench
from hyperbench import train
from hyperbench.train import negative_sampler
from hyperbench.train.markdown_logger import MarkdownTableLogger


def test_markdown_table_logger_basic_functions(tmp_path):
    experiment_name = "exp1"

    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        precision=2,
    )
    logger.clear(experiment_name)

    assert logger.name == "MarkdownTableLogger"
    assert logger.version == "model_a"
    assert logger.save_dir == str(tmp_path)
    assert logger.experiment_name == experiment_name


def test_log_hyperparams_is_noop(tmp_path):
    experiment_name = "exp_hparams"

    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
    )
    logger.clear(experiment_name)

    assert logger.log_hyperparams({"lr": 0.001}) is None


def test_markdown_table_logger_log_metrics_accumulates_metrics(tmp_path):
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp2",
    )

    logger.log_metrics({"test_auc": 0.80, "train_loss": 0.50})
    logger.log_metrics({"val_loss": 0.40})

    logger.finalize("success")
    store = logger.store
    assert store == {"model_a": {"test_auc": 0.80, "train_loss": 0.50, "val_loss": 0.40}}


def test_markdown_table_logger_finalize_does_not_save_when_no_results(tmp_path):
    experiment_name = "exp3"

    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
    )
    logger.finalize("success")

    assert not (tmp_path / "comparison" / "results.md").exists()


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


def test_build_comparison_table_correct_trail(tmp_path):
    logger_a = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_table_trail",
        precision=2,
    )
    logger_b = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_table_trail",
        precision=2,
    )

    results = {
        "model_a": {"test_auc": 0.9123, "test_loss": 0.123, "train_loss": 0.254},
        "model_b": {"test_auc": 0.8821, "val_f1": 0.88},
    }

    logger_a.log_metrics(results["model_a"])
    logger_b.log_metrics(results["model_b"])

    logger_a.finalize("success")
    logger_b.finalize("success")
    expected_table = dedent(
        """
        ## Test Results

        | Model | test_auc | test_loss |
        | --- | --- | --- |
        | model_a | 0.91 | 0.12 |
        | model_b | 0.88 | - |

        ## Train Results

        | Model | train_loss |
        | --- | --- |
        | model_a | 0.25 |

        ## Val Results

        | Model | val_f1 |
        | --- | --- |
        | model_b | 0.88 |
        """
    ).lstrip()

    table = (tmp_path / "comparison" / "results.md").read_text()
    assert table == expected_table


@pytest.mark.parametrize(
    "metrics, expect_train, expect_val",
    [
        ({"test_auc": 0.91}, None, None),  # neither train nor val
        ({"test_auc": 0.91, "train_loss": 0.25}, True, None),  # train only
        ({"test_auc": 0.91, "val_f1": 0.88}, None, True),  # val only
        ({"test_auc": 0.91, "train_loss": 0.25, "val_f1": 0.88}, True, True),  # both
    ],
)
def test_finalize_train_val_section_branches(tmp_path, metrics, expect_train, expect_val):
    experiment_name = f"exp_branch_{expect_train}_{expect_val}"
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        precision=3,
    )

    logger.log_metrics(metrics)
    logger.finalize("success")

    content = (tmp_path / "comparison" / "results.md").read_text()

    assert "## Test Results" in content

    if expect_train:
        assert "## Train Results" in content
    else:
        assert "## Train Results" not in content

    if expect_val:
        assert "## Val Results" in content
    else:
        assert "## Val Results" not in content


def test_finalize_no_relevant_metrics_writes_no_file(tmp_path):
    experiment_name = "exp_no_sections"
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        precision=3,
    )

    logger.log_metrics({"epoch": 1})  # ignored by split logic
    logger.finalize("success")

    assert not (tmp_path / "comparison" / "results.md").exists()


def test_finalize_writes_train_and_val_sections_and_file(tmp_path):
    experiment_name = "exp_cover_train_val_write"
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        precision=3,
    )

    logger.log_metrics({"test_auc": 0.91, "train_loss": 0.25, "val_f1": 0.88})
    logger.finalize("success")

    result_path = tmp_path / "comparison" / "results.md"
    assert result_path.exists()

    content = result_path.read_text()
    assert "## Test Results" in content
    assert "## Train Results" in content
    assert "## Val Results" in content


def test_finalize_writes_val_section_when_train_missing(tmp_path):
    experiment_name = "exp_cover_val_only_branch"
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        precision=3,
    )

    logger.log_metrics({"test_auc": 0.91, "val_f1": 0.88})
    logger.finalize("success")

    result_path = tmp_path / "comparison" / "results.md"
    assert result_path.exists()

    content = result_path.read_text()
    assert "## Train Results" not in content
    assert "## Val Results" in content
