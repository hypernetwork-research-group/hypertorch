import pytest

from textwrap import dedent
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from hypertorch.train import MarkdownTableLogger


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


def test_markdown_table_logger_rejects_negative_precision(tmp_path):
    with pytest.raises(ValueError, match="'precision' must be non-negative"):
        MarkdownTableLogger(
            save_dir=str(tmp_path),
            model_name="model_a",
            experiment_name="negative_precision",
            precision=-1,
        )


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

    logger.log_metrics({"test/auc": 0.80, "train/loss": 0.50})
    logger.log_metrics({"val/loss": 0.40})

    logger.finalize("success")
    store = logger.store
    assert store == {"model_a": {"test/auc": 0.80, "train/loss": 0.50, "val/loss": 0.40}}


def test_markdown_table_logger_does_not_log_or_save_on_nonzero_rank(tmp_path, monkeypatch):
    experiment_name = "exp_nonzero_rank"
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
    )
    logger.clear(experiment_name)
    monkeypatch.setattr(rank_zero_only, "rank", 1)

    logger.log_metrics({"test/auc": 0.80})
    logger.finalize("success")

    assert logger.store == {}
    assert not (tmp_path / "comparison").exists()


def test_clear_removes_metrics_only_for_requested_experiment(tmp_path):
    first_logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_clear_first",
    )
    second_logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_clear_second",
    )

    first_logger.log_metrics({"test/auc": 0.80, "train/loss": 0.50})
    second_logger.log_metrics({"test/auc": 0.90})
    first_logger.clear("exp_clear_first")

    assert first_logger.store == {}
    assert second_logger.store == {"model_b": {"test/auc": 0.90}}

    second_logger.clear("exp_clear_second")
    assert second_logger.store == {}


def test_destroy_removes_metrics_for_all_experiments(tmp_path):
    first_logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_destroy_first",
    )
    second_logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_destroy_second",
    )

    first_logger.log_metrics({"test/auc": 0.80})
    second_logger.log_metrics({"test/auc": 0.90})
    first_logger.destroy()

    assert first_logger.store == {}
    assert second_logger.store == {}


def test_markdown_table_logger_finalize_does_not_save_when_no_results(tmp_path):
    experiment_name = "exp3"

    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
    )
    logger.finalize("success")

    assert not (tmp_path / "comparison" / "overall.md").exists()


def test_save_comparison_tables_no_test_results(tmp_path):
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_save_no_test",
        precision=3,
    )

    logger.log_metrics({"train/loss": 0.254, "val/loss": 0.312})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "overall.md").read_text()
    assert "## Test Results" not in content
    assert "## Train Results" in content
    assert "## Val Results" in content
    assert r"| model\_a | 0.312 |" in content
    assert r"| model\_a | 0.254 |" in content


def test_save_comparison_tables_no_train_results(tmp_path):
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_save_no_train",
        precision=3,
    )

    logger.log_metrics({"test/auc": 0.254, "val/loss": 0.312})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "overall.md").read_text()
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
        "model_a": {"test/auc": 0.9123, "test/loss": 0.123, "train/loss": 0.254},
        "model_b": {"test/auc": 0.8821, "val/f1": 0.88},
    }

    logger_a.log_metrics(results["model_a"])
    logger_b.log_metrics(results["model_b"])

    logger_a.finalize("success")
    logger_b.finalize("success")
    expected_table = dedent(
        r"""
        ## Test Results

        | Model | test/auc | test/loss |
        | --- | --- | --- |
        | model\_a | 0.91 | 0.12 |
        | model\_b | 0.88 | - |

        ## Train Results

        | Model | train/loss |
        | --- | --- |
        | model\_a | 0.25 |

        ## Val Results

        | Model | val/f1 |
        | --- | --- |
        | model\_b | 0.88 |
        """
    ).lstrip()

    table = (tmp_path / "comparison" / "overall.md").read_text()
    assert table == expected_table


@pytest.mark.parametrize(
    "metrics, expect_train, expect_val",
    [
        ({"test/auc": 0.91}, None, None),  # neither train nor val
        ({"test/auc": 0.91, "train/loss": 0.25}, True, None),  # train only
        ({"test/auc": 0.91, "val/f1": 0.88}, None, True),  # val only
        ({"test/auc": 0.91, "train/loss": 0.25, "val/f1": 0.88}, True, True),  # both
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

    content = (tmp_path / "comparison" / "overall.md").read_text()

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

    assert not (tmp_path / "comparison" / "overall.md").exists()


def test_finalize_writes_train_and_val_sections_and_file(tmp_path):
    experiment_name = "exp_cover_train_val_write"
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        precision=3,
    )

    logger.log_metrics({"test/auc": 0.91, "train/loss": 0.25, "val/f1": 0.88})
    logger.finalize("success")

    result_path = tmp_path / "comparison" / "overall.md"
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

    logger.log_metrics({"test/auc": 0.91, "val/f1": 0.88})
    logger.finalize("success")

    result_path = tmp_path / "comparison" / "overall.md"
    assert result_path.exists()

    content = result_path.read_text()
    assert "## Train Results" not in content
    assert "## Val Results" in content


def test_all_md_files_written_to_comparison_subdir(tmp_path):
    experiment_name = "exp_comparison_subdir"
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        precision=3,
    )

    logger.log_metrics({"test/auc": 0.91, "train/loss": 0.25, "val/f1": 0.88})
    logger.finalize("success")

    comparison_dir = tmp_path / "comparison"
    assert comparison_dir.exists() and comparison_dir.is_dir()

    overall_md = comparison_dir / "overall.md"
    train_md = comparison_dir / "train.md"
    val_md = comparison_dir / "val.md"
    test_md = comparison_dir / "test.md"

    assert overall_md.exists()
    assert train_md.exists()
    assert val_md.exists()
    assert test_md.exists()


def test_markdown_table_escapes_model_and_metric_labels(tmp_path):
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name=r"model\|`*_{}[]()#+-.!~$&<>",
        experiment_name="exp_markdown_escape_labels",
        precision=2,
    )

    logger.log_metrics({r"test/metric\|`*_{}[]()#+-.!~$&<>": 1.23})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "test.md").read_text()
    expected_table = dedent(
        r"""
        ## Test Results

        | Model | test/metric\\\|\`\*\_\{\}\[\]\(\)\#\+\-\.\!\~\$&amp;&lt;&gt; |
        | --- | --- |
        | model\\\|\`\*\_\{\}\[\]\(\)\#\+\-\.\!\~\$&amp;&lt;&gt; | 1.23 |
        """
    ).lstrip()

    assert content == expected_table


def test_markdown_table_escapes_label_control_characters(tmp_path):
    logger = MarkdownTableLogger(
        save_dir=str(tmp_path),
        model_name="model\nnext\tend\rcarriage",
        experiment_name="exp_markdown_escape_control_chars",
        precision=2,
    )

    logger.log_metrics({"test/metric\nnext\tend\rcarriage": 1.23})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "test.md").read_text()

    assert "| Model | test/metric next end carriage |" in content
    assert "| model next end carriage | 1.23 |" in content
