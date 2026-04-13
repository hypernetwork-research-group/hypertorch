import pytest

from unittest.mock import MagicMock, patch
from hyperbench.train import MultiModelTrainer
from hyperbench.types import ModelConfig
from hyperbench.tests import new_mock_trainer

from hyperbench.train.latex_logger import LaTexTableConfig, LaTexTableLogger


@pytest.fixture
def mock_option_configs():
    options: LaTexTableConfig = {
        "table_caption": "Test Results",
        "sort_by": ["des", "asc"],
        "border": True,
    }
    return options


def test_latex_logger_basics(tmp_path, mock_option_configs):

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp1",
        options=mock_option_configs,
    )

    assert logger.name == "LaTexTableLogger"
    assert logger.version == "model_a"
    assert logger.store == {}
    assert logger.save_dir == str(tmp_path)
    assert logger.experiment_name == "exp1"


def test_latex_logger_log_hyperparams_is_noop(tmp_path, mock_option_configs):

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_hparams",
        options=mock_option_configs,
    )

    assert logger.log_hyperparams({"lr": 0.001}) is None


def test_latex_logger_log_metrics_accumulates_metrics(tmp_path, mock_option_configs):

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp2",
        options=mock_option_configs,
    )

    logger.log_metrics({"test_auc": 0.80, "train_loss": 0.50})
    logger.log_metrics({"val_loss": 0.40})

    logger.finalize("success")
    store = logger.store
    assert store == {"model_a": {"test_auc": 0.80, "train_loss": 0.50, "val_loss": 0.40}}


def test_markdown_table_logger_finalize_does_not_save_when_no_results(
    tmp_path, mock_option_configs
):
    experiment_name = "exp3"

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        options=mock_option_configs,
    )
    logger.finalize("success")

    assert not (tmp_path / "comparison" / "overall.tex").exists()
    assert not (tmp_path / "comparison" / "test.tex").exists()


def test_save_comparison_tables_no_test_results(tmp_path, mock_option_configs):
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp4",
        options=mock_option_configs,
    )

    logger.log_metrics({"train_loss": 0.50, "val_loss": 0.40})
    logger.finalize("success")

    assert (tmp_path / "comparison" / "overall.tex").exists()
    assert (tmp_path / "comparison" / "test.tex").exists()

    test_content = (tmp_path / "comparison" / "test.tex").read_text()
    assert test_content.strip() == ""


def test_save_comparison_tables_no_val_results(tmp_path, mock_option_configs):
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp5",
        options=mock_option_configs,
    )

    logger.log_metrics({"test_auc": 0.80, "train_loss": 0.50})
    logger.finalize("success")

    assert (tmp_path / "comparison" / "overall.tex").exists()
    assert (tmp_path / "comparison" / "test.tex").exists()


def test_save_comparison_tables_only_val_results(tmp_path, mock_option_configs):
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp6",
        options=mock_option_configs,
    )

    logger.log_metrics({"val_loss": 0.80})
    logger.finalize("success")

    assert (tmp_path / "comparison" / "overall.tex").exists()
    assert (tmp_path / "comparison" / "test.tex").exists()


def test_finalize_no_relevant_metrics_writes_no_file(tmp_path, mock_option_configs):
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_no_sections",
        options=mock_option_configs,
    )

    logger.log_metrics({"epoch": 1})  # ignored by split logic
    logger.finalize("success")

    assert not (tmp_path / "comparison" / "overall.tex").exists()
    assert not (tmp_path / "comparison" / "test.tex").exists()


def test_clear_removes_metrics_for_experiment(tmp_path, mock_option_configs):
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_clear",
        options=mock_option_configs,
    )

    logger.log_metrics({"test_auc": 0.80, "train_loss": 0.50})
    logger.clear("exp_clear")

    assert logger.store == {}


def test_finalize_writes_section_spacing_and_midrule_lines(tmp_path, mock_option_configs):
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_section_rules",
        options=mock_option_configs,
    )

    logger.log_metrics({"test_auc": 0.90, "train_loss": 0.40})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "overall.tex").read_text()

    # Two sections are emitted here: Test Results and Train Results.
    assert content.count(r"\addlinespace[3pt]") == 2
    assert content.count(r"\multicolumn") == 2
    assert content.count(r"\midrule") >= 2
