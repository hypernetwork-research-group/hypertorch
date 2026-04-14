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


def test_build_comparison_table_returns_empty_for_empty_results(tmp_path, mock_option_configs):
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_build_empty",
        options=mock_option_configs,
    )

    results = {}
    logger.log_metrics(results)
    logger.finalize("success")

    assert not (tmp_path / "comparison" / "overall.tex").exists()


def test_build_comparison_table_formats_table_and_escapes_values(tmp_path, mock_option_configs):
    logger_a = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_build_full",
        options=mock_option_configs,
    )
    logger_b = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_build_full",
        options=mock_option_configs,
    )

    results = {
        "model_a": {"test_auc": 0.91},
        "model_b": {"test_auc": 0.8, "test_loss": 1.23},
    }

    logger_a.log_metrics(results["model_a"])
    logger_b.log_metrics(results["model_b"])
    logger_a.finalize("success")
    logger_b.finalize("success")

    table = (tmp_path / "comparison" / "overall.tex").read_text()
    assert r"\begin{tabular}{l|c|c}" in table
    assert r"\multicolumn{3}{c}{\textbf{Test Results}} \\" in table
    assert r"Model & test\_auc & test\_loss \\" in table
    assert r"model\_a & \cellcolor[HTML]{59FF59}\underline{0.9100} & - \\" in table
    assert (
        r"model\_b & \cellcolor[HTML]{FF5959}0.8000 & \cellcolor[HTML]{59FF59}\underline{1.2300} \\"
        in table
    )
    assert r"\end{tabular}" in table


def test_finalize_builds_test_table_trail_without_private_call(tmp_path, mock_option_configs):
    logger_a = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_table_trail_public",
        options=mock_option_configs,
    )
    logger_b = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_table_trail_public",
        options=mock_option_configs,
    )

    logger_a.log_metrics({"test_auc": 0.9123, "test_loss": 0.1234})
    logger_b.log_metrics({"test_auc": 0.8821})

    logger_a.finalize("success")
    logger_b.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()

    # Validate deterministic trail similarly to markdown logger tests.
    assert r"\multicolumn{3}{c}{\textbf{Test Results}} \\" in content
    assert r"Model & test\_auc & test\_loss \\" in content
    assert r"model\_a & " in content
    assert r"model\_b & " in content
    assert r" & - \\" in content


def test_finalize_applies_per_column_sort_order(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": "Per-column sort",
        "sort_by": ["des", "asc"],
        "border": True,
    }

    logger_a = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_sort_per_column",
        options=options,
    )
    logger_b = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_sort_per_column",
        options=options,
    )

    logger_a.log_metrics({"test_auc": 0.90, "test_loss": 0.50})
    logger_b.log_metrics({"test_auc": 0.80, "test_loss": 0.20})
    logger_a.finalize("success")
    logger_b.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()

    # test_auc uses "des" so higher value is best (model_a).
    assert r"model\_a & \cellcolor[HTML]{59FF59}\underline{0.9000}" in content
    # test_loss uses "asc" so lower value is best (model_b).
    assert (
        r"model\_b & \cellcolor[HTML]{FF5959}0.8000 & \cellcolor[HTML]{59FF59}\underline{0.2000}"
        in content
    )


def test_finalize_raises_on_invalid_sort_order(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": "Invalid sort",
        "sort_by": ["invalid"],
        "border": True,
    }

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_invalid_sort",
        options=options,
    )

    logger.log_metrics({"test_auc": 0.90})

    with pytest.raises(ValueError, match=r"Invalid sort_by value"):
        logger.finalize("success")


def test_finalize_border_false_uses_non_bordered_tabular(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": "No border",
        "sort_by": ["asc"],
        "border": False,
    }

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_no_border",
        options=options,
    )

    logger.log_metrics({"test_auc": 0.90, "test_loss": 0.40})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()
    assert r"\begin{tabular}{lcc}" in content
    assert r"\hline" not in content
    assert r"\toprule" in content
