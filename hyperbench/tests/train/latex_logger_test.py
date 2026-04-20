import pytest
from textwrap import dedent

from unittest.mock import MagicMock, patch
from hyperbench.train import MultiModelTrainer
from hyperbench.types import ModelConfig
from hyperbench.tests import new_mock_trainer

from hyperbench.train.latex_logger import (
    LaTexTableConfig,
    LaTexTableLogger,
    colorize_metric_value,
)


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
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp3",
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
    assert (
        colorize_metric_value(
            metric="test_auc",
            value=0.80,
            text="0.8000",
            metric_bounds=None,
            sort_order="asc",
        )
        == "0.8000"
    )
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


def test_finalize_writes_comprehensive_overall_table_trail(tmp_path, mock_option_configs):
    logger_a = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_overall_trail",
        options=mock_option_configs,
    )
    logger_b = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_overall_trail",
        options=mock_option_configs,
    )

    logger_a.log_metrics({"test_auc": 0.9123, "test_loss": 0.123, "train_loss": 0.254})
    logger_b.log_metrics({"test_auc": 0.8821, "val_f1": 0.88})

    logger_a.finalize("success")
    logger_b.finalize("success")

    expected_table = dedent(
        r"""
        % Requires: \usepackage{booktabs}
        % Requires: \usepackage[table]{xcolor}
        \begin{table}[htbp]
        \centering
        \caption{Test Results}
        \begin{tabular}{l|c|c}
        \hline
        \addlinespace[3pt]
        \multicolumn{3}{c}{\textbf{Test Results}} \\
        \midrule
        Model & test\_auc & test\_loss \\
        model\_a & \cellcolor[HTML]{59FF59}\underline{0.9123} & \cellcolor[HTML]{59FF59}\underline{0.1230} \\
        model\_b & \cellcolor[HTML]{FF5959}0.8821 & - \\
        \hline
        \addlinespace[3pt]
        \multicolumn{3}{c}{\textbf{Train Results}} \\
        \midrule
        Model & train\_loss &  \\
        model\_a & \cellcolor[HTML]{59FF59}\underline{0.2540} &  \\
        \hline
        \addlinespace[3pt]
        \multicolumn{3}{c}{\textbf{Val Results}} \\
        \midrule
        Model & val\_f1 &  \\
        model\_b & \cellcolor[HTML]{59FF59}\underline{0.8800} &  \\
        \hline
        \end{tabular}
        \end{table}
        """
    ).lstrip()

    table = (tmp_path / "comparison" / "overall.tex").read_text()
    assert table == expected_table


def test_finalize_writes_comprehensive_test_table_trail(tmp_path, mock_option_configs):
    logger_a = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_test_trail",
        options=mock_option_configs,
    )
    logger_b = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_test_trail",
        options=mock_option_configs,
    )

    logger_a.log_metrics({"test_auc": 0.9123, "test_loss": 0.1234})
    logger_b.log_metrics({"test_auc": 0.8821})
    logger_a.finalize("success")
    logger_b.finalize("success")

    expected_table = dedent(
        r"""
        % Requires: \usepackage{booktabs}
        % Requires: \usepackage[table]{xcolor}
        \begin{table}[htbp]
        \centering
        \caption{Test Results}
        \begin{tabular}{l|c|c}
        \hline
        \addlinespace[3pt]
        \multicolumn{3}{c}{\textbf{Test Results}} \\
        \midrule
        Model & test\_auc & test\_loss \\
        model\_a & \cellcolor[HTML]{59FF59}\underline{0.9123} & \cellcolor[HTML]{59FF59}\underline{0.1234} \\
        model\_b & \cellcolor[HTML]{FF5959}0.8821 & - \\
        \hline
        \end{tabular}
        \end{table}
        """
    ).lstrip()

    content = (tmp_path / "comparison" / "test.tex").read_text()
    assert content == expected_table


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
    assert r"\toprule" in content
    assert content.split(r"\end{tabular}")[0].rstrip().endswith(r"\hline")


def test_table_construction_with_sparse_metrics(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": "Sparse Metrics",
        "sort_by": ["des"],
        "border": True,
    }

    logger_a = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_sparse_metrics",
        options=options,
    )
    logger_b = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_sparse_metrics",
        options=options,
    )

    logger_a.log_metrics({"test_auc": 0.91, "test_loss": 0.12})
    logger_b.log_metrics({"test_auc": 0.88})
    logger_a.finalize("success")
    logger_b.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()
    assert r"Model & test\_auc & test\_loss \\" in content
    assert r"model\_b & " in content
    assert r" & - \\" in content


def test_best_value_underline_only_numeric(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": "Best Value Underline",
        "sort_by": ["des"],
        "border": True,
    }

    logger_a = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_best_value_underline",
        options=options,
    )
    logger_b = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_best_value_underline",
        options=options,
    )

    logger_a.log_metrics({"test_note": "x", "test_auc": 0.91})
    logger_b.log_metrics({"test_note": "y", "test_auc": 0.88})
    logger_a.finalize("success")
    logger_b.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()
    assert r"test\_note" in content
    assert r"\underline{0.9100}" in content
    assert r" & - \\" in content


def test_final_rule_converts_section_rule_for_non_bordered(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": "Non-Bordered Final Rule",
        "sort_by": ["asc"],
        "border": False,
    }

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_final_rule_non_bordered",
        options=options,
    )

    logger.log_metrics({"test_auc": 0.90})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()
    assert r"\midrule" in content
    assert content.split(r"\end{tabular}")[0].rstrip().endswith(r"\hline")


def test_empty_caption_string_omits_caption_command(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": "",
        "sort_by": ["asc"],
        "border": True,
    }

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_empty_caption",
        options=options,
    )

    logger.log_metrics({"test_auc": 0.90})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()
    assert r"\begin{table}[htbp]" in content
    assert r"\caption{" not in content
    assert r"\begin{tabular}" in content


def test_non_bordered_wraps_with_table_environment(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": "Table Environment",
        "sort_by": ["asc"],
        "border": False,
    }

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_table_environment",
        options=options,
    )

    logger.log_metrics({"test_auc": 0.90})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()

    assert content.split(r"\end{tabular}")[0].rstrip().endswith(r"\hline")
    assert r"\begin{table}[htbp]" in content
    assert r"\centering" in content


def test_finalize_bordered_table_ends_with_hline_before_tabular_close(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": "Bordered end rule",
        "sort_by": ["asc"],
        "border": True,
    }

    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_border_end_rule",
        options=options,
    )

    logger.log_metrics({"test_auc": 0.90, "test_loss": 0.40})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()

    assert r"\begin{tabular}{l|c|c}" in content
    assert content.split(r"\end{tabular}")[0].rstrip().endswith(r"\hline")
