import pytest
import re

from textwrap import dedent
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from hypertorch.train import (
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


def test_latex_logger_rejects_negative_precision(tmp_path, mock_option_configs):
    with pytest.raises(ValueError, match="'precision' must be non-negative"):
        LaTexTableLogger(
            save_dir=str(tmp_path),
            model_name="model_a",
            experiment_name="negative_precision",
            precision=-1,
            options=mock_option_configs,
        )


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

    logger.log_metrics({"test/auc": 0.80, "train/loss": 0.50})
    logger.log_metrics({"val/loss": 0.40})

    logger.finalize("success")
    store = logger.store
    assert store == {"model_a": {"test/auc": 0.80, "train/loss": 0.50, "val/loss": 0.40}}


def test_latex_logger_does_not_log_or_save_on_nonzero_rank(
    tmp_path,
    mock_option_configs,
    monkeypatch,
):
    experiment_name = "exp_nonzero_rank"
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        options=mock_option_configs,
    )
    logger.clear(experiment_name)
    monkeypatch.setattr(rank_zero_only, "rank", 1)

    logger.log_metrics({"test/auc": 0.80})
    logger.finalize("success")

    assert logger.store == {}
    assert not (tmp_path / "comparison").exists()


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

    logger.log_metrics({"train/loss": 0.50, "val/loss": 0.40})
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

    logger.log_metrics({"test/auc": 0.80, "train/loss": 0.50})
    assert (
        colorize_metric_value(
            metric="test/auc",
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


def test_colorize_metric_value_rejects_invalid_sort_order():
    with pytest.raises(ValueError, match=re.escape("'sort_order' must be 'asc' or 'des'")):
        colorize_metric_value(
            metric="test/auc",
            value=0.8,
            text="0.8000",
            metric_bounds={"test/auc": (0.1, 0.9)},
            sort_order="invalid",
        )


def test_save_comparison_tables_only_val_results(tmp_path, mock_option_configs):
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp6",
        options=mock_option_configs,
    )

    logger.log_metrics({"val/loss": 0.80})
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


def test_clear_removes_metrics_only_for_requested_experiment(tmp_path, mock_option_configs):
    first_logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_clear_first",
        options=mock_option_configs,
    )
    second_logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_clear_second",
        options=mock_option_configs,
    )

    first_logger.log_metrics({"test/auc": 0.80, "train/loss": 0.50})
    second_logger.log_metrics({"test/auc": 0.90})
    first_logger.clear("exp_clear_first")

    assert first_logger.store == {}
    assert second_logger.store == {"model_b": {"test/auc": 0.90}}

    second_logger.clear("exp_clear_second")
    assert second_logger.store == {}


def test_destroy_removes_metrics_for_all_experiments(tmp_path, mock_option_configs):
    first_logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_destroy_first",
        options=mock_option_configs,
    )
    second_logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_b",
        experiment_name="exp_destroy_second",
        options=mock_option_configs,
    )

    first_logger.log_metrics({"test/auc": 0.80})
    second_logger.log_metrics({"test/auc": 0.90})
    first_logger.destroy()

    assert first_logger.store == {}
    assert second_logger.store == {}


def test_finalize_writes_section_spacing_and_midrule_lines(tmp_path, mock_option_configs):
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name="exp_section_rules",
        options=mock_option_configs,
    )

    logger.log_metrics({"test/auc": 0.90, "train/loss": 0.40})
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

    logger_a.log_metrics({"test/auc": 0.9123, "test/loss": 0.123, "train/loss": 0.254})
    logger_b.log_metrics({"test/auc": 0.8821, "val/f1": 0.88})

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
        Model & test/auc & test/loss \\
        model\_a & \cellcolor[HTML]{59FF59}\underline{0.9123} & """
        r"""\cellcolor[HTML]{59FF59}\underline{0.1230} \\
        model\_b & \cellcolor[HTML]{FF5959}0.8821 & - \\
        \hline
        \addlinespace[3pt]
        \multicolumn{3}{c}{\textbf{Train Results}} \\
        \midrule
        Model & train/loss &  \\
        model\_a & \cellcolor[HTML]{59FF59}\underline{0.2540} &  \\
        \hline
        \addlinespace[3pt]
        \multicolumn{3}{c}{\textbf{Val Results}} \\
        \midrule
        Model & val/f1 &  \\
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

    logger_a.log_metrics({"test/auc": 0.9123, "test/loss": 0.1234})
    logger_b.log_metrics({"test/auc": 0.8821})
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
        Model & test/auc & test/loss \\
        model\_a & \cellcolor[HTML]{59FF59}\underline{0.9123} & """
        r"""\cellcolor[HTML]{59FF59}\underline{0.1234} \\
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

    logger_a.log_metrics({"test/auc": 0.90, "test/loss": 0.50})
    logger_b.log_metrics({"test/auc": 0.80, "test/loss": 0.20})
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

    logger.log_metrics({"test/auc": 0.90})

    with pytest.raises(ValueError, match=re.escape("Invalid 'sort_by' value")):
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

    logger.log_metrics({"test/auc": 0.90, "test/loss": 0.40})
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

    logger_a.log_metrics({"test/auc": 0.91, "test/loss": 0.12})
    logger_b.log_metrics({"test/auc": 0.88})
    logger_a.finalize("success")
    logger_b.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()
    assert r"Model & test/auc & test/loss \\" in content
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

    logger_a.log_metrics({"test/note": "x", "test/auc": 0.91})
    logger_b.log_metrics({"test/note": "y", "test/auc": 0.88})
    logger_a.finalize("success")
    logger_b.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()
    assert r"test/note" in content
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

    logger.log_metrics({"test/auc": 0.90})
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

    logger.log_metrics({"test/auc": 0.90})
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

    logger.log_metrics({"test/auc": 0.90})
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

    logger.log_metrics({"test/auc": 0.90, "test/loss": 0.40})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()

    assert r"\begin{tabular}{l|c|c}" in content
    assert content.split(r"\end{tabular}")[0].rstrip().endswith(r"\hline")


def test_all_tex_files_written_to_comparison_subdir(tmp_path):
    experiment_name = "exp_comparison_subdir"
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name="model_a",
        experiment_name=experiment_name,
        options={"table_caption": "Comparison Table", "sort_by": ["asc"], "border": True},
    )

    logger.log_metrics({"test/auc": 0.91, "train/loss": 0.25, "val/f1": 0.88})
    logger.finalize("success")

    comparison_dir = tmp_path / "comparison"
    assert comparison_dir.exists() and comparison_dir.is_dir()

    overall_tex = comparison_dir / "overall.tex"
    train_tex = comparison_dir / "train.tex"
    val_tex = comparison_dir / "val.tex"
    test_tex = comparison_dir / "test.tex"

    assert overall_tex.exists()
    assert train_tex.exists()
    assert val_tex.exists()
    assert test_tex.exists()


def test_finalize_escapes_latex_labels_without_reescaping_inserted_commands(tmp_path):
    options: LaTexTableConfig = {
        "table_caption": r"caption\ / & % $ # _ { } ~ ^" + "\nnext\tend\rcarriage",
        "sort_by": ["asc"],
        "border": True,
    }
    logger = LaTexTableLogger(
        save_dir=str(tmp_path),
        model_name=r"model\ / & % $ # _ { } ~ ^" + "\nnext\tend\rcarriage",
        experiment_name="exp_latex_escape_labels",
        options=options,
    )

    logger.log_metrics({r"test/\&%$#_{}~^" + "\nnext\tend\rcarriage": 0.42})
    logger.finalize("success")

    content = (tmp_path / "comparison" / "test.tex").read_text()

    assert (
        r"\caption{caption\textbackslash{} / \& \% \$ \# \_ \{ \} "
        r"\textasciitilde{} \textasciicircum{} next end carriage}"
    ) in content
    assert (
        r"Model & test/\textbackslash{}\&\%\$\#\_\{\}\textasciitilde{}"
        r"\textasciicircum{} next end carriage \\"
    ) in content
    assert (
        r"model\textbackslash{} / \& \% \$ \# \_ \{ \} "
        r"\textasciitilde{} \textasciicircum{} next end carriage & "
    ) in content
    assert r"\slash\{\}" not in content
