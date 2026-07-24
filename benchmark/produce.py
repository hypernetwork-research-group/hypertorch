import os
import pandas as pd


def clean_model_name(model_name: str) -> str:
    model_name = model_name.replace("\\_", "_")
    if "_" in model_name:
        model_name = "_".join(model_name.split("_")[:-1])
    return model_name


def create_results_csv(folder: str, output_folder: str):
    path = os.path.join(os.path.dirname(__file__), folder)
    output_path = os.path.join(path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for dataset_name in os.listdir(path):
        if dataset_name == "output":
            continue
        dataset_df = pd.DataFrame(
            columns=pd.Index(
                ["model", "accuracy", "roc-auc", "precision"],
                name="metric",
            )
        )
        dataset_path = os.path.join(path, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        for experiment in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, experiment)
            if not os.path.isdir(model_path):
                continue
            latex_test = os.path.join(model_path, "comparison", "test.tex")

            print(f"Processing latex_test {latex_test}...")
            if os.path.exists(latex_test):
                with open(latex_test) as f:
                    content = f.read()
                    # print content between \midrule and \hline
                    start = content.find("\\midrule")
                    end = content.find("\\hline", start)
                    if start != -1 and end != -1:
                        content = content[start + len("\\midrule") : end].strip()
                        for line in content.splitlines():
                            if line.strip() == "":
                                continue
                            split_line = line.split("&")
                            model = split_line[0].strip()
                            if model == "Model":
                                continue

                            model = clean_model_name(split_line[0].strip())
                            accuracy = clean_metric(split_line[1].strip())
                            roc_auc = clean_metric(split_line[2].strip())
                            precision = clean_metric(split_line[3].strip())
                            dataset_df.loc[len(dataset_df)] = [model, accuracy, roc_auc, precision]
        dataset_df.to_csv(os.path.join(output_path, f"{dataset_name}_metrics.csv"))


def clean_metric(value: str) -> float:
    """Clean the metric value by removing LaTeX formatting and converting to float."""
    if "\\cellcolor" in value and "\\underline" in value:
        value = value.rsplit("\\underline{", maxsplit=1)[-1].split("}", maxsplit=1)[0].strip()
    if "\\cellcolor" in value:
        value = value.split("}")[-1].strip()
    try:
        return float(value)
    except ValueError:
        return float("nan")  # Return NaN if conversion fails


def build_table_with_std(folder: str):
    path = os.path.join(os.path.dirname(__file__), folder)
    for csvs in os.listdir(path):
        if csvs.endswith("metrics.csv"):
            print(f"Processing {csvs}...")
            df = pd.read_csv(os.path.join(path, csvs), index_col=0)
            for model in df["model"].unique():
                for metric in ["accuracy", "roc-auc", "precision"]:
                    mean = df[df["model"] == model][metric].mean()
                    std = df[df["model"] == model][metric].std()
                    df.loc[df["model"] == model, metric] = f"{mean:.4f} ± {std:.4f}"
            df = df.drop_duplicates(subset=["model"])
            df.to_csv(os.path.join(path, f"{csvs.split('_metrics.csv')[0]}_metrics_mean_std.csv"))


def create_latex_table(dataset_name: str, task: str, folder: str, output_folder: str):
    path = os.path.join(os.path.dirname(__file__), folder)
    setup_latex = (
        r"""
\begin{table}[htbp]
\label{tab:"""
        + task
        + r"""_results_"""
        + dataset_name
        + r"""}
\centering
\caption{Results for """
        + task
        + r""" - dataset """
        + dataset_name
        + r""".}
\begin{tabular}{lccc}
\toprule
\addlinespace[3pt]
\multicolumn{4}{c}{\textbf{Test Results}} \\
\midrule
Model & accuracy & roc-auc & precision \\
"""
    )

    df = pd.read_csv(os.path.join(path, f"{dataset_name}_metrics_mean_std.csv"), index_col=0)
    lines = []
    accuracies = df["accuracy"].apply(lambda x: float(x.split("±")[0].strip()))
    best_three_accuracy = sorted(accuracies.nlargest(3).values, reverse=True)
    rocs_auc = df["roc-auc"].apply(lambda x: float(x.split("±")[0].strip()))
    best_three_roc_auc = sorted(rocs_auc.nlargest(3).values, reverse=True)
    precisions = df["precision"].apply(lambda x: float(x.split("±")[0].strip()))
    best_three_precision = sorted(precisions.nlargest(3).values, reverse=True)
    list_of_colors = ["green!40", "yellow!40", "orange!40"]

    for _, row in df.iterrows():
        model = row["model"]
        model: str = model.replace("_", "\\_")
        accuracy = row["accuracy"]
        roc_auc = row["roc-auc"]
        precision = row["precision"]
        if float(accuracy.split("±")[0].strip()) in best_three_accuracy:
            # use the index of the value in best_three_accuracy to get the color
            color_index = best_three_accuracy.index(float(accuracy.split("±")[0].strip()))
            accuracy = f"\\cellcolor{{{list_of_colors[color_index]}}} {accuracy}"
        if float(roc_auc.split("±")[0].strip()) in best_three_roc_auc:
            # use the index of the value in best_three_roc_auc to get the color
            color_index = best_three_roc_auc.index(float(roc_auc.split("±")[0].strip()))
            roc_auc = f"\\cellcolor{{{list_of_colors[color_index]}}} {roc_auc}"
        if float(precision.split("±")[0].strip()) in best_three_precision:
            # use the index of the value in best_three_precision to get the color
            color_index = best_three_precision.index(float(precision.split("±")[0].strip()))
            precision = f"\\cellcolor{{{list_of_colors[color_index]}}} {precision}"
        line = f"{model} & {accuracy} & {roc_auc} & {precision} \\\\"
        lines.append(line)

    end_latex = r"""
\bottomrule
\end{tabular}
\end{table}
    """

    output_path = os.path.join(path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, f"{dataset_name}_metrics_mean_std.tex"), "w") as f:
        f.write(setup_latex)
        f.write("\n".join(lines))
        f.write(end_latex)


if __name__ == "__main__":
    task = "hlp"  # TODO: change to "nc" for node classification
    create_results_csv(f"results_{task}", "output")
    build_table_with_std(f"results_{task}/output")
    path = os.path.join(os.path.dirname(__file__), f"results_{task}/output")
    for csvs in os.listdir(path):
        if csvs.endswith("metrics_mean_std.csv"):
            dataset_name = csvs.split("_metrics_mean_std.csv")[0]
            create_latex_table(dataset_name, task, f"results_{task}/output", "latex")
