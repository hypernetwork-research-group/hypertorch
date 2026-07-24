import os
import pandas as pd

hlp_list_models = [
    "gcn",
    "common_neighbors",
    "hgnn",
    "hgnnp",
    "hnhn",
    "hypergcn_no_mediator",
    "hypergcn_with_mediator",
    "mlp",
    "nhp",
    "villain_node",
    "villain_hyperedge",
    "node2vec",
]


def clean_model_name(model_name: str) -> str:
    model_name = model_name.replace("\\_", "_")
    if "_" in model_name:
        model_name = "_".join(model_name.split("_")[:-1])
    return model_name


def create_results_csv(folder: str):
    path = os.path.join(os.path.dirname(__file__), folder)

    for dataset_name in os.listdir(path):
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
        dataset_df.to_csv(os.path.join(path, f"{dataset_name}_metrics.csv"))


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


def create_latex_table(dataset_name: str, folder: str):
    path = os.path.join(os.path.dirname(__file__), folder)
    setup_latex = (
        r"""
\begin{table}[htbp]
\label{tab:hlp_results_"""
        + dataset_name
        + r"""}
\centering
\caption{Results for HLP - dataset """
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
    for _, row in df.iterrows():
        model = row["model"]
        accuracy = row["accuracy"]
        roc_auc = row["roc-auc"]
        precision = row["precision"]
        line = f"{model} & {accuracy} & {roc_auc} & {precision} \\\\"
        lines.append(line)

    end_latex = r"""
\bottomrule
\end{tabular}
\end{table}
    """

    with open(os.path.join(path, f"{dataset_name}_metrics_mean_std.tex"), "w") as f:
        f.write(setup_latex)
        f.write("\n".join(lines))
        f.write(end_latex)


if __name__ == "__main__":
    # create_results_csv("results_hlp")
    # build_table_with_std("results_hlp")
    path = os.path.join(os.path.dirname(__file__), "results_hlp")
    for csvs in os.listdir(path):
        if csvs.endswith("metrics_mean_std.csv"):
            dataset_name = csvs.split("_metrics_mean_std.csv")[0]
            create_latex_table(dataset_name, "results_hlp")
