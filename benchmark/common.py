import os
import pandas as pd

from torchmetrics import MetricCollection
from hypertorch.hyperlink_prediction import GCNPredictor, CommonNeighborsPredictor
from hypertorch.types import ModelConfig
from hypertorch.data import DataLoader, Dataset, RandomNegativeSampler, get_dataset_by_name


def load_common_neighbors(
    metrics: MetricCollection,
    num_nodes: int,
    train_dataset: Dataset,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
) -> list[ModelConfig]:
    model = CommonNeighborsPredictor(
        train_hdata=train_dataset.hdata,
        aggregation="mean",
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"common-neighbors_{num_run}",
            version="hyperlink-prediction",
            model=model,
            is_trainable=False,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 5,
            },
        ),
    ]

    return configs


def load_gcn(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
) -> list[ModelConfig]:
    model = GCNPredictor(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "drop_rate": 0.1,
            "bias": True,
            "improved": False,
            "add_self_loops": True,
            "normalize": True,
            "cached": False,
            "graph_reduction_strategy": "clique_expansion",
            "num_nodes": num_nodes,
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"gcn_{num_run}",
            version="hyperlink-prediction",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 5,
            },
        )
    ]

    return configs


def prepare(
    dataset_name: str,
    split_ratios: list[float],
    k_nodes: int = 2,
    seed: int = 42,
    test_set_negative_ratio: float = 0.6,
) -> tuple[Dataset, Dataset, Dataset, int, int]:

    dataset = get_dataset_by_name(dataset_name)
    dataset.remove_hyperedges_with_fewer_than_k_nodes(k=k_nodes)

    num_features = dataset.hdata.x.shape[1] if dataset.hdata.x is not None else 32

    # Split dataset into train, val and test (70/10/20)
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=split_ratios,
        node_space_setting="transductive",
        shuffle=True,
        seed=seed,
    )

    # Add negative samples to all splits
    for name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        num_positive_samples = len(ds)
        num_negative_samples = (
            num_positive_samples
            if name in ["Train", "Val"]  # 1:1 ratio of pos:neg samples
            else int(num_positive_samples * test_set_negative_ratio)  # 60% negatives for test set
        )
        negative_sampler = RandomNegativeSampler(
            num_negative_samples=num_negative_samples,
            num_nodes_per_sample=int(ds.stats()["avg_degree_hyperedge"]),
        )
        ds_with_negatives = ds.add_negative_samples(negative_sampler, seed=seed)

        if name == "Train":
            train_dataset = ds_with_negatives
        elif name == "Val":
            val_dataset = ds_with_negatives
        else:
            test_dataset = ds_with_negatives

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        dataset.hdata.num_nodes,
        num_features,
    )


def merge_all_results(dir_path: str, output_file: str = "merged_results.csv"):

    all_results = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith("metrics.csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                dataset_name = root.split(os.sep)[2]
                model = root.split(os.sep)[3]
                model_name = model.split("_")[0]
                model_run = model.split("_")[1]
                df.insert(0, "dataset", dataset_name)
                df.insert(1, "model_name", model_name)
                df.insert(2, "model_run", model_run)
                all_results.append(df)

    if all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        merged_df.to_csv(os.path.join(dir_path, output_file), index=False)
