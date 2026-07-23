import argparse
import os
import pandas as pd
import psutil
import torch
from torch import Tensor

from torchmetrics import MetricCollection
from hypertorch.node_classification import (
    GCNClassifier,
    CommonNeighborsClassifier,
    HGNNClassifier,
    HGNNPClassifier,
    HNHNClassifier,
    HyperGCNClassifier,
    MLPClassifier,
    VilLainClassifier,
    Node2VecGCNClassifier,
    Node2VecGCNNCConfig,
)
from hypertorch.types import ModelConfig, TaskEnum
from hypertorch.data import DataLoader, Dataset, get_dataset_by_name


def collect_hw_stats_row(
    run: int,
    dataset: str,
    model: str,
    before_stats: tuple[float, float, float],
    after_stats: tuple[float, float, float],
    execution_time: float,
) -> dict[str, float | int | str]:
    cpu_usage_before, ram_usage_before, gpu_usage_before = before_stats
    cpu_usage_after, ram_usage_after, gpu_usage_after = after_stats

    return {
        "run": run,
        "dataset": dataset,
        "model": model,
        "execution_time": execution_time,
        "cpu_usage_before": cpu_usage_before,
        "ram_usage_before": ram_usage_before,
        "gpu_usage_before": gpu_usage_before,
        "cpu_usage_after": cpu_usage_after,
        "ram_usage_after": ram_usage_after,
        "gpu_usage_after": gpu_usage_after,
        "cpu_usage_diff": cpu_usage_after - cpu_usage_before,
        "ram_usage_diff": ram_usage_after - ram_usage_before,
        "gpu_usage_diff": gpu_usage_after - gpu_usage_before,
    }


def retrieve_hw_stats() -> tuple[float, float, float]:
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return cpu_usage, ram_usage, gpu_usage


def load_common_neighbors(
    metrics: MetricCollection,
    num_nodes: int,
    train_dataset: Dataset,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    model = CommonNeighborsClassifier(
        train_hdata=train_dataset.hdata,
        num_classes=num_classes,
        aggregation="mean",
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"common-neighbors_{num_run}",
            version="node-classification",
            model=model,
            is_trainable=False,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
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
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    model = GCNClassifier(
        classifier_config={
            "in_channels": num_features,
            "out_channels": num_classes,
            "hidden_channels": 16,
            "num_layers": 2,
            "drop_rate": 0.3,
            "graph_reduction_strategy": "clique_expansion",
            "num_nodes": num_nodes,
        },
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"gcn_{num_run}",
            version="node-classification",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
            },
        )
    ]

    return configs


def load_hgnn(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    model = HGNNClassifier(
        classifier_config={
            "in_channels": num_features,
            "out_channels": num_classes,
            "hidden_channels": 64,
            "drop_rate": 0.3,
        },
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hgnn_{num_run}",
            version="node-classification",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
            },
        ),
    ]

    return configs


def load_hgnnp(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    model = HGNNPClassifier(
        classifier_config={
            "in_channels": num_features,
            "out_channels": num_classes,
            "hidden_channels": 64,
            "drop_rate": 0.3,
        },
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hgnnp_{num_run}",
            version="node-classification",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
            },
        ),
    ]

    return configs


def load_hnhn(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    model = HNHNClassifier(
        classifier_config={
            "in_channels": num_features,
            "out_channels": num_classes,
            "hidden_channels": 64,
            "drop_rate": 0.3,
        },
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hnhn_{num_run}",
            version="node-classification",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
            },
        ),
    ]

    return configs


def load_hypergcn_no_mediator(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    model = HyperGCNClassifier(
        classifier_config={
            "in_channels": num_features,
            "out_channels": num_classes,
            "hidden_channels": 64,
            "drop_rate": 0.3,
            "use_mediator": False,
            "fast": False,
            "seed": 42,
        },
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hypergcn_no_med_{num_run}",
            version="node-classification",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
            },
        ),
    ]

    return configs


def load_hypergcn_with_mediator(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    model = HyperGCNClassifier(
        classifier_config={
            "in_channels": num_features,
            "out_channels": num_classes,
            "hidden_channels": 64,
            "drop_rate": 0.3,
            "use_mediator": True,
            "fast": False,
            "seed": 42,
        },
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hypergcn_no_med_{num_run}",
            version="node-classification",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
            },
        ),
    ]

    return configs


def load_mlp(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    model = MLPClassifier(
        classifier_config={
            "in_channels": num_features,
            "out_channels": num_classes,
            "hidden_channels": 64,
            "num_layers": 3,
            "drop_rate": 0.3,
        },
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"mlp_{num_run}",
            version="node-classification",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
            },
        ),
    ]

    return configs

    # def load_nhp(
    #     metrics: MetricCollection,
    #     num_nodes: int,
    #     train_loader: DataLoader,
    #     val_loader: DataLoader,
    #     test_loader: DataLoader,
    #     num_run: int,
    #     num_features: int = 32,
    #     max_epochs: int = 100,
    # ) -> list[ModelConfig]:
    #     model = NHPClassifier(
    #         encoder_config={
    #             "in_channels": num_features,
    #             "hidden_channels": 512,
    #             "aggregation": "maxmin",
    #         },
    #         lr=0.001,
    #         weight_decay=5e-4,
    #         metrics=metrics,
    #     )

    #     configs = [
    #         ModelConfig(
    #             name=f"nhp_{num_run}",
    #             version="node-classification",
    #             model=model,
    #             train_dataloader=train_loader,
    #             val_dataloader=val_loader,
    #             test_dataloader=test_loader,
    #             trainer_kwargs={
    #                 "max_epochs": max_epochs,
    #             },
    #         ),
    #     ]

    #     return configs


def load_villain_node(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    villain_node = VilLainClassifier(
        encoder_config={
            "embedding_dim": 128,
            "labels_per_subspace": 8,
            "training_steps": 4,
            "generation_steps": 28,
            "tau": 1.0,
            "eps": 1e-10,
            # 40% weight on the VilLain loss, 60% weight on the classifier loss
            "villain_loss_weight": 0.4,
            # Transductive splits keep global node IDs available for the full node space.
            "num_nodes": num_nodes,
        },
        classifier_config={
            "out_channels": num_classes,
        },
        lr=0.01,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"villain_node_{num_run}",
            version="node-classification",
            model=villain_node,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
            },
        ),
    ]

    return configs


def load_n2v_joint(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_hyperedge_index: Tensor,
    num_run: int,
    num_features: int = 32,
    max_epochs: int = 100,
    num_classes: int = 2,
) -> list[ModelConfig]:
    gcn_config: Node2VecGCNNCConfig = {
        "out_channels": num_classes,
        "hidden_channels": num_features,
        "num_layers": 2,
        "drop_rate": 0.1,
        "bias": True,
        "improved": False,
        "add_self_loops": True,
        "normalize": True,
        "cached": False,
        "graph_reduction_strategy": "clique_expansion",
        "num_nodes": num_nodes,
    }
    node2vec_joint = Node2VecGCNClassifier(
        classifier_config={
            "mode": "joint",
            "num_features": num_features,
            "node2vec_config": {
                "context_size": 10,
                "walk_length": 20,
                "num_walks_per_node": 10,
                "p": 1.0,
                "q": 1.0,
                "num_negative_samples": 1,
                "train_hyperedge_index": train_hyperedge_index,
                "num_nodes": num_nodes,
                "graph_reduction_strategy": "clique_expansion",
                "random_walk_batch_size": 128,
                "node2vec_loss_weight": 0.4,
            },
            "gcn_config": gcn_config,
        },
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"node2vec_joint_{num_run}",
            version="node-classification",
            model=node2vec_joint,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": max_epochs,
            },
        ),
    ]

    return configs


def prepare(
    dataset_name: str,
    split_ratios: list[float],
    k_nodes: int = 2,
    seed: int = 42,
    test_set_negative_ratio: float = 0.6,
    task: TaskEnum = TaskEnum.HYPERLINK_PREDICTION,
) -> tuple[Dataset, Dataset, Dataset, int, int]:

    dataset = get_dataset_by_name(dataset_name=dataset_name, sampling_strategy="node", task=task)
    dataset.remove_hyperedges_with_fewer_than_k_nodes(k=k_nodes)
    # dataset.hdata.y = node_labels_from_node_degrees(
    #     node_incidences=dataset.hdata.hyperedge_index[0],
    #     num_nodes=dataset.hdata.num_nodes,
    # )
    num_features = dataset.hdata.x.shape[1] if dataset.hdata.x is not None else 32
    # Split dataset into train, val and test (70/10/20)
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=split_ratios,
        node_space_setting="transductive",
        shuffle=True,
        seed=seed,
    )

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
                model_name = "_".join(model.split("_")[:-1])
                model_run = model.split("_")[-1]
                df.insert(0, "dataset", dataset_name)
                df.insert(1, "model_name", model_name)
                df.insert(2, "model_run", model_run)
                all_results.append(df)

    if all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        merged_df.to_csv(os.path.join(dir_path, output_file), index=False)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-features", type=int, default=32)
    parser.add_argument("--seed", type=int, nargs=3, default=[42, 43, 44])
    parser.add_argument("--k-nodes", type=int, default=2)
    parser.add_argument("--test-set-negative-ratio", type=float, default=0.6)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--task", type=TaskEnum, default=TaskEnum.HYPERLINK_PREDICTION)
    return parser.parse_args()
