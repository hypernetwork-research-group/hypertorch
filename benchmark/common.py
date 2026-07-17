import os
import pandas as pd
from torch import Tensor

from torchmetrics import MetricCollection
from hypertorch.hyperlink_prediction import (
    GCNPredictor,
    CommonNeighborsPredictor,
    HGNNPredictor,
    HGNNPPredictor,
    HNHNPredictor,
    HyperGCNPredictor,
    MLPPredictor,
    NHPPredictor,
    Node2VecPredictor,
    VilLainPredictor,
)
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
                "max_epochs": 3,
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
                "max_epochs": 3,
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
) -> list[ModelConfig]:
    model = HGNNPredictor(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hgnn_{num_run}",
            version="hyperlink-prediction",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 3,
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
) -> list[ModelConfig]:
    model = HGNNPPredictor(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hgnnp_{num_run}",
            version="hyperlink-prediction",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 3,
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
) -> list[ModelConfig]:
    model = HNHNPredictor(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 400,
            "out_channels": 400,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.3,
        },
        aggregation="mean",
        lr=0.04,
        weight_decay=5e-4,
        scheduler_step_size=100,
        scheduler_gamma=0.51,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hnhn_{num_run}",
            version="hyperlink-prediction",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 3,
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
) -> list[ModelConfig]:
    model = HyperGCNPredictor(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
            "use_mediator": False,
            "fast": False,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hypergcn_no_med_{num_run}",
            version="hyperlink-prediction",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 3,
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
) -> list[ModelConfig]:
    model = HyperGCNPredictor(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
            "use_mediator": True,
            "fast": False,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"hypergcn_no_med_{num_run}",
            version="hyperlink-prediction",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 3,
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
) -> list[ModelConfig]:
    model = MLPPredictor(
        encoder_config={
            "in_channels": num_features,
            "out_channels": num_features,
            "hidden_channels": 64,
            "num_layers": 3,
            "drop_rate": 0.3,
        },
        aggregation="mean",
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"mlp_{num_run}",
            version="hyperlink-prediction",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 3,
            },
        ),
    ]

    return configs


def load_nhp(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
) -> list[ModelConfig]:
    model = NHPPredictor(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 512,
            "aggregation": "maxmin",
        },
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"nhp_{num_run}",
            version="hyperlink-prediction",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 3,
            },
        ),
    ]

    return configs


def load_villain_node(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
) -> list[ModelConfig]:
    villain_node = VilLainPredictor(
        encoder_config={
            "embedding_dim": 128,
            "labels_per_subspace": 8,
            "training_steps": 4,
            "generation_steps": 128,
            "tau": 1.0,
            "eps": 1e-10,
            "villain_loss_weight": 1.0,
            # Transductive splits keep the full node space.
            "num_nodes": num_nodes,
        },
        embedding_mode="node",
        aggregation="maxmin",
        lr=0.01,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"villain_node_{num_run}",
            version="hyperlink-prediction",
            model=villain_node,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 3,
            },
        ),
    ]

    return configs


def load_villain_hyperedge(
    metrics: MetricCollection,
    num_nodes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_run: int,
    num_features: int = 32,
) -> list[ModelConfig]:
    villain_hyperedge = VilLainPredictor(
        encoder_config={
            "embedding_dim": 128,
            "labels_per_subspace": 8,
            "training_steps": 4,
            "generation_steps": 28,
            "tau": 1.0,
            "eps": 1e-10,
            "villain_loss_weight": 0.4,
            "num_nodes": num_nodes,
        },
        embedding_mode="hyperedge",
        lr=0.01,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"villain_hyperedge_{num_run}",
            version="hyperlink-prediction",
            model=villain_hyperedge,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 100,
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
) -> list[ModelConfig]:
    node2vec_joint = Node2VecPredictor(
        encoder_config={
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
                # Transductive splits keep the full node space.
                "num_nodes": num_nodes,
                "graph_reduction_strategy": "clique_expansion",
                "random_walk_batch_size": 128,
                # We count the node2vec loss as 40% of the total loss (the rest is the HLP loss)
                "node2vec_loss_weight": 0.4,
            },
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name=f"node2vec_joint_{num_run}",
            version="hyperlink-prediction",
            model=node2vec_joint,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            trainer_kwargs={
                "max_epochs": 100,
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
