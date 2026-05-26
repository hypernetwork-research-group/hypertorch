import lightning as L
import torch

from functools import cache
from pathlib import Path
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)

from hyperbench.data import (
    Dataset,
    AlgebraDataset,
    DataLoader,
    SamplingStrategy,
    LaplacianPositionalEncodingEnricher,
    NodeEnricher,
    RandomNegativeSampler,
)
from hyperbench.train import MultiModelTrainer
from hyperbench.types import ModelConfig, HData
from torch import Generator


NUM_WORKERS = 2


def create_seeded_torch_generator(
    device: torch.device,
    seed: int | None,
) -> Generator | None:
    """
    Create a seeded torch generator when a seed is provided.

    Args:
        device: Device where the generator should be created.
        seed: Optional seed for deterministic random operations.

    Returns:
        generator: A seeded torch.Generator when ``seed`` is provided, otherwise ``None``.
    """
    if seed is None:
        return None
    generator = Generator(device=device)
    generator.manual_seed(seed)
    return generator


@cache
def _cached_split_datasets(
    sampling_strategy: SamplingStrategy,
) -> tuple[Dataset, Dataset, Dataset]:
    generator = create_seeded_torch_generator(device=torch.device("cpu"), seed=42)
    # dataset = AlgebraDataset(sampling_strategy=sampling_strategy)
    x = torch.randn((100, 4), generator=generator)  # 100 nodes with 4 features each
    hyperedge_index = torch.cat(  # 200 hyperedges, each connecting 5 nodes
        [
            torch.stack(
                [
                    torch.randint(0, 100, (5,), generator=generator),  # 5 nodes per hyperedge
                    torch.full((5,), i),  # hyperedge ID
                ],
                dim=0,
            )
            for i in range(200)
        ],
        dim=1,
    )

    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    dataset = AlgebraDataset(hdata=hdata, sampling_strategy=sampling_strategy)
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2], shuffle=True, seed=42, node_space_setting="transductive"
    )
    return train_dataset, val_dataset, test_dataset


def common_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "auc": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "avg_precision": BinaryAveragePrecision(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }
    )


def splits_dataset(sampling_strategy: SamplingStrategy) -> tuple[Dataset, Dataset, Dataset]:
    train_dataset, val_dataset, test_dataset = _cached_split_datasets(sampling_strategy)

    return (
        train_dataset.update_from_hdata(train_dataset.hdata.clone()),
        val_dataset.update_from_hdata(val_dataset.hdata.clone()),
        test_dataset.update_from_hdata(test_dataset.hdata.clone()),
    )


def add_negatives(
    train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset
) -> tuple[Dataset, Dataset, Dataset]:
    # Add negative samples to all splits
    for name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        num_negative_samples = (
            ds.hdata.num_hyperedges
            if name in ["Train", "Val"]  # 1:1 ratio of pos:neg samples
            else int(ds.hdata.num_hyperedges * 0.6)  # 60% negatives for test set
        )
        negative_sampler = RandomNegativeSampler(
            num_negative_samples=num_negative_samples,
            num_nodes_per_sample=int(ds.stats()["avg_degree_hyperedge"]),
        )
        ds_with_negatives = ds.add_negative_samples(negative_sampler, seed=42)

        if name == "Train":
            train_dataset = ds_with_negatives
        elif name == "Val":
            val_dataset = ds_with_negatives
        else:
            test_dataset = ds_with_negatives

    return train_dataset, val_dataset, test_dataset


def enrich_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    num_features: int = 32,
    enricher: NodeEnricher | None = None,
) -> None:
    if enricher is None:
        train_dataset.enrich_node_features(
            enricher=LaplacianPositionalEncodingEnricher(
                num_features=num_features,
                # In transductive setting, use total number of nodes to ensure consistent encoding across splits
                # as the train dataset contain all nodes but may have no hyperedges where they appear
                num_nodes=train_dataset.hdata.num_nodes,
            ),
            enrichment_mode="replace",
        )
    else:
        train_dataset.enrich_node_features(
            enricher=enricher,
            enrichment_mode="replace",
        )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)


def loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch=False,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if not batch:
        train_loader = DataLoader(
            train_dataset,
            sample_full_hypergraph=True,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            sample_full_hypergraph=True,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )
        tests_loader = DataLoader(
            test_dataset,
            sample_full_hypergraph=True,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )
        tests_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

    return train_loader, val_loader, tests_loader


def model_configs(
    train_loader: DataLoader,
    val_loader: DataLoader,
    tests_loader: DataLoader,
    name: str,
    version: str,
    model: L.LightningModule,
) -> list[ModelConfig]:
    configs = [
        ModelConfig(
            name=name,
            version=version,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=tests_loader,
        )
    ]

    return configs


def add_model_configs(
    configs: list[ModelConfig],
    train_loader: DataLoader,
    val_loader: DataLoader,
    tests_loader: DataLoader,
    name: str,
    version: str,
    model: L.LightningModule,
) -> list[ModelConfig]:
    new_config = ModelConfig(
        name=name,
        version=version,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=tests_loader,
    )
    configs.append(new_config)
    return configs


def multi_model_trainer(
    configs: list[ModelConfig],
    max_epochs=3,
    accelerator="auto",
    log_every_n_steps=1,
    enable_checkpointing=False,
    auto_start_tensorboard=False,
    auto_wait=False,
    path: Path | str | None = None,
    experiment_name: str = "integration_test",
):
    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=max_epochs,
        accelerator=accelerator,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
        auto_start_tensorboard=auto_start_tensorboard,
        auto_wait=auto_wait,
        default_root_dir=path,
        experiment_name=experiment_name,
    ) as trainer:
        trainer.fit_all(
            train_dataloader=configs[0].train_dataloader,
            val_dataloader=configs[0].val_dataloader,
            verbose=True,
        )
        trainer.test_all(dataloader=configs[0].test_dataloader, verbose=True)
