from typing import Literal
from collections.abc import Sequence
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
    DataLoader,
    SamplingStrategy,
    LaplacianPositionalEncodingEnricher,
    NodeEnricher,
    RandomNegativeSampler,
)
from hyperbench.train import MultiModelTrainer
from hyperbench.types import ModelConfig, HData
from torch import Generator


SEED = 42


def __create_seeded_torch_generator(
    device: torch.device,
    seed: int | None,
) -> Generator | None:

    if seed is None:
        return None
    generator = Generator(device=device)
    generator.manual_seed(seed)
    return generator


generator = __create_seeded_torch_generator(device=torch.device("cpu"), seed=SEED)


@cache
def _cached_split_dataset(
    sampling_strategy: SamplingStrategy,
    dataset: Dataset | None = None,
    node_space_setting: Literal["transductive", "inductive"] = "transductive",
) -> tuple[Dataset, Dataset, Dataset]:
    if dataset is None:
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
        dataset = Dataset.from_hdata(hdata, sampling_strategy=sampling_strategy)

    stats = dataset.stats()
    print(f"Dataset stats: {stats['num_nodes']}, {stats['num_hyperedges']}")
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        shuffle=True,
        seed=SEED,
        node_space_setting=node_space_setting,
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


def split_dataset(
    sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
    dataset: Dataset | None = None,
    node_space_setting: Literal["transductive", "inductive"] = "transductive",
) -> tuple[Dataset, Dataset, Dataset]:
    train_dataset, val_dataset, test_dataset = _cached_split_dataset(
        sampling_strategy, dataset, node_space_setting
    )

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
    batch_size: int = 1,
    sample_full_hypergraph: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sample_full_hypergraph=sample_full_hypergraph,
        shuffle=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sample_full_hypergraph=sample_full_hypergraph,
        shuffle=False,
        generator=generator,
    )
    tests_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sample_full_hypergraph=sample_full_hypergraph,
        shuffle=False,
        generator=generator,
    )
    return train_loader, val_loader, tests_loader


def model_configs_with_single_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    tests_loader: DataLoader,
    name: str,
    version: str,
    model: L.LightningModule,
    is_trainable: bool = True,
) -> list[ModelConfig]:
    configs = [
        ModelConfig(
            name=name,
            version=version,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=tests_loader,
            is_trainable=is_trainable,
        )
    ]

    return configs


def model_configs(
    train_loader: DataLoader,
    val_loader: DataLoader,
    tests_loader: DataLoader,
    name: str,
    version: str,
    model: Sequence[L.LightningModule],
) -> list[ModelConfig]:
    configs = []
    for m in model:
        config = ModelConfig(
            name=name,
            version=version,
            model=m,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=tests_loader,
        )
        configs.append(config)
    return configs


def train_test_loop(
    configs: list[ModelConfig],
    max_epochs=3,
    accelerator="auto",
    log_every_n_steps=1,
    enable_checkpointing=False,
    auto_start_tensorboard=False,
    auto_wait=False,
    path: Path | str | None = None,
    experiment_name: str = "integration_test",
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
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
            train_dataloader=train_loader or configs[0].train_dataloader,
            val_dataloader=val_loader or configs[0].val_dataloader,
            verbose=True,
        )
        trainer.test_all(dataloader=test_loader or configs[0].test_dataloader, verbose=True)
