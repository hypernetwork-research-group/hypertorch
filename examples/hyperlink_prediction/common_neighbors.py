from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hypertorch.hlp import CommonNeighborsHlpModule
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig
from hypertorch.data import (
    AlgebraDataset,
    DataLoader,
    RandomNegativeSampler,
)


if __name__ == "__main__":
    verbose = False
    num_workers = 8
    metrics = MetricCollection(
        {
            "auc": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "avg_precision": BinaryAveragePrecision(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }
    )

    print("Loading and preparing dataset...")

    dataset = AlgebraDataset(sampling_strategy="hyperedge", task="hyperlink-prediction")
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    # Split dataset into train and test portions. CommonNeighbors uses the training
    # hyperedges as its known graph, but does not train or validate parameters.
    train_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.3], shuffle=True, seed=42, node_space_setting="transductive"
    )
    if verbose:
        print(f"Train dataset:\n {train_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

    # Save train hyperedge index before adding negatives
    train_hyperedge_index = train_dataset.hdata.hyperedge_index

    # Add negative samples only to the test split, since the model is evaluated directly.
    negative_sampler = RandomNegativeSampler(
        num_negative_samples=int(test_dataset.hdata.num_hyperedges * 0.6),
        num_nodes_per_sample=int(test_dataset.stats()["avg_degree_hyperedge"]),
    )
    test_dataset = test_dataset.add_negative_samples(negative_sampler, seed=42)

    if verbose:
        print(f"Test dataset after adding negative samples: {test_dataset.hdata}\n")

    print("Creating dataloader...")

    test_loader = DataLoader(
        test_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    mean_cn_module = CommonNeighborsHlpModule(
        train_hyperedge_index=train_hyperedge_index,
        aggregation="mean",
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="common_neighbors",
            version="mean",
            model=mean_cn_module,
            is_trainable=False,
            test_dataloader=test_loader,
        ),
    ]

    print("Starting evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=0,
        accelerator="auto",
        log_every_n_steps=10,
        enable_checkpointing=False,
        auto_start_tensorboard=True,
        auto_wait=True,
        devices=1,
        test_devices=1,
    ) as trainer:
        trainer.test_all(dataloader=test_loader, verbose=True)

    print("Complete!")
