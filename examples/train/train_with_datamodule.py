from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hypertorch.hlp import MLPHlpModule
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig
from hypertorch.data import (
    AlgebraDataset,
    DataLoader,
    LaplacianPositionalEncodingEnricher,
    RandomNegativeSampler,
)


if __name__ == "__main__":
    verbose = False
    num_workers = 8
    num_features = 32
    metrics = MetricCollection(
        {
            "auc": BinaryAUROC(),
            "avg_precision": BinaryAveragePrecision(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }
    )

    print("Loading and preparing dataset...")

    dataset = AlgebraDataset(sampling_strategy="hyperedge", task="hyperlink-prediction")
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    # Split dataset into train, val and test (70/10/20)
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        shuffle=True,
        seed=42,
        node_space_setting="transductive",
    )
    if verbose:
        print(f"Train dataset:\n {train_dataset.hdata}\n")
        print(f"Val dataset:\n {val_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

    # Add negative samples to all splits
    for name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        num_positive_samples = len(ds)
        num_negative_samples = (
            num_positive_samples
            if name in ["Train", "Val"]  # 1:1 ratio of pos:neg samples
            else int(num_positive_samples * 0.6)  # 60% negatives for test set
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

        if verbose:
            print(f"{name} dataset after adding negative samples: {ds_with_negatives.hdata}\n")

    print("Enriching node features...")

    train_dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(
            num_features=num_features,
            # Transductive splits keep the full node space.
            num_nodes=train_dataset.hdata.num_nodes,
        ),
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

    print("Creating datamodule...")

    data_module = DataLoader.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = MLPHlpModule(
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

    model_configs = [ModelConfig(name="mlp", version="mean", model=model)]

    print("Starting training and evaluation...")

    with MultiModelTrainer(
        model_configs=model_configs,
        max_epochs=60,
        accelerator="auto",
        log_every_n_steps=10,
        enable_checkpointing=False,
        devices=1,
        test_devices=1,
    ) as trainer:
        # No need to explicitly provide `train_dataloader`, `val_dataloader`, and `test_dataloader`,
        # the data module will provision them automatically based on the datasets provided
        # to `DataLoader.from_datasets()`.
        trainer.fit_all(datamodule=data_module, verbose=True)
        trainer.test_all(datamodule=data_module, verbose=True)

    print("Complete!")
