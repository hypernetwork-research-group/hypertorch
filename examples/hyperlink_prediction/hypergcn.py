from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hypertorch.hyperlink_prediction import HyperGCNPredictor
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
            "accuracy": BinaryAccuracy(),
            "avg_precision": BinaryAveragePrecision(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }
    )

    print("Loading and preparing dataset...")

    dataset = AlgebraDataset(sampling_strategy="hyperedge", task="hyperlink-prediction")
    dataset.remove_hyperedges_with_fewer_than_k_nodes(k=2)
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    # Split dataset into train, val and test (70/10/20)
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        node_space_setting="transductive",
        shuffle=True,
        seed=42,
    )
    if verbose:
        print(f"Train dataset:\n {train_dataset.hdata}\n")
        print(f"Val dataset:\n {val_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

    # Save train hyperedge index before adding negatives (for CommonNeighbors)
    train_hyperedge_index = train_dataset.hdata.hyperedge_index

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
            num_nodes=dataset.hdata.num_nodes,
        ),
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

    print("Creating dataloaders...")

    train_loader = DataLoader(
        train_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    hypergcn_no_mediator = HyperGCNPredictor(
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

    hypergcn_with_mediator = HyperGCNPredictor(
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
            name="hypergcn-no-mediator",
            version="hyperlink-prediction",
            model=hypergcn_no_mediator,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
        ModelConfig(
            name="hypergcn-with-mediator",
            version="hyperlink-prediction",
            model=hypergcn_with_mediator,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
    ]

    print("Starting training and evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=10,
        accelerator="auto",
        log_every_n_steps=1,
        enable_checkpointing=False,
        devices=1,
        test_devices=1,
    ) as trainer:
        trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader, verbose=True)
        trainer.test_all(dataloader=test_loader, verbose=True)

    print("Complete!")
