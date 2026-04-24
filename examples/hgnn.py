from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from lightning.pytorch.callbacks import EarlyStopping
from hyperbench.hlp import HGNNHlpModule
from hyperbench.nn import LaplacianPositionalEncodingEnricher
from hyperbench.train import MultiModelTrainer, RandomNegativeSampler
from hyperbench.types import HData, ModelConfig
from hyperbench.data import AlgebraDataset, DataLoader, SamplingStrategy


if __name__ == "__main__":
    verbose = False
    num_workers = 8
    sampling_strategy = SamplingStrategy.HYPEREDGE
    metrics = MetricCollection(
        {
            "auc": BinaryAUROC(),
            "avg_precision": BinaryAveragePrecision(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }
    )

    print("Loading and preparing dataset...")

    dataset = AlgebraDataset(sampling_strategy=sampling_strategy, prepare=True)
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    # Split dataset into train and test (80/20)
    train_dataset, test_dataset = dataset.split(ratios=[0.8, 0.2], shuffle=True, seed=42)

    # Split train into train and val (87.5/12.5 of train = 70/10 of total)
    train_dataset, val_dataset = train_dataset.split(ratios=[0.875, 0.125], shuffle=True, seed=42)
    if verbose:
        print(f"Train dataset (before train/val split):\n {train_dataset.hdata}\n")
        print(f"Train dataset (after train/val split):\n {train_dataset.hdata}\n")
        print(f"Val dataset:\n {val_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

    # Save train hyperedge index before adding negatives (for CommonNeighbors)
    train_hyperedge_index = train_dataset.hdata.hyperedge_index

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
        neg_hdata = negative_sampler.sample(ds.hdata)
        combined_hdata = HData.cat_same_node_space([ds.hdata, neg_hdata])
        shuffled_hdata = combined_hdata.shuffle(seed=42)
        ds_with_negatives = ds.update_from_hdata(shuffled_hdata)

        if name == "Train":
            train_dataset = ds_with_negatives
        elif name == "Val":
            val_dataset = ds_with_negatives
        else:
            test_dataset = ds_with_negatives

        if verbose:
            print(f"{name} dataset after adding negative samples: {shuffled_hdata}\n")

    print("Enriching node features...")

    train_dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(num_features=32),
        enrichment_mode="replace",
    )
    val_dataset.hdata.x = train_dataset.hdata.x[: val_dataset.hdata.num_nodes]
    test_dataset.hdata.x = train_dataset.hdata.x[:, : test_dataset.hdata.num_nodes]

    print("Creating dataloaders...")

    train_loader_full_hypergraph = DataLoader(
        train_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader_full_hypergraph = DataLoader(
        val_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader_full_hypergraph = DataLoader(
        test_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    mean_hgnn_module = HGNNHlpModule(
        encoder_config={
            "in_channels": 32,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
            "fast": False,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="hgnn",
            version="mean",
            model=mean_hgnn_module,
            train_dataloader=train_loader_full_hypergraph,
            val_dataloader=val_loader_full_hypergraph,
            test_dataloader=test_loader_full_hypergraph,
        ),
    ]

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=30,
        mode="min",
    )

    print("Starting training and evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=60,
        accelerator="auto",
        log_every_n_steps=1,
        callbacks=[early_stopping],
        enable_checkpointing=False,
        auto_start_tensorboard=True,
        auto_wait=True,
    ) as trainer:
        trainer.fit_all(
            train_dataloader=train_loader_full_hypergraph,
            val_dataloader=val_loader_full_hypergraph,
            verbose=True,
        )
        trainer.test_all(dataloader=test_loader_full_hypergraph, verbose=True)

    print("Complete!")
