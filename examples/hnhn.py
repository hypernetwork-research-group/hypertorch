from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hyperbench.data import AlgebraDataset, DataLoader, SamplingStrategy
from hyperbench.hlp import HNHNHlpModule
from hyperbench.nn import LaplacianPositionalEncodingEnricher
from hyperbench.train import MultiModelTrainer, RandomNegativeSampler
from hyperbench.types import HData, ModelConfig


if __name__ == "__main__":
    verbose = False
    num_workers = 8
    num_features = 128
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

    dataset = AlgebraDataset(sampling_strategy=sampling_strategy)
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    train_dataset, test_dataset = dataset.split(
        ratios=[0.8, 0.2], shuffle=True, seed=42, node_space_setting="transductive"
    )
    train_dataset, val_dataset = train_dataset.split(
        ratios=[0.875, 0.125], shuffle=True, seed=42, node_space_setting="transductive"
    )
    if verbose:
        print(f"Train dataset (before train/val split):\n {train_dataset.hdata}\n")
        print(f"Train dataset (after train/val split):\n {train_dataset.hdata}\n")
        print(f"Val dataset:\n {val_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

    for name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        num_negative_samples = (
            ds.hdata.num_hyperedges
            if name in ["Train", "Val"]
            else int(ds.hdata.num_hyperedges * 0.6)
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
        enricher=LaplacianPositionalEncodingEnricher(
            num_features=num_features,
            # In transductive setting, use total number of nodes to ensure consistent encoding across splits
            # as the train dataset contain all nodes but may have no hyperedges where they appear
            num_nodes=train_dataset.hdata.num_nodes,
        ),
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

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

    mean_hnhn_module = HNHNHlpModule(
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
            name="hnhn",
            version="mean",
            model=mean_hnhn_module,
            train_dataloader=train_loader_full_hypergraph,
            val_dataloader=val_loader_full_hypergraph,
            test_dataloader=test_loader_full_hypergraph,
        ),
    ]

    print("Starting training and evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=200,
        accelerator="auto",
        log_every_n_steps=1,
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
