from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hyperbench.data import AlgebraDataset, DataLoader, SamplingStrategy
from hyperbench.hlp import NHPHlpModule
from hyperbench.nn import Node2VecEnricher
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

    dataset = AlgebraDataset(sampling_strategy=sampling_strategy, prepare=True)
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    train_dataset, test_dataset = dataset.split(
        ratios=[0.8, 0.2], shuffle=True, seed=42, node_space_setting="transductive"
    )
    train_dataset, val_dataset = train_dataset.split(
        ratios=[0.875, 0.125], shuffle=True, seed=42, node_space_setting="transductive"
    )

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
        shuffled_hdata = HData.cat_same_node_space([ds.hdata, neg_hdata]).shuffle(seed=42)
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

    node2vec_enricher = Node2VecEnricher(
        num_features=num_features,
        context_size=10,
        walk_length=20,
        num_walks_per_node=10,
        num_negative_samples=1,
        num_nodes=dataset.hdata.num_nodes,
        num_epochs=10,
        learning_rate=0.01,
        batch_size=128,
        sparse=False,
        verbose=verbose,
    )

    train_dataset.enrich_node_features(
        enricher=node2vec_enricher,
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

    print("Creating dataloaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
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

    maxmin_nhp_module = NHPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 512,
            "aggregation": "maxmin",
        },
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    mean_nhp_module = NHPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 512,
            "aggregation": "mean",
        },
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="nhp",
            version="maxmin",
            model=maxmin_nhp_module,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
        ModelConfig(
            name="nhp",
            version="mean",
            model=mean_nhp_module,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
    ]

    print("Starting training and evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=50,
        accelerator="auto",
        log_every_n_steps=1,
        enable_checkpointing=False,
        auto_start_tensorboard=True,
        auto_wait=True,
    ) as trainer:
        trainer.fit_all(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            verbose=True,
        )
        trainer.test_all(dataloader=test_loader, verbose=True)

    print("Complete!")
