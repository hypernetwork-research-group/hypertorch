from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hyperbench.data import CoraDataset, DataLoader, SamplingStrategy
from hyperbench.hlp import VilLainHlpModule
from hyperbench.train import MultiModelTrainer, RandomNegativeSampler
from hyperbench.types import ModelConfig


if __name__ == "__main__":
    verbose = False
    num_workers = 8
    sampling_strategy = SamplingStrategy.HYPEREDGE
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

    dataset = CoraDataset(sampling_strategy=sampling_strategy)
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    # Split dataset into train, val and test (70/10/20)
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        shuffle=True,
        seed=42,
        node_space_setting="transductive",
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
        ds_with_negatives = ds.add_negative_samples(negative_sampler, seed=42)

        if name == "Train":
            train_dataset = ds_with_negatives
        elif name == "Val":
            val_dataset = ds_with_negatives
        else:
            test_dataset = ds_with_negatives

        if verbose:
            print(f"{name} dataset after adding negative samples: {ds_with_negatives.hdata}\n")

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

    node_villain_module = VilLainHlpModule(
        encoder_config={
            "num_nodes": train_dataset.hdata.num_nodes,
            "embedding_dim": 128,
            "labels_per_subspace": 8,
            "training_steps": 4,
            "generation_steps": 128,
            "tau": 1.0,
            "eps": 1e-10,
            "villain_loss_weight": 1.0,
        },
        embedding_mode="node",
        aggregation="maxmin",
        lr=0.01,
        weight_decay=0.0,
        metrics=metrics,
    )

    hyperedge_villain_module = VilLainHlpModule(
        encoder_config={
            "num_nodes": train_dataset.hdata.num_nodes,
            "embedding_dim": 128,
            "labels_per_subspace": 8,
            "training_steps": 4,
            "generation_steps": 28,
            "tau": 1.0,
            "eps": 1e-10,
            "villain_loss_weight": 1.0,
        },
        embedding_mode="hyperedge",
        lr=0.01,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="villain",
            version="node_maxmin",
            model=node_villain_module,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
        ModelConfig(
            name="villain",
            version="hyperedge",
            model=hyperedge_villain_module,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
    ]

    print("Starting training and evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=100,
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
