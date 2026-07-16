from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hypertorch.hyperlink_prediction import Node2VecPredictor
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig
from hypertorch.data import (
    AlgebraDataset,
    DataLoader,
    Node2VecEnricher,
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

    print("Adding negative samples...")

    for name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        num_positive_samples = len(ds)
        num_negative_samples = (
            num_positive_samples if name in ["Train", "Val"] else int(num_positive_samples * 0.6)
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

    print("Computing Node2Vec embeddings from the train graph...")

    node2vec_enricher = Node2VecEnricher(
        num_features=num_features,
        context_size=10,
        walk_length=20,
        num_walks_per_node=10,
        num_negative_samples=1,
        # Transductive splits keep the full node space.
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
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
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

    node2vec_precomputed = Node2VecPredictor(
        encoder_config={
            "mode": "precomputed",
            "num_features": num_features,
            "node2vec_config": {},
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    train_hyperedge_index = train_dataset.hdata.hyperedge_index
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
                "num_nodes": dataset.hdata.num_nodes,
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
            name="node2vec-precomputed",
            version="hyperlink-prediction",
            model=node2vec_precomputed,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
        ModelConfig(
            name="node2vec-joint",
            version="hyperlink-prediction",
            model=node2vec_joint,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
    ]

    print("Starting training and evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=60,
        accelerator="auto",
        log_every_n_steps=1,
        enable_checkpointing=False,
        devices=1,
        test_devices=1,
    ) as trainer:
        trainer.fit_all(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            verbose=True,
        )
        trainer.test_all(dataloader=test_loader, verbose=True)

    print("Complete!")
