from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hyperbench.data import AlgebraDataset, DataLoader, SamplingStrategy
from hyperbench.hlp import Node2VecHlpModule
from hyperbench.nn import Node2VecEnricher
from hyperbench.train import MultiModelTrainer, RandomNegativeSampler
from hyperbench.types import HData, ModelConfig


if __name__ == "__main__":
    verbose = False
    num_workers = 8
    num_features = 32
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
    dataset.remove_hyperedges_with_fewer_than_k_nodes(k=2)
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    # Split dataset into train and test (80/20)
    train_dataset, test_dataset = dataset.split(
        ratios=[0.8, 0.2], shuffle=True, seed=42, node_space_setting="transductive"
    )

    # Split train into train and val (87.5/12.5 of train = 70/10 of total)
    train_dataset, val_dataset = train_dataset.split(
        ratios=[0.875, 0.125], shuffle=True, seed=42, node_space_setting="transductive"
    )
    if verbose:
        print(f"Train dataset:\n {train_dataset.hdata}\n")
        print(f"Val dataset:\n {val_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

    print("Adding negative samples...")

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

    print("Computing Node2Vec embeddings from the train graph...")

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
        batch_size=128,
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

    precomputed_node2vec_module = Node2VecHlpModule(
        encoder_config={
            "mode": "precomputed",
            "num_features": num_features,
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    train_hyperedge_index = train_dataset.hdata.hyperedge_index
    joint_node2vec_module = Node2VecHlpModule(
        encoder_config={
            "mode": "joint",
            "num_features": num_features,
            "context_size": 10,
            "walk_length": 20,
            "num_walks_per_node": 10,
            "p": 1.0,
            "q": 1.0,
            "num_negative_samples": 1,
            "train_hyperedge_index": train_hyperedge_index,
            "num_nodes": dataset.hdata.num_nodes,
            "graph_reduction_strategy": "clique_expansion",
            "random_walk_batch_size": 128,
            "node2vec_loss_weight": 1.0,
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="node2vec",
            version="precomputed",
            model=precomputed_node2vec_module,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
        ModelConfig(
            name="node2vec",
            version="joint",
            model=joint_node2vec_module,
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
