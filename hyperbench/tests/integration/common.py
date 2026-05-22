from hyperbench.types import ModelConfig
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hyperbench.hlp import (
    GCNHlpModule,
    HGNNHlpModule,
    HGNNPHlpModule,
)
from hyperbench.nn import LaplacianPositionalEncodingEnricher
from hyperbench.train import MultiModelTrainer, RandomNegativeSampler
from hyperbench.data import AlgebraDataset, DataLoader, SamplingStrategy

NUM_WORKERS = 2
NUM_FEATURES = 32
SAMPLING_STRATEGY = SamplingStrategy.HYPEREDGE


def common_standard_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "auc": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "avg_precision": BinaryAveragePrecision(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }
    )


def splits_dataset():
    dataset = AlgebraDataset(sampling_strategy=SAMPLING_STRATEGY)

    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2], shuffle=True, seed=42, node_space_setting="transductive"
    )

    return train_dataset, val_dataset, test_dataset


def add_negatives(train_dataset, val_dataset, test_dataset):
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


def datasets_enrichers(train_dataset, val_dataset, test_dataset):
    train_dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(
            num_features=NUM_FEATURES,
            # In transductive setting, use total number of nodes to ensure consistent encoding across splits
            # as the train dataset contain all nodes but may have no hyperedges where they appear
            num_nodes=train_dataset.hdata.num_nodes,
        ),
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)


def loaders(train_dataset, val_dataset, test_dataset):
    train_loader_full_hypergraph = DataLoader(
        train_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    val_loader_full_hypergraph = DataLoader(
        val_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    test_loader_full_hypergraph = DataLoader(
        test_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )

    return train_loader_full_hypergraph, val_loader_full_hypergraph, test_loader_full_hypergraph


def model_configs(
    train_loader_full_hypergraph,
    val_loader_full_hypergraph,
    test_loader_full_hypergraph,
    name,
    version,
    module,
) -> list[ModelConfig]:
    configs = [
        ModelConfig(
            name=name,
            version=version,
            model=module,
            train_dataloader=train_loader_full_hypergraph,
            val_dataloader=val_loader_full_hypergraph,
            test_dataloader=test_loader_full_hypergraph,
        )
    ]

    return configs


def multi_model_trainer(
    configs,
    max_epochs=5,
    accelerator="auto",
    log_every_n_steps=1,
    enable_checkpointing=False,
    auto_start_tensorboard=False,
    auto_wait=True,
):
    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=max_epochs,
        accelerator=accelerator,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
        auto_start_tensorboard=auto_start_tensorboard,
        auto_wait=auto_wait,
    ) as trainer:
        trainer.fit_all(
            train_dataloader=configs[0].train_dataloader,
            val_dataloader=configs[0].val_dataloader,
            verbose=True,
        )
        trainer.test_all(dataloader=configs[0].test_dataloader, verbose=True)


def gcn_model():
    num_features = NUM_FEATURES
    metrics = common_standard_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset()
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    datasets_enrichers(train_dataset, val_dataset, test_dataset)

    train_loader_full_hypergraph, val_loader_full_hypergraph, test_loader_full_hypergraph = loaders(
        train_dataset, val_dataset, test_dataset
    )

    mean_gcn_module = GCNHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "drop_rate": 0.1,
            "bias": True,
            "improved": False,
            "add_self_loops": True,
            "normalize": True,
            "cached": False,
            "graph_reduction_strategy": "clique_expansion",
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader_full_hypergraph,
        val_loader_full_hypergraph,
        test_loader_full_hypergraph,
        name="gcn",
        version="mean",
        module=mean_gcn_module,
    )

    multi_model_trainer(configs)


def hgnn_model():
    num_features = NUM_FEATURES
    metrics = common_standard_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset()
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    datasets_enrichers(train_dataset, val_dataset, test_dataset)

    train_loader_full_hypergraph, val_loader_full_hypergraph, test_loader_full_hypergraph = loaders(
        train_dataset, val_dataset, test_dataset
    )

    mean_hgnn_module = HGNNHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader_full_hypergraph,
        val_loader_full_hypergraph,
        test_loader_full_hypergraph,
        name="hgnn",
        version="mean",
        module=mean_hgnn_module,
    )

    multi_model_trainer(configs)


def hgnnp_model():

    num_features = NUM_FEATURES

    metrics = common_standard_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset()

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    datasets_enrichers(train_dataset, val_dataset, test_dataset)

    train_loader_full_hypergraph, val_loader_full_hypergraph, test_loader_full_hypergraph = loaders(
        train_dataset, val_dataset, test_dataset
    )

    mean_hgnnp_module = HGNNPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader_full_hypergraph,
        val_loader_full_hypergraph,
        test_loader_full_hypergraph,
        name="hgnnp",
        version="mean",
        module=mean_hgnnp_module,
    )

    multi_model_trainer(configs)
