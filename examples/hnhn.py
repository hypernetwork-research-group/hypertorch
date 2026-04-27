import torch

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from lightning.pytorch.callbacks import EarlyStopping

from hyperbench.data import AlgebraDataset, DataLoader, SamplingStrategy
from hyperbench.hlp import HNHNHlpModule
from hyperbench.nn import LaplacianPositionalEncodingEnricher
from hyperbench.train import MultiModelTrainer, RandomNegativeSampler
from hyperbench.types import HData, ModelConfig


def assign_train_node_features(train_dataset, target_dataset) -> tuple[int, int]:
    train_hdata = train_dataset.hdata
    target_hdata = target_dataset.hdata

    if train_hdata.global_node_ids is None or target_hdata.global_node_ids is None:
        raise ValueError("Expected global_node_ids to align train features across splits.")

    num_features = train_hdata.x.size(1)
    target_x = torch.zeros(
        (target_hdata.num_nodes, num_features),
        dtype=train_hdata.x.dtype,
        device=train_hdata.x.device,
    )

    train_rows_by_global_id = {
        int(global_node_id): row_idx
        for row_idx, global_node_id in enumerate(train_hdata.global_node_ids.tolist())
    }

    num_matched = 0
    for target_row_idx, global_node_id in enumerate(target_hdata.global_node_ids.tolist()):
        train_row_idx = train_rows_by_global_id.get(int(global_node_id))
        if train_row_idx is None:
            target_x[target_row_idx] = torch.zeros(
                num_features, dtype=train_hdata.x.dtype, device=train_hdata.x.device
            )
            continue
        target_x[target_row_idx] = train_hdata.x[train_row_idx]
        num_matched += 1

    target_hdata.x = target_x
    return num_matched, target_hdata.num_nodes - num_matched


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

    train_dataset, test_dataset = dataset.split(ratios=[0.8, 0.2], shuffle=True, seed=42)
    train_dataset, val_dataset = train_dataset.split(ratios=[0.875, 0.125], shuffle=True, seed=42)
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
        enricher=LaplacianPositionalEncodingEnricher(num_features=64),
        enrichment_mode="replace",
    )
    val_matched, val_missing = assign_train_node_features(train_dataset, val_dataset)
    test_matched, test_missing = assign_train_node_features(train_dataset, test_dataset)

    if verbose:
        print(
            f"Val train-feature matches: {val_matched}/{val_dataset.hdata.num_nodes} "
            f"(missing: {val_missing})"
        )
        print(
            f"Test train-feature matches: {test_matched}/{test_dataset.hdata.num_nodes} "
            f"(missing: {test_missing})"
        )

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
            "in_channels": 64,
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

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=100,
        mode="min",
    )

    print("Starting training and evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=200,
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
