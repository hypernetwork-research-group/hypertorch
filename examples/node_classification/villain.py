from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score
from hypertorch.data import CoraDataset, DataLoader
from hypertorch.nc import VilLainClassifier
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig
from hypertorch.utils import node_labels_from_node_degrees


if __name__ == "__main__":
    verbose = False
    num_workers = 8
    num_classes = 3
    metrics = MetricCollection(
        {
            "auc": MulticlassAUROC(num_classes=num_classes),
            "accuracy": MulticlassAccuracy(num_classes=num_classes),
            "f1": MulticlassF1Score(num_classes=num_classes),
        }
    )

    print("Loading and preparing dataset...")

    dataset = CoraDataset(sampling_strategy="node", task="node-classification")
    dataset.hdata.y = node_labels_from_node_degrees(
        node_incidences=dataset.hdata.hyperedge_index[0],
        num_nodes=dataset.hdata.num_nodes,
        num_classes=num_classes,
    )
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

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
        batch_size=64,
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

    model = VilLainClassifier(
        encoder_config={
            "embedding_dim": 128,
            "labels_per_subspace": 8,
            "training_steps": 4,
            "generation_steps": 28,
            "tau": 1.0,
            "eps": 1e-10,
            # 40% weight on the VilLain loss, 60% weight on the classifier loss
            "villain_loss_weight": 0.4,
            # Transductive splits keep global node IDs available for the full node space.
            "num_nodes": dataset.hdata.num_nodes,
        },
        classifier_config={
            "out_channels": num_classes,
        },
        lr=0.01,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="villain",
            version="node-classification",
            model=model,
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
