from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score
from hypertorch.data import AlgebraDataset, DataLoader
from hypertorch.nc import CommonNeighborsNcModule
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

    dataset = AlgebraDataset(sampling_strategy="node", task="node-classification")
    dataset.hdata.y = node_labels_from_node_degrees(
        node_incidences=dataset.hdata.hyperedge_index[0],
        num_nodes=dataset.hdata.num_nodes,
    )
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    # Split nodes into train and test masks over the same transductive hypergraph.
    # CommonNeighbors uses the labeled train nodes as class references.
    train_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.3],
        node_space_setting="transductive",
        shuffle=True,
        seed=42,
    )
    if verbose:
        print(f"Train dataset:\n {train_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

    print("Creating dataloader...")

    test_loader = DataLoader(
        test_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = CommonNeighborsNcModule(
        train_hdata=train_dataset.hdata,
        num_classes=num_classes,
        aggregation="mean",
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="common-neighbors",
            version="node-classification",
            model=model,
            is_trainable=False,
            test_dataloader=test_loader,
        ),
    ]

    print("Starting evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=0,
        accelerator="auto",
        enable_checkpointing=False,
        devices=1,
        test_devices=1,
    ) as trainer:
        trainer.test_all(dataloader=test_loader, verbose=True)

    print("Complete!")
