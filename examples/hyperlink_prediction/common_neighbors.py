from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hypertorch.hyperlink_prediction import CommonNeighborsPredictor
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig
from hypertorch.data import (
    AlgebraDataset,
    DataLoader,
    RandomNegativeSampler,
)


if __name__ == "__main__":
    verbose = False
    num_workers = 8
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
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    # Split dataset into train and test portions. CommonNeighbors uses the training
    # hyperedges as its known graph, but does not train or validate parameters.
    train_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.3], shuffle=True, seed=42, node_space_setting="transductive"
    )
    if verbose:
        print(f"Train dataset:\n {train_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

    # Add negative samples only to the test split, since the model is evaluated directly.
    negative_sampler = RandomNegativeSampler(
        num_negative_samples=int(len(test_dataset) * 0.6),
        num_nodes_per_sample=int(test_dataset.stats()["avg_degree_hyperedge"]),
    )
    test_dataset = test_dataset.add_negative_samples(negative_sampler, seed=42)

    if verbose:
        print(f"Test dataset after adding negative samples: {test_dataset.hdata}\n")

    print("Creating dataloader...")

    test_loader = DataLoader(
        test_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = CommonNeighborsPredictor(
        train_hdata=train_dataset.hdata,
        aggregation="mean",
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="common-neighbors",
            version="hyperlink-prediction",
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
