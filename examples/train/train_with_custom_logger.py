from pathlib import Path

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hypertorch.hlp import MLPHlpModule
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig
from hypertorch.data import (
    AlgebraDataset,
    DataLoader,
    LaplacianPositionalEncodingEnricher,
    RandomNegativeSampler,
    SamplingStrategy,
)

from typing import Any, ClassVar
from lightning.pytorch.loggers import Logger


class CustomLogger(Logger):
    __shared_stores: ClassVar[dict[str, dict[str, dict[str, Any]]]] = {}

    def __init__(
        self, experiment_name: str, model_name: str, save_dir: str | Path = "hypertorch_logs"
    ):
        super().__init__()
        self.__experiment_name = experiment_name
        self.__model_name = model_name
        self.__save_dir = Path(save_dir) / experiment_name

        if experiment_name not in self.__shared_stores:
            self.__shared_stores[experiment_name] = {}

    @property
    def name(self) -> str:
        return "CustomLogger"

    @property
    def version(self) -> str:
        return "0.1"

    @property
    def experiment_name(self) -> str | Path:
        return self.__experiment_name

    @property
    def store(self) -> dict[str, dict[str, Any]]:
        """Access the shared store for the current experiment."""
        return dict(self.__shared_stores.get(self.__experiment_name, {}))

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Accumulate metrics for this model. Called by Lightning on every log step.

        Keeps only the latest value for each metric name. For example, if
        "val_auc" is logged at step 10 and step 20, only the step 20 value is kept.
        """
        store = self.__shared_stores[self.__experiment_name]
        if self.__model_name not in store:
            store[self.__model_name] = {}
        store[self.__model_name].update(metrics)

    def finalize(self, status: str) -> None:
        """Save accumulated metrics to a JSON file."""
        import json

        store = self.__shared_stores.get(self.__experiment_name, {})
        self.__save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.__save_dir / "results.json"

        with open(save_path, "w") as f:
            json.dump(store, f, indent=4)

        print(f"Finalized experiment '{self.__experiment_name}' with status '{status}'.")
        print(f"Saved results to {save_path}")


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

    dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE)
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    # Split dataset into train, val and test (70/10/20)
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        node_space_setting="transductive",
        cover_all_nodes_in_train_split=True,
        shuffle=True,
        seed=42,
    )
    if verbose:
        print(f"Train dataset:\n {train_dataset.hdata}\n")
        print(f"Val dataset:\n {val_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

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

        if verbose:
            print(f"{name} dataset after adding negative samples: {ds_with_negatives.hdata}\n")

    print("Enriching node features...")

    train_dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(
            num_features=num_features,
            # We are using transductive with all nodes coverage in the train split
            num_nodes=dataset.hdata.num_nodes,
        ),
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

    print("Creating dataloaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,  # or 256
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

    mean_mlp_module = MLPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "out_channels": num_features,
            "hidden_channels": 64,
            "num_layers": 3,
            "drop_rate": 0.3,
        },
        aggregation="mean",
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="mlp",
            version="mean",
            model=mean_mlp_module,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
    ]

    print("Starting training and evaluation...")
    json_logger = CustomLogger(experiment_name="custom_logger_example", model_name="mlp")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=100,
        accelerator="auto",
        log_every_n_steps=10,
        enable_checkpointing=False,
        auto_start_tensorboard=True,
        auto_wait=True,
        devices=1,
        test_devices=1,
        logger=json_logger,  # here you can pass the custom logger to the trainer
    ) as trainer:
        trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader, verbose=True)
        trainer.test_all(dataloader=test_loader, verbose=True)

    print("Complete!")
