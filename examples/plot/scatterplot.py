from pathlib import Path
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from hypertorch.data import (
    AlgebraDataset,
    DataLoader,
    LaplacianPositionalEncodingEnricher,
    RandomNegativeSampler,
)
from hypertorch.hyperlink_prediction import MLPPredictor
from hypertorch.plot import LogParser, ScatterPlotter
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig


def build_mlp_model(
    num_features: int,
    hidden_channels: int,
    aggregation: str,
    drop_rate: float,
    metrics: MetricCollection,
) -> MLPPredictor:
    """Helper to instantiate MLPPredictor models with varied hyperparameters."""
    return MLPPredictor(
        encoder_config={
            "in_channels": num_features,
            "out_channels": num_features,
            "hidden_channels": hidden_channels,
            "num_layers": 2,
            "drop_rate": drop_rate,
        },
        aggregation=aggregation,
        metrics=metrics,
    )


def main() -> None:
    """
    Train 3 distinct model configurations for 3 epochs each and generate
    Train vs. Val generalization scatter plots comparing them on the y = x line.
    """
    print("Loading and preparing AlgebraDataset...")
    dataset = AlgebraDataset(sampling_strategy="hyperedge")

    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        node_space_setting="transductive",
        shuffle=True,
        seed=42,
    )

    for name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        negative_sampler = RandomNegativeSampler(
            num_negative_samples=len(ds),
            num_nodes_per_sample=int(ds.stats()["avg_degree_hyperedge"]),
        )
        ds_with_negatives = ds.add_negative_samples(negative_sampler, seed=42)
        if name == "Train":
            train_dataset = ds_with_negatives
        elif name == "Val":
            val_dataset = ds_with_negatives
        else:
            test_dataset = ds_with_negatives

    num_features = 32
    train_dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(
            num_features=num_features,
            num_nodes=train_dataset.hdata.num_nodes,
        ),
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

    train_loader = DataLoader(train_dataset, sample_full_hypergraph=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Shared metrics collection
    metrics_collection = MetricCollection(
        {
            "auc": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
        }
    )

    model_configs = [
        ModelConfig(
            name="mlp",
            version="mean_h64",
            model=build_mlp_model(
                num_features=num_features,
                hidden_channels=64,
                aggregation="mean",
                drop_rate=0.2,
                metrics=metrics_collection,
            ),
        ),
        ModelConfig(
            name="mlp",
            version="sum_h32",
            model=build_mlp_model(
                num_features=num_features,
                hidden_channels=32,
                aggregation="sum",
                drop_rate=0.1,
                metrics=metrics_collection,
            ),
        ),
        ModelConfig(
            name="mlp",
            version="max_h128",
            model=build_mlp_model(
                num_features=num_features,
                hidden_channels=128,
                aggregation="max",
                drop_rate=0.3,
                metrics=metrics_collection,
            ),
        ),
    ]

    # Train all 3 models for 3 epochs
    print("\nTraining 3 MLP variants for 3 epochs each...")
    with MultiModelTrainer(
        model_configs=model_configs,
        max_epochs=3,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=True,
    ) as trainer:
        trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader)
        trainer.test_all(dataloader=test_loader)

    """Scatterplot implementation here."""

    print("\nParsing experiment metrics...")
    parser = LogParser()
    parser.discover_from_latest_dir(num=3)
    parsed_runs = parser.parse_all()

    print("Generating Train vs. Val generalization scatter plots...")
    plotter = ScatterPlotter()
    averaged_runs = [metrics.average() for metrics in parsed_runs]

    for metrics in averaged_runs:
        print(f"Loaded run: '{metrics.experiment_name}' with metrics {metrics.names()}")

    saved_plots = plotter.plot(averaged_runs)

    print(f"\nGenerated {len(saved_plots)} plot(s):")
    for plot_path in saved_plots:
        print(f" -> {Path(plot_path).resolve()}")


if __name__ == "__main__":
    main()