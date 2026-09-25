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
from hypertorch.train import LinePlotter, LogParser, MultiModelTrainer
from hypertorch.types import ModelConfig


def main() -> None:
    """
    Creating a basic model and running it in order to generate the necessary
    experiment folder in order to showcase the parsing and plotting functions
    """
    print("Loading and preparing AlgebraDataset...")
    dataset = AlgebraDataset(sampling_strategy="hyperedge")

    # Split dataset into train, val, and test splits
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        node_space_setting="transductive",
        shuffle=True,
        seed=42,
    )

    # Add negative samples to all splits
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

    # Enrich node features
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

    # Create dataloaders
    train_loader = DataLoader(train_dataset, sample_full_hypergraph=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Configure metrics
    metrics_collection = MetricCollection(
        {
            "auc": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
        }
    )

    # Configure model
    model = MLPPredictor(
        encoder_config={
            "in_channels": num_features,
            "out_channels": num_features,
            "hidden_channels": 64,
            "num_layers": 2,
            "drop_rate": 0.2,
        },
        aggregation="mean",
        metrics=metrics_collection,
    )

    model_configs = [ModelConfig(name="mlp", version="mean", model=model)]

    # 3-epoch experiment to populate experiment metrics
    print("\nTraining MLP model for 3 epochs...")
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

    """Application of LogParser and LinePlot can be seen here:"""

    # Instancing LogParser to find and parse the latest metrics
    print("\nParsing experiment metrics...")
    parser = LogParser()
    metrics = parser.parse()

    # Metrics stores the directory as well as the names of the metrics found
    print(f"Discovered experiment directory: {metrics.experiment_dir}")
    print(f"Available metrics to plot: {metrics.names()}")

    # instantiating LinePlotter to plot metrics
    print("Generating line plots...")
    plotter = LinePlotter()
    saved_plots = plotter.plot(metrics)

    print(f"\nGenerated {len(saved_plots)} plot(s):")
    for plot_path in saved_plots:
        print(f" -> {Path(plot_path).resolve()}")


if __name__ == "__main__":
    main()
