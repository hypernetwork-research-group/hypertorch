import torch.nn.functional as F

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from pathlib import Path
from typing import cast
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor
from hyperbench.hlp import MLPHlpModule
from hyperbench.train import MultiModelTrainer
from hyperbench.types import ModelConfig
from hyperbench.data import (
    AlgebraDataset,
    DataLoader,
    LaplacianPositionalEncodingEnricher,
    RandomNegativeSampler,
    SamplingStrategy,
)


def last_checkpoint_path(config: ModelConfig) -> Path:
    if config.trainer is None:
        raise RuntimeError(f"Trainer not initialized for model {config.full_model_name()}.")

    checkpoint_callback = config.trainer.checkpoint_callback
    if not isinstance(checkpoint_callback, ModelCheckpoint):
        raise RuntimeError(f"No checkpoint callback found for model {config.full_model_name()}.")
    if checkpoint_callback.last_model_path == "":
        raise RuntimeError(f"No last checkpoint was saved for model {config.full_model_name()}.")

    return Path(checkpoint_callback.last_model_path)


if __name__ == "__main__":
    verbose = False
    num_features = 32
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

    dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE)
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    train_dataset, val_dataset, predict_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        shuffle=True,
        seed=42,
        node_space_setting="transductive",
    )

    for name, ds in [("Train", train_dataset), ("Val", val_dataset)]:
        negative_sampler = RandomNegativeSampler(
            num_negative_samples=ds.hdata.num_hyperedges,
            num_nodes_per_sample=int(ds.stats()["avg_degree_hyperedge"]),
        )
        ds_with_negatives = ds.add_negative_samples(negative_sampler, seed=42)

        if name == "Train":
            train_dataset = ds_with_negatives
        else:
            val_dataset = ds_with_negatives

    print("Enriching node features...")

    train_dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(
            num_features=num_features,
            num_nodes=train_dataset.hdata.num_nodes,
        ),
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    predict_dataset.enrich_node_features_from(train_dataset)

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

    train_config = ModelConfig(
        name="mlp",
        version="checkpoint-predict",
        model=MLPHlpModule(
            encoder_config={
                "in_channels": num_features,
                "out_channels": num_features,
                "hidden_channels": 64,
                "num_layers": 3,
                "drop_rate": 0.3,
            },
            aggregation="mean",
            metrics=metrics,
        ),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    print("Training model and saving the last checkpoint...")

    with MultiModelTrainer(
        model_configs=[train_config],
        experiment_name="checkpoint_predict",
        callbacks=[ModelCheckpoint(save_last=True)],
        max_epochs=100,
        accelerator="auto",
        log_every_n_steps=10,
        enable_checkpointing=True,
        auto_start_tensorboard=True,
        auto_wait=True,
    ) as trainer:
        trainer.fit_all(verbose=True)
        checkpoint_path = last_checkpoint_path(train_config)

    print(f"Saved checkpoint: {checkpoint_path}")
    print("Loading checkpoint into a fresh model before prediction...")

    predict_loader = DataLoader(
        predict_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    predict_config = ModelConfig(
        name="mlp",
        version="checkpoint-predict",
        model=MLPHlpModule(
            encoder_config={
                "in_channels": num_features,
                "out_channels": num_features,
                "hidden_channels": 64,
                "num_layers": 3,
                "drop_rate": 0.3,
            },
            aggregation="mean",
        ),
    )

    with MultiModelTrainer(
        model_configs=[predict_config],
        experiment_name="checkpoint_predict_load",
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
        devices=1,
        test_devices=1,
    ):
        if predict_config.trainer is None:
            raise RuntimeError(
                f"Trainer not initialized for model {predict_config.full_model_name()}."
            )

        prediction_batches = predict_config.trainer.predict(
            model=predict_config.model,
            dataloaders=predict_loader,
            ckpt_path=checkpoint_path,
        )

    if prediction_batches is None or len(prediction_batches) < 1:
        raise RuntimeError("No predictions were returned.")

    prediction_batches_as_tensor = cast(Tensor, prediction_batches[0])
    print(f"Predicted {prediction_batches_as_tensor.numel()} hyperedge scores.")

    probs = F.sigmoid(prediction_batches_as_tensor)
    print("First 18 predictions:")
    print(probs[:18])
    print("Complete!")
