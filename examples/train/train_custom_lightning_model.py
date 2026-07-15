import lightning as L
import torch

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from torch import Tensor, nn, optim
from hypertorch.data import DataLoader, Dataset
from hypertorch.train import MultiModelTrainer
from hypertorch.types import HData, ModelConfig


class CustomHlpPredictor(L.LightningModule):
    """A hyperlink predictor implemented as a Lightning module."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 8,
        metrics: MetricCollection | None = None,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = 0.01
        self.train_metrics = metrics.clone(prefix="train/") if metrics is not None else None
        self.val_metrics = metrics.clone(prefix="val/") if metrics is not None else None
        self.test_metrics = metrics.clone(prefix="test/") if metrics is not None else None

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute one binary-classification logit per sample.

        Args:
            x: Input tensor of shape ``(batch_size, in_channels)``.

        Returns:
            logits: Tensor of shape ``(batch_size,)``.
        """
        return self.network(x).squeeze(-1)

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Compute and log the training loss.

        Args:
            batch: HData containing hyperedge features and labels.
            batch_idx: Batch index, unused.

        Returns:
            loss: Binary cross-entropy loss.
        """
        return self.__shared_step(batch, "train")

    def validation_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Compute and log the validation loss.

        Args:
            batch: HData containing hyperedge features and labels.
            batch_idx: Batch index, unused.

        Returns:
            loss: Binary cross-entropy loss.
        """
        return self.__shared_step(batch, "val")

    def test_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Compute and log the test loss.

        Args:
            batch: HData containing hyperedge features and labels.
            batch_idx: Batch index, unused.

        Returns:
            loss: Binary cross-entropy loss.
        """
        return self.__shared_step(batch, "test")

    def configure_optimizers(self) -> optim.Adam:
        """
        Configure the optimizer.

        Returns:
            optimizer: Adam optimizer for the model parameters.
        """
        return optim.Adam(self.parameters(), lr=self.lr)

    def __shared_step(self, batch: HData, stage: str) -> Tensor:
        """
        Compute and log the loss for a given stage (train, val, test).

        Args:
            batch: HData containing hyperedge features and labels.
            stage: Stage name, one of "train", "val", or "test".

        Returns:
            loss: Binary cross-entropy loss.
        """
        x = batch.x[batch.target_hyperedge_mask]

        logits = self(x)
        labels = batch.y[batch.target_hyperedge_mask]
        loss = self.loss_fn(logits, labels)

        batch_size = labels.size(0)
        self.log(f"{stage}/loss", loss, batch_size=batch_size)

        metrics = self.__metrics_for_stage(stage)
        if metrics is not None:
            metrics.update(torch.sigmoid(logits), labels.long())
            self.log_dict(
                metrics,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # Compute and log metrics at epoch end
                batch_size=batch_size,
            )
        return loss

    def __metrics_for_stage(self, stage: str) -> MetricCollection | None:
        match stage:
            case "train":
                return self.train_metrics
            case "val":
                return self.val_metrics
            case "test":
                return self.test_metrics
            case _:
                raise ValueError(f"Unsupported stage: {stage}")


def create_dataset_for_hyperlink_prediction(features: Tensor, labels: Tensor) -> Dataset:
    """
    Create a hyperlink prediction dataset for the example's in-memory data.

    Args:
        features: Input features, one per node.
        labels: Binary labels, one per hyperedge.

    Returns:
        dataset: hypertorch dataset containing the supplied samples.
    """
    num_nodes = features.size(0)
    node_ids = torch.arange(num_nodes, dtype=torch.long)
    hyperedge_ids = node_ids.clone()  # Each node is its own hyperedge for this example
    hdata = HData(
        x=features,
        hyperedge_index=torch.stack((node_ids, hyperedge_ids)),
        y=labels,
        task="hyperlink-prediction",
    )
    return Dataset.from_hdata(hdata, sampling_strategy="hyperedge")


if __name__ == "__main__":
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

    train_dataset = create_dataset_for_hyperlink_prediction(
        features=torch.tensor(
            [
                [-2.0, -1.0],
                [-1.0, -2.0],
                [-1.0, -1.0],
                [-0.5, -1.5],
                [0.5, 1.5],
                [1.0, 1.0],
                [1.0, 2.0],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        labels=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    )
    val_dataset = create_dataset_for_hyperlink_prediction(
        features=torch.tensor(
            [[-1.5, -0.5], [-0.5, -0.5], [0.5, 0.5], [1.5, 0.5]], dtype=torch.float32
        ),
        labels=torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
    )
    test_dataset = create_dataset_for_hyperlink_prediction(
        features=torch.tensor(
            [[-2.0, -0.5], [-0.5, -2.0], [0.5, 2.0], [2.0, 0.5]], dtype=torch.float32
        ),
        labels=torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
    )

    data_module = DataLoader.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = CustomHlpPredictor(in_channels=2, hidden_channels=8, metrics=metrics)
    model_config = ModelConfig(
        name="custom_hlp_predictor",
        version="lightning",
        model=model,
    )

    with MultiModelTrainer(
        model_configs=[model_config],
        experiment_name="custom_lightning_model",
        max_epochs=20,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
    ) as trainer:
        trainer.fit_all(
            train_dataloader=data_module.train_dataloader(),
            val_dataloader=data_module.val_dataloader(),
        )
        trainer.test_all(dataloader=data_module.test_dataloader())
