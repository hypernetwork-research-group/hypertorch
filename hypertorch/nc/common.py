import torch
import lightning as L

from torch import Tensor, nn
from typing import Any
from torchmetrics import MetricCollection
from hypertorch.hlp.common import stage_metric_name, stage_metric_prefix
from hypertorch.types import HData
from hypertorch.utils import Stage


class NcModule(L.LightningModule):
    """
    A LightningModule base class for multiclass node-classification models.

    Attributes:
        encoder: Optional encoder module. Defaults to ``None``.
        classifier: Module that maps node features to class logits.
        loss_fn: Multiclass loss function.
        metrics_log_kwargs: Keyword arguments passed to metric log calls.
        train_metrics: Optional training metrics.
        val_metrics: Optional validation metrics.
        test_metrics: Optional test metrics.
    """

    def __init__(
        self,
        classifier: nn.Module,
        loss_fn: nn.Module,
        encoder: nn.Module | None = None,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the NC module.

        Args:
            classifier: Module producing one class-logit row per node.
            loss_fn: Loss function used on supervised target nodes.
            encoder: Optional encoder module. Defaults to ``None``.
            metrics: Optional metric collection cloned independently per stage.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
        """
        super().__init__()
        self.encoder: nn.Module | None = encoder
        self.classifier: nn.Module = classifier
        self.loss_fn: nn.Module = loss_fn
        self.metrics_log_kwargs: dict[str, Any] = metrics_log_kwargs or {}

        if metrics is not None:
            self.train_metrics: MetricCollection | None = metrics.clone(
                prefix=stage_metric_prefix(Stage.TRAIN)
            )
            self.val_metrics: MetricCollection | None = metrics.clone(
                prefix=stage_metric_prefix(Stage.VAL)
            )
            self.test_metrics: MetricCollection | None = metrics.clone(
                prefix=stage_metric_prefix(Stage.TEST)
            )
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None

    def target_logits_and_labels(self, logits: Tensor, batch: HData) -> tuple[Tensor, Tensor]:
        """
        Select supervised node logits and labels from a batch.

        Args:
            logits: Node logits of shape ``[num_nodes, num_classes]``.
            batch: Batch containing node labels and an optional target mask.

        Returns:
            target_logits: Logits for supervised nodes.
            target_labels: Labels for supervised nodes.
        """
        target_node_mask = batch.target_node_mask
        if target_node_mask is None:
            target_node_mask = torch.ones(batch.num_nodes, dtype=torch.bool, device=batch.device)

        target_logits = logits[target_node_mask]
        target_labels = batch.y[target_node_mask]
        return target_logits, target_labels

    def _compute_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        batch_size: int,
        stage: Stage,
    ) -> Tensor:
        """
        Compute and log loss based on logits and labels.

        Args:
            logits: The predicted logits from the model.
            labels: The true labels corresponding to the logits.
            batch_size: The size of the current batch, used for logging.
            stage: The current stage (train/val/test) for logging purposes.

        Returns:
            loss: The computed loss tensor.
        """
        loss = self.loss_fn(logits, labels)
        self.log(
            name=stage_metric_name(stage, "loss"),
            value=loss,
            prog_bar=True,
            batch_size=batch_size,
            **self.metrics_log_kwargs,
        )
        return loss

    def _compute_metrics(
        self,
        logits: Tensor,
        labels: Tensor,
        batch_size: int,
        stage: Stage,
    ) -> None:
        """
        Compute and log metrics based on logits and labels.

        Uses class-based torchmetrics with proper multi-batch accumulation:
        1. ``update()`` accumulates predictions/targets across batches.
        2. Passing the MetricCollection to ``self.log_dict()`` tells Lightning to call
            ``compute()`` at epoch end and ``reset()`` automatically.

        Args:
            logits: The predicted logits from the model.
            labels: The true labels corresponding to the logits.
            batch_size: The size of the current batch, used for logging.
            stage: The current stage (train/val/test) for logging purposes.
        """
        stage_metrics = self._get_stage_metrics(stage)
        if stage_metrics is None:
            return  # No metrics to compute for this stage
        self._configure_metric_distributed_available(stage_metrics)

        # Accumulate predictions/targets for this batch
        stage_metrics.update(logits, labels)
        self.log_dict(
            stage_metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,  # Compute and log metrics at epoch end for proper accumulation
            batch_size=batch_size,
            **self.metrics_log_kwargs,
        )

    def _get_stage_metrics(self, stage: Stage) -> MetricCollection | None:
        """
        Return the metric collection for the given stage.
        """
        match stage:
            case Stage.TRAIN:
                return self.train_metrics
            case Stage.VAL:
                return self.val_metrics
            case Stage.TEST:
                return self.test_metrics
            case _:
                raise ValueError(f"Unrecognized stage: {stage}")

    def _configure_metric_distributed_available(self, metrics: MetricCollection) -> None:
        """
        Make torchmetrics sync decisions follow the active Lightning trainer.
        """
        for metric in metrics.values(copy_state=False):
            metric.distributed_available_fn = self._distributed_available_fn

    def _distributed_available_fn(self) -> bool:
        """
        Return whether metrics should synchronize for the current trainer.
        """
        trainer = self._trainer
        if trainer is None:
            return False
        return (
            trainer.world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
