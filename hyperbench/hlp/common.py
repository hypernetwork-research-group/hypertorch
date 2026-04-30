import torch
import lightning as L

from torch import Tensor, nn
from typing import Any, Dict, Optional
from torchmetrics import MetricCollection
from hyperbench.train import NegativeSampler, NegativeSamplingSchedule, NegativeSamplingScheduler
from hyperbench.types import HData
from hyperbench.utils import Stage


class HlpModule(L.LightningModule):
    """
    A LightningModule for HLP models with optional negative sampling.

    Args:
        encoder: Optional encoder module. Defaults to ``None`` as not all HLP model will use an encoder.
        decoder: Decoder module to use to predict whether hyperedges are positive or negative.
        loss_fn: Loss function.
        metrics: Optional ``MetricCollection`` of torchmetrics to compute during evaluation.
            Cloned per stage (train, val, test) for independent state accumulation.
        negative_sampler: Optional negative sampler. If ``None``, no negative sampling is performed.
        negative_sampling_schedule: When to perform negative sampling during training. Defaults to ``EVERY_EPOCH``.
        negative_sampling_every_n: If using ``EVERY_N_EPOCHS`` schedule, how many epochs between negative sampling runs. Defaults to ``1``.
    """

    def __init__(
        self,
        decoder: nn.Module,
        loss_fn: nn.Module,
        encoder: Optional[nn.Module] = None,
        metrics: Optional[MetricCollection] = None,
        negative_sampler: Optional[NegativeSampler] = None,
        negative_sampling_schedule: NegativeSamplingSchedule = NegativeSamplingSchedule.EVERY_EPOCH,
        negative_sampling_every_n: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn

        if metrics is not None:
            self.train_metrics = metrics.clone(prefix=f"{Stage.TRAIN.value}_")
            self.val_metrics = metrics.clone(prefix=f"{Stage.VAL.value}_")
            self.test_metrics = metrics.clone(prefix=f"{Stage.TEST.value}_")
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None

        self.__negative_sampling_scheduler = None
        if negative_sampler is not None:
            self.__negative_sampling_scheduler = NegativeSamplingScheduler(
                negative_sampler,
                negative_sampling_schedule,
                negative_sampling_every_n,
            )

    @property
    def negative_sampling_config(self) -> Dict[str, Any]:
        if self.__negative_sampling_scheduler is None:
            return {}
        return self.__negative_sampling_scheduler.config

    def _compute_loss(
        self,
        scores: Tensor,
        labels: Tensor,
        batch_size: int,
        stage: Stage,
    ) -> Tensor:
        """
        Compute and log loss based on scores and labels.

        Args:
            scores: The predicted scores from the model.
            labels: The true labels corresponding to the scores.
            batch_size: The size of the current batch, used for logging.
            stage: The current stage (train/val/test) for logging purposes.

        Returns:
            The computed loss tensor.
        """
        loss = self.loss_fn(scores, labels)
        self.log(name=f"{stage.value}_loss", value=loss, prog_bar=True, batch_size=batch_size)
        return loss

    def _compute_metrics(
        self,
        scores: Tensor,
        labels: Tensor,
        batch_size: int,
        stage: Stage,
    ) -> None:
        """
        Compute and log metrics based on scores and labels.

        Uses class-based torchmetrics with proper multi-batch accumulation:
        1. ``update()`` accumulates predictions/targets across batches.
        2. Passing the MetricCollection to ``self.log_dict()`` tells Lightning to call ``compute()`` at epoch end and ``reset()`` automatically.

        Args:
            scores: The predicted scores (logits) from the model.
            labels: The true labels corresponding to the scores.
            batch_size: The size of the current batch, used for logging.
            stage: The current stage (train/val/test) for logging purposes.
        """
        stage_metrics = self._get_stage_metrics(stage)
        if stage_metrics is None:
            return  # No metrics to compute

        # Apply sigmoid to convert logits to probabilities as BinaryAUROC
        # and BinaryAveragePrecision expect probabilities in [0, 1]
        preds = torch.sigmoid(scores)
        targets = labels.long()

        # Accumulate predictions/targets for this batch
        stage_metrics.update(preds, targets)

        self.log_dict(
            stage_metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,  # Compute and log metrics at epoch end, not per step, for proper accumulation
            batch_size=batch_size,
        )

    def _get_stage_metrics(self, stage: Stage) -> Optional[MetricCollection]:
        """
        Return the metric collection for the given stage, or ``None``.

        Args:
            stage: The current stage (train/val/test) for which to get metrics.

        Returns:
            The metric collection corresponding to the given stage, or ``None`` if no metrics are configured.
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

    def _should_sample_negatives(self) -> bool:
        """Whether to resample negatives for the current epoch."""
        if self.__negative_sampling_scheduler is None:
            raise ValueError(
                "Asked to check negative sampling schedule but no negative sampler is configured."
            )
        return self.__negative_sampling_scheduler.should_sample(self.current_epoch)

    def _sample_negatives(self, batch: HData) -> HData:
        """
        Sample fresh negatives if the schedule requires it, otherwise return cache.

        Args:
            batch: The current batch of data for which to sample negatives.

        Returns:
            A batch of negative samples, either freshly sampled or from cache.
        """
        if self.__negative_sampling_scheduler is None:
            raise ValueError("Asked to sample negatives but no negative sampler is not configured.")
        return self.__negative_sampling_scheduler.sample(batch, self.current_epoch)
