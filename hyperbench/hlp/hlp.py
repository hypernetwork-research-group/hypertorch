import lightning as L
from enum import Enum

from torch import Tensor, nn
from typing import Any, Callable, Dict, Optional, TypeAlias
from hyperbench.train import NegativeSampler, NegativeSamplingSchedule, NegativeSamplingScheduler
from hyperbench.types import HData


MetricFn: TypeAlias = Callable[[Tensor, Tensor], Tensor]


class Stage(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class HlpModule(L.LightningModule):
    """
    A LightningModule for HLP models with optional negative sampling.

    Args:
        encoder: Optional encoder module. Defaults to ``None`` as not all HLP model will use an encoder.
        decoder: Decoder module to use to predict whether hyperedges are positive or negative.
        loss_fn: Loss function.
        metrics: Optional dictionary of metric functions to compute during evaluation.
        negative_sampler: Optional negative sampler. If ``None``, no negative sampling is performed.
        negative_sampling_schedule: When to perform negative sampling during training. Defaults to ``EVERY_EPOCH``.
        negative_sampling_every_n: If using ``EVERY_N_EPOCHS`` schedule, how many epochs between negative sampling runs. Defaults to ``1``.
    """

    def __init__(
        self,
        decoder: nn.Module,
        loss_fn: nn.Module,
        encoder: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, MetricFn]] = None,
        negative_sampler: Optional[NegativeSampler] = None,
        negative_sampling_schedule: NegativeSamplingSchedule = NegativeSamplingSchedule.EVERY_EPOCH,
        negative_sampling_every_n: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.metrics = metrics if metrics is not None else {}

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

        Args:
            scores: The predicted scores from the model.
            labels: The true labels corresponding to the scores.
            batch_size: The size of the current batch, used for logging.
            stage: The current stage (train/val/test) for logging purposes.
        """
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(scores, labels)
            self.log(
                name=f"{stage.value}_{metric_name}",
                value=metric_value,
                prog_bar=True,
                batch_size=batch_size,
            )

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
