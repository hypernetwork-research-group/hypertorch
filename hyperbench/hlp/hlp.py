import lightning as L
from enum import Enum

from torch import Tensor, nn
from typing import Callable, Dict, Optional, TypeAlias
from hyperbench.train import NegativeSampler, NegativeSamplingSchedule
from hyperbench.types import HData


MetricFn: TypeAlias = Callable[[Tensor, Tensor], Tensor]


class Stage(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class HlpModule(L.LightningModule):
    """
    A LightningModule for the HLP model with optional negative sampling.

    Args:
        decoder: Decoder module to use to predict whether hyperedges are positive or negative.
        loss_fn: Loss function.
        encoder: Optional encoder module. Defaults to ``None`` as not all HLP model will use an encoder.
        metrics: Optional dictionary of metric functions to compute during evaluation.
        negative_sampler: Optional negative sampler. If None, no negative sampling is performed.
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
        self.negative_sampler = negative_sampler
        self.negative_sampling_schedule = negative_sampling_schedule
        self.negative_sampling_every_n = negative_sampling_every_n
        self.__cached_negative_samples: Optional[HData] = None

    def _compute_loss(
        self,
        scores: Tensor,
        labels: Tensor,
        batch_size: int,
        stage: Stage,
    ) -> Tensor:
        """Compute and log loss based on scores and labels."""
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
        """Compute and log metrics based on scores and labels."""
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
        if self.negative_sampler is None:
            return False

        match self.negative_sampling_schedule:
            case NegativeSamplingSchedule.EVERY_EPOCH:
                return True
            case NegativeSamplingSchedule.EVERY_N_EPOCHS:
                return self.current_epoch % self.negative_sampling_every_n == 0
            case NegativeSamplingSchedule.FIRST_EPOCH:
                return self.current_epoch == 0

        return False

    def _sample_negatives(self, batch: HData) -> HData:
        """Sample fresh negatives if the schedule requires it, otherwise return cache."""
        if self._should_sample_negatives():
            if self.negative_sampler is not None:
                self.__cached_negative_samples = self.negative_sampler.sample(batch)

        if self.__cached_negative_samples is None:
            raise ValueError("Negative sampling requested but no negative sampler is provided.")

        return self.__cached_negative_samples
