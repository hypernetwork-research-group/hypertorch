import torch
import lightning as L

from torch import Tensor, nn
from typing import Any
from torchmetrics import MetricCollection
from hyperbench.data import NegativeSampler, NegativeSamplingSchedule, NegativeSamplingScheduler
from hyperbench.types import HData
from hyperbench.utils import Stage


class HlpModule(L.LightningModule):
    """
    A LightningModule for HLP models with optional negative sampling.

    Attributes:
        encoder: Optional encoder module. Defaults to ``None`` as not
            all HLP model use an encoder.
        decoder: Decoder module to use to predict whether hyperedges are positive or negative.
        loss_fn: Loss function.
        metrics_log_kwargs: Additional keyword arguments to pass to all ``self.log`` calls
            for metrics. Useful for configuring distributed synchronization behavior of
            ``torchmetrics``. Defaults to ``None``.
        train_metrics: Optional metric collection for training.
        val_metrics: Optional metric collection for validation.
        test_metrics: Optional metric collection for testing.
        __negative_sampling_scheduler: Optional negative-sampling scheduler.
    """

    def __init__(
        self,
        decoder: nn.Module,
        loss_fn: nn.Module,
        encoder: nn.Module | None = None,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
        negative_sampler: NegativeSampler | None = None,
        negative_sampling_schedule: NegativeSamplingSchedule = "every_epoch",
        negative_sampling_every_n: int = 1,
    ):
        """
        Initialize the HLP Lightning module.

        Args:
            decoder: Decoder module used to score hyperedges.
            loss_fn: Loss function.
            encoder: Optional encoder module. Defaults to ``None``.
            metrics: Optional metric collection cloned independently per stage.
                Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Defaults to ``None``.
            negative_sampler: Optional negative sampler. Defaults to ``None``.
            negative_sampling_schedule: Schedule controlling when negatives are sampled.
                Defaults to ``"every_epoch"``.
            negative_sampling_every_n: Epoch interval for ``"every_n_epochs"`` scheduling.
                Defaults to ``1``.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.metrics_log_kwargs = metrics_log_kwargs or {}

        if metrics is not None:
            self.train_metrics = metrics.clone(prefix=stage_metric_prefix(Stage.TRAIN))
            self.val_metrics = metrics.clone(prefix=stage_metric_prefix(Stage.VAL))
            self.test_metrics = metrics.clone(prefix=stage_metric_prefix(Stage.TEST))
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
    def negative_sampling_config(self) -> dict[str, Any]:
        """
        Return the configured negative-sampling options.

        Returns:
            config: Scheduler configuration, or an empty dictionary when negative sampling is off.
        """
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
            loss: The computed loss tensor.
        """
        loss = self.loss_fn(scores, labels)
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
        scores: Tensor,
        labels: Tensor,
        batch_size: int,
        stage: Stage,
    ) -> None:
        """
        Compute and log metrics based on scores and labels.

        Uses class-based torchmetrics with proper multi-batch accumulation:
        1. ``update()`` accumulates predictions/targets across batches.
        2. Passing the MetricCollection to ``self.log_dict()`` tells Lightning to call
            ``compute()`` at epoch end and ``reset()`` automatically.

        Args:
            scores: The predicted scores (logits) from the model.
            labels: The true labels corresponding to the scores.
            batch_size: The size of the current batch, used for logging.
            stage: The current stage (train/val/test) for logging purposes.
        """
        stage_metrics = self._get_stage_metrics(stage)
        if stage_metrics is None:
            return  # No metrics to compute
        self._configure_metric_distributed_available(stage_metrics)

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
            on_epoch=True,  # Compute and log metrics at epoch end for proper accumulation
            batch_size=batch_size,
            **self.metrics_log_kwargs,
        )

    def _get_stage_metrics(self, stage: Stage) -> MetricCollection | None:
        """
        Return the metric collection for the given stage, or ``None``.

        Args:
            stage: The current stage (train/val/test) for which to get metrics.

        Returns:
            metrics: The metric collection corresponding to the given stage, or ``None``
                if no metrics are configured.
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

        TorchMetrics checks ``jit_distributed_available()`` by default. After DDP
        training, the process group can still be alive while a separate test trainer is
        intentionally single-device. In that case metric states must not all-gather.

        Args:
            metrics: The metric collection to configure.
        """
        for metric in metrics.values(copy_state=False):
            metric.distributed_available_fn = self._distributed_available_fn

    def _distributed_available_fn(self) -> bool:
        """
        Return whether metrics should synchronize for the current trainer.

        This is needed as ``distributed_available_fn`` defaults to a check of
        ``torch.distributed.is_available()`` and ``torch.distributed.is_initialized()``.
        In our case, we can have a single-device test trainer after DDP training,
        so we want to disable metric synchronization if the trainer is not multi-process, not
        only if ``torch.distributed`` is not initialized.
        The issue is that without ``trainer.world_size > 1``, the single-device trainer
        tries to sync across DDP process group and hangs because it's the only one in that group.

        Returns:
            True when the attached trainer is multi-process and torch.distributed is initialized.
        """
        trainer = self._trainer
        if trainer is None:
            return False
        return (
            trainer.world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

    def _should_sample_negatives(self) -> bool:
        """
        Whether to resample negatives for the current epoch.
        """
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
            negatives: A batch of negative samples, either freshly sampled or from cache.
        """
        if self.__negative_sampling_scheduler is None:
            raise ValueError("Asked to sample negatives but no negative sampler is not configured.")
        return self.__negative_sampling_scheduler.sample(batch, self.current_epoch)


def stage_metric_name(stage: Stage, metric_name: str) -> str:
    """
    Build a metric name with its stage prefix.

    Args:
        stage: Metric stage.
        metric_name: Metric name without a stage prefix.

    Returns:
        metric_name: Stage-prefixed metric name.
    """
    return f"{stage_metric_prefix(stage)}{metric_name}"


def stage_metric_prefix(stage: Stage) -> str:
    """
    Build the metric prefix for a stage.

    Args:
        stage: Metric stage.

    Returns:
        prefix: Stage prefix ending with ``"/"``.
    """
    return f"{stage.value}/"
