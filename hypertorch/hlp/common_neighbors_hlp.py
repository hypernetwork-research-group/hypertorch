import torch
import warnings

from torch import Tensor, nn
from typing import Any, Literal
from torchmetrics import MetricCollection
from hypertorch.models import CommonNeighbors
from hypertorch.types import HData, Hypergraph
from hypertorch.utils import Stage

from hypertorch.hlp.common import HlpModule


class CommonNeighborsHlpModule(HlpModule):
    """
    A LightningModule for the CommonNeighbors model with optional negative sampling.

    Attributes:
        encoder: Optional encoder module inherited from ``HlpModule``. Defaults to ``None``.
        decoder: Common-neighbor decoder inherited from ``HlpModule``.
        loss_fn: Loss function inherited from ``HlpModule``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``HlpModule``.
        train_metrics: Optional training metrics inherited from ``HlpModule``.
        val_metrics: Optional validation metrics inherited from ``HlpModule``.
        test_metrics: Optional test metrics inherited from ``HlpModule``.
        node_to_neighbors: Precomputed training-world node neighborhoods.
        automatic_optimization: Disabled because this module has no trainable optimization.
    """

    def __init__(
        self,
        train_hyperedge_index: Tensor,
        aggregation: Literal["mean", "min", "sum"] = "mean",
        decoder: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the CommonNeighbors HLP module.

        Args:
            train_hyperedge_index: Training hyperedge index used to precompute neighborhoods.
            aggregation: Common-neighbor aggregation method. Defaults to ``"mean"``.
            decoder: Optional decoder module. Defaults to ``CommonNeighbors``.
            loss_fn: Optional loss function. Defaults to ``BCEWithLogitsLoss``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Defaults to ``None``.
        """
        super().__init__(
            decoder=decoder if decoder is not None else CommonNeighbors(aggregation),
            loss_fn=loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        # Pre-compute neighbors of training nodes based on training edges only
        # to create a "known world" for the model to make predictions from
        self.node_to_neighbors: dict[int, set[int]] = Hypergraph.from_hyperedge_index(
            train_hyperedge_index
        ).neighbors_of_all()

        # Disable automatic optimization since there is no training
        self.automatic_optimization: bool = False

    def forward(self, hyperedge_index: Tensor) -> Tensor:
        """
        Compute common neighbor scores for the given hyperedges.

        Args:
            hyperedge_index: Tensor containing incidence information for the hyperedges to score.
        """
        return self.decoder(hyperedge_index, self.node_to_neighbors)

    def on_fit_start(self) -> None:
        """
        Warn users if they are running unnecessary training epochs.
        """
        if self.trainer.max_epochs is None or self.trainer.max_epochs > 0:
            warnings.warn(
                f"{self.__class__.__name__} is a non-trainable heuristic model. "
                "No optimization occurs. Set max_epochs=0 in your trainer for instant evaluation.",
                UserWarning,
                stacklevel=2,
            )

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Return a zero loss for the non-trainable heuristic.

        Args:
            batch: Input batch, unused.
            batch_idx: Batch index, unused.

        Returns:
            loss: Zero scalar tensor on the module device.
        """
        return torch.tensor(0.0, dtype=torch.float, device=self.device)

    def validation_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Return a zero validation loss for the non-trainable heuristic.

        Args:
            batch: Input batch, unused.
            batch_idx: Batch index, unused.

        Returns:
            loss: Zero scalar tensor on the module device.
        """
        return torch.tensor(0.0, dtype=torch.float, device=self.device)

    def test_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Evaluate a test batch.

        Args:
            batch: Test batch.
            batch_idx: Batch index, unused.

        Returns:
            loss: Test loss.
        """
        return self.__step(batch, stage=Stage.TEST)

    def predict_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Predict common-neighbor scores for a batch.

        Args:
            batch: Prediction batch.
            batch_idx: Batch index, unused.

        Returns:
            scores: Predicted hyperedge scores.
        """
        return self.forward(batch.hyperedge_index)

    def configure_optimizers(self) -> None:
        """
        Return no optimizer for the non-trainable heuristic.

        Returns:
            optimizer: No optimizer is configured, so always returns ``None``.
        """
        # No training, so no optimizers needed

    def __step(self, batch: HData, stage: Stage) -> Tensor:
        """
        Shared evaluation logic for all stages.

        Args:
            batch: `HData` object containing the hypergraph.
            stage: The current stage of evaluation
                (e.g., ``Stage.TRAIN``, ``Stage.VAL``, ``Stage.TEST``).

        Returns:
            loss: The computed loss.
        """
        scores = self.forward(batch.hyperedge_index)
        labels = batch.y

        # We need to use the number of hyperedges as batch size for logging purposes,
        # since each hyperedge is a separate prediction
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)

        return loss
