import torch
import logging as log

from torch import Tensor, nn
from typing import Dict, Optional
from hyperbench.models import CommonNeighbors
from hyperbench.nn import Aggregation
from hyperbench.types import HData, Hypergraph

from .hlp import HlpModule, MetricFn, Stage


class CommonNeighborsHlpModule(HlpModule):
    """
    A LightningModule for the CommonNeighbors model with optional negative sampling.

    Args:
        aggregation: The aggregation method for common neighbors ("mean", "min", or "sum").
        decoder: An optional decoder module. Defaults to :class:`CommonNeighbors`.
        loss_fn: An optional loss function. Defaults to ``BCEWithLogitsLoss``.
    """

    # No train/validation, so return default loss
    __DEFAULT_LOSS = torch.tensor(0.0)

    def __init__(
        self,
        train_hyperedge_index: Tensor,
        aggregation: Optional[Aggregation] = None,
        decoder: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, MetricFn]] = None,
    ):
        super().__init__(
            decoder=decoder if decoder is not None else CommonNeighbors(aggregation=aggregation),
            loss_fn=loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics,
        )

        # Pre-compute neighbors of training nodes based on training edges only
        # to create a "known world" for the model to make predictions from
        self.node_to_neighbors = Hypergraph.from_hyperedge_index(
            train_hyperedge_index
        ).neighbors_of_all()

        # Disable automatic optimization since there is no training
        self.automatic_optimization = False

    def on_fit_start(self) -> None:
        """Warn users if they are running unnecessary training epochs."""
        if self.trainer.max_epochs is None or self.trainer.max_epochs > 0:
            log.warning(
                f"{self.__class__.__name__} is a non-trainable heuristic model. "
                "No optimization occurs. Set max_epochs=0 in your trainer for instant evaluation."
            )

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__DEFAULT_LOSS

    def validation_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__DEFAULT_LOSS

    def test_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__step(batch, stage=Stage.TEST)

    def predict_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.decoder(batch.hyperedge_index, self.node_to_neighbors)

    def configure_optimizers(self):
        # No training, so no optimizers needed
        return None

    def __step(self, batch: HData, stage: Stage) -> Tensor:
        """
        Shared evaluation logic for all stages.

        Args:
            batch: :class:`HData` object containing the hypergraph.
            stage: The current stage of evaluation (e.g., ``Stage.TRAIN``, ``Stage.VAL``, ``Stage.TEST``).

        Returns:
            The computed loss.
        """
        scores: Tensor = self.decoder(batch.hyperedge_index, self.node_to_neighbors)
        labels = batch.y
        batch_size = batch.num_nodes

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)

        return loss
