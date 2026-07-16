import torch
import warnings

from torch import Tensor, nn
from typing import Any, Literal
from torchmetrics import MetricCollection
from hypertorch.models import CommonNeighbors
from hypertorch.nn import CommonNeighborsNodeScorer
from hypertorch.types import HData, HyperedgeIndex, Hypergraph, Neighborhood
from hypertorch.utils import Stage

from hypertorch.nc.common import NCClassifier


class CommonNeighborsClassifier(NCClassifier):
    """
    A LightningModule for the CommonNeighbors-based NC classifier.

    Attributes:
        encoder: Optional encoder module inherited from ``NCClassifier``. Defaults to ``None``.
        classifier: Identity placeholder inherited from ``NCClassifier``.
        loss_fn: Loss function inherited from ``NCClassifier``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``NCClassifier``.
        train_metrics: Optional training metrics inherited from ``NCClassifier``.
        val_metrics: Optional validation metrics inherited from ``NCClassifier``.
        test_metrics: Optional test metrics inherited from ``NCClassifier``.
        node_to_neighbors: Precomputed training-world node neighborhoods.
        automatic_optimization: Disabled because this module has no trainable optimization.
    """

    def __init__(
        self,
        train_hdata: HData,
        num_classes: int,
        aggregation: Literal["mean", "min", "sum"] = "sum",
        exclude_self_reference: bool = True,
        classifier: nn.Module | None = None,
        loss_fn: nn.Module | None = None,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the CommonNeighbors-based NC classifier.

        Args:
            train_hdata: Training data used to precompute neighborhoods and labeled
                reference nodes.
            num_classes: Number of node classes.
            aggregation: Method used to aggregate reference-node scores per class.
                Defaults to ``"sum"``.
            exclude_self_reference: Whether a node should ignore itself when it also
                appears among labeled reference nodes. Defaults to ``True``.
            classifier: Optional classifier module.
            loss_fn: Optional loss function. Defaults to ``CrossEntropyLoss``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Defaults to ``None``.
        """
        super().__init__(
            classifier=classifier
            if classifier is not None
            else self.__default_classifier(
                train_hdata=train_hdata,
                num_classes=num_classes,
                aggregation=aggregation,
                exclude_self_reference=exclude_self_reference,
            ),
            loss_fn=loss_fn if loss_fn is not None else nn.CrossEntropyLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        # Pre-compute neighbors of training nodes based on training edges only
        # to create a "known world" for the model to make predictions from
        self.node_to_neighbors: dict[int, Neighborhood] = Hypergraph.from_hyperedge_index(
            hyperedge_index=HyperedgeIndex(
                hyperedge_index=train_hdata.hyperedge_index,
            )
            .to_global(global_node_ids=train_hdata.global_node_ids)
            .item
        ).neighbors_of_all()

        self.automatic_optimization: bool = False

    def forward(self, global_node_ids: Tensor) -> Tensor:
        """
        Compute common-neighbor class scores for the given nodes.

        Args:
            global_node_ids: Stable node IDs matching the training-world node IDs.

        Returns:
            logits: Node-class scores of shape ``(num_nodes, num_classes)``.
        """
        return self.classifier(
            candidate_nodes=global_node_ids,
            node_to_neighbors=self.node_to_neighbors,
        )

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
        Predict common-neighbor class scores for a batch.

        Args:
            batch: Prediction batch.
            batch_idx: Batch index, unused.

        Returns:
            logits: Predicted node-class scores.
        """
        return self.forward(batch.global_node_ids)

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
            stage: Current evaluation stage.

        Returns:
            loss: Computed loss.
        """
        scores = self.forward(batch.global_node_ids)
        target_logits, target_labels = self._target_logits_and_labels(scores, batch)
        batch_size = int(target_labels.size(0))

        loss = self._compute_loss(target_logits, target_labels, batch_size, stage)
        self._compute_metrics(target_logits, target_labels, batch_size, stage)

        return loss

    def __class_to_node_ids(self, train_hdata: HData, num_classes: int) -> dict[int, list[int]]:
        """
        Group labeled training nodes by class.

        Args:
            train_hdata: Training data containing node labels and train node mask.
            num_classes: Number of node classes.

        Returns:
            class_to_node_ids: Dictionary mapping class IDs to lists of node IDs.
        """
        class_to_node_ids: dict[int, list[int]] = {class_id: [] for class_id in range(num_classes)}
        train_node_ids = train_hdata.global_node_ids[train_hdata.target_node_mask].tolist()
        train_class_labels = train_hdata.y[train_hdata.target_node_mask].tolist()

        for node_id, class_label in zip(train_node_ids, train_class_labels, strict=True):
            class_to_node_ids[int(class_label)].append(int(node_id))

        return class_to_node_ids

    def __default_classifier(
        self,
        train_hdata: HData,
        num_classes: int,
        aggregation: Literal["mean", "min", "sum"],
        exclude_self_reference: bool = True,
    ) -> CommonNeighbors:
        """
        Create a default CommonNeighbors classifier with a CommonNeighborsNodeScorer.

        Args:
            train_hdata: Training data containing node labels and train node mask.
            num_classes: Number of node classes.
            aggregation: Method used to aggregate reference-node scores per class.
            exclude_self_reference: Whether a node should ignore itself when it also
                appears among labeled reference nodes.

        Returns:
            classifier: A CommonNeighbors classifier with a CommonNeighborsNodeScorer.
        """
        class_to_node_ids = self.__class_to_node_ids(
            train_hdata=train_hdata,
            num_classes=num_classes,
        )

        scorer = CommonNeighborsNodeScorer(
            num_classes=num_classes,
            class_to_node_ids=class_to_node_ids,
            aggregation=aggregation,
            exclude_self_reference=exclude_self_reference,
        )

        return CommonNeighbors(aggregation=aggregation, scorer=scorer)
