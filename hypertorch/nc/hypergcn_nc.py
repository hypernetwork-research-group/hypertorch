from torch import Tensor, nn, optim
from typing import Any, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hypertorch.models import HyperGCN
from hypertorch.types import HData
from hypertorch.utils import Stage

from hypertorch.nc.common import NcModule


class HyperGCNClassifierConfig(TypedDict):
    """
    Configuration for the HyperGCN classifier in ``HyperGCNNcModule``.

    Attributes:
        in_channels: Number of input features per node.
        hidden_channels: Number of hidden units in the intermediate HyperGCN layer.
        out_channels: Number of node classes.
        bias: Whether to include bias terms. Defaults to ``True``.
        use_batch_normalization: Whether to use batch normalization. Defaults to ``False``.
        drop_rate: Dropout rate. Defaults to ``0.5``.
        use_mediator: Whether to use mediator nodes for hyperedge-to-edge conversion.
            Defaults to ``False``.
        fast: Whether to cache the graph structure after first computation.
            Defaults to ``True``.
        seed: Optional random seed for the random reduction of hyperedges to edges.
            Defaults to ``None``.
    """

    in_channels: int
    hidden_channels: int
    out_channels: int
    bias: NotRequired[bool]
    use_batch_normalization: NotRequired[bool]
    drop_rate: NotRequired[float]
    use_mediator: NotRequired[bool]
    fast: NotRequired[bool]
    seed: NotRequired[int]


class HyperGCNNcModule(NcModule):
    """
    A LightningModule for HyperGCN-based multiclass node classification.

    Uses HyperGCN to transform node features and hypergraph structure directly into
    per-node class logits. During training, validation, and testing, loss and metrics
    are computed on supervised target nodes selected by ``HData.target_node_mask``.

    Attributes:
        encoder: Optional encoder module inherited from ``NcModule``. Defaults to ``None``.
        classifier: HyperGCN classifier module inherited from ``NcModule``.
        loss_fn: Loss function inherited from ``NcModule``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``NcModule``.
        train_metrics: Optional training metrics inherited from ``NcModule``.
        val_metrics: Optional validation metrics inherited from ``NcModule``.
        test_metrics: Optional test metrics inherited from ``NcModule``.
        lr: Learning rate for the optimizer. Defaults to ``0.01``.
        weight_decay: L2 regularization. Defaults to ``5e-4``.
    """

    def __init__(
        self,
        classifier_config: HyperGCNClassifierConfig,
        loss_fn: nn.Module | None = None,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the HyperGCN NC module.

        Args:
            classifier_config: Configuration for the HyperGCN classifier.
            loss_fn: Optional loss function. Defaults to ``CrossEntropyLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.01``.
            weight_decay: L2 regularization. Defaults to ``5e-4``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior
                of ``torchmetrics``. Defaults to ``None``.
        """
        classifier = HyperGCN(
            in_channels=classifier_config["in_channels"],
            hidden_channels=classifier_config["hidden_channels"],
            num_classes=classifier_config["out_channels"],
            bias=classifier_config.get("bias", True),
            use_batch_normalization=classifier_config.get("use_batch_normalization", False),
            drop_rate=classifier_config.get("drop_rate", 0.5),
            use_mediator=classifier_config.get("use_mediator", False),
            fast=classifier_config.get("fast", True),
            seed=classifier_config.get("seed"),
        )

        super().__init__(
            classifier=classifier,
            loss_fn=loss_fn if loss_fn is not None else nn.CrossEntropyLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.lr: float = lr
        self.weight_decay: float = weight_decay

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Predict node-class logits from node features and hypergraph structure.

        Examples:
            Given 4 nodes with 3 features each and 2 output classes:
                >>> x.shape
                ... torch.Size([4, 3])
                >>> hyperedge_index.shape
                ... torch.Size([2, 6])

            The forward pass maps each node to one row of class logits:
                >>> logits = model.forward(x, hyperedge_index)
                >>> logits.shape
                ... torch.Size([4, 2])

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge connectivity of shape ``(2, num_incidences)``.

        Returns:
            logits: Node-class logits of shape ``(num_nodes, num_classes)``.
        """
        return self.classifier(x, hyperedge_index)

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Run a training step.

        Args:
            batch: Training batch.
            batch_idx: Batch index, unused.

        Returns:
            loss: Training loss.
        """
        return self.__eval_step(batch, Stage.TRAIN)

    def validation_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Run a validation step.

        Args:
            batch: Validation batch.
            batch_idx: Batch index, unused.

        Returns:
            loss: Validation loss.
        """
        return self.__eval_step(batch, Stage.VAL)

    def test_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Run a test step.

        Args:
            batch: Test batch.
            batch_idx: Batch index, unused.

        Returns:
            loss: Test loss.
        """
        return self.__eval_step(batch, Stage.TEST)

    def predict_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Predict node-class logits for a batch.

        Args:
            batch: Prediction batch.
            batch_idx: Batch index, unused.

        Returns:
            logits: Predicted node-class logits.
        """
        return self.forward(batch.x, batch.hyperedge_index)

    def configure_optimizers(self) -> optim.Adam:
        """
        Configure the optimizer.

        Returns:
            optimizer: Adam optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def __eval_step(self, batch: HData, stage: Stage) -> Tensor:
        """
        Run shared evaluation logic for a stage.

        Args:
            batch: Input batch.
            stage: Current evaluation stage.

        Returns:
            loss: Computed loss.
        """
        logits = self.forward(batch.x, batch.hyperedge_index)
        target_logits, target_labels = self._target_logits_and_labels(logits, batch)
        batch_size = int(target_labels.size(0))

        loss = self._compute_loss(target_logits, target_labels, batch_size, stage)
        self._compute_metrics(target_logits, target_labels, batch_size, stage)
        return loss
