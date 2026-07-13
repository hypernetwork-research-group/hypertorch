from torch import Tensor, nn, optim
from typing import Any, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hypertorch.models import GCN
from hypertorch.types import (
    EdgeIndex,
    GraphReductionStrategy,
    GraphReductionStrategyEnum,
    HData,
    HyperedgeIndex,
)
from hypertorch.utils import ActivationFn, Stage

from hypertorch.nc.common import NcModule


class GCNClassifierConfig(TypedDict):
    """
    Configuration for the GCN classifier in ``GCNNcModule``.

    Attributes:
        in_channels: Number of input features per node.
        out_channels: Number of node classes.
        hidden_channels: Number of hidden units in intermediate GCN layers.
        num_layers: Number of GCN layers. Defaults to ``2``.
        drop_rate: Dropout rate applied after each hidden GCN layer. Defaults to ``0.0``.
        bias: Whether to include bias terms. Defaults to ``True``.
        improved: Whether to use the improved GCN normalization. Defaults to ``False``.
        add_self_loops: Whether to add self-loops before convolution. Defaults to ``True``.
        normalize: Whether to normalize the adjacency matrix in ``GCNConv``. Defaults to ``True``.
        cached: Whether to cache the normalized graph in ``GCNConv``. Defaults to ``False``.
        graph_reduction_strategy: Strategy for reducing the hypergraph to a graph.
            Defaults to ``"clique_expansion"``.
        num_nodes: Optional total number of nodes to use during graph reduction.
        activation_fn: Activation function to use after each hidden layer.
            Defaults to ``nn.ReLU``.
        activation_fn_kwargs: Keyword arguments for the activation function.
            Defaults to empty dict.
    """

    in_channels: int
    out_channels: int
    hidden_channels: NotRequired[int]
    num_layers: NotRequired[int]
    drop_rate: NotRequired[float]
    bias: NotRequired[bool]
    improved: NotRequired[bool]
    add_self_loops: NotRequired[bool]
    normalize: NotRequired[bool]
    cached: NotRequired[bool]
    graph_reduction_strategy: NotRequired[GraphReductionStrategy]
    num_nodes: NotRequired[int]
    activation_fn: NotRequired[ActivationFn]
    activation_fn_kwargs: NotRequired[dict]


class GCNNcModule(NcModule):
    """
    A LightningModule for GCN-based multiclass node classification.

    Reduces the hypergraph to a graph, then applies GCN
    layers to produce per-node class logits.

    Attributes:
        encoder: Optional encoder module inherited from ``NcModule``. Defaults to ``None``.
        classifier: GCN classifier module inherited from ``NcModule``.
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
        classifier_config: GCNClassifierConfig,
        loss_fn: nn.Module | None = None,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the GCN NC module.

        Args:
            classifier_config: Configuration for the GCN classifier.
            loss_fn: Optional loss function. Defaults to ``CrossEntropyLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.01``.
            weight_decay: L2 regularization. Defaults to ``5e-4``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior
                of ``torchmetrics``. Defaults to ``None``.
        """
        classifier = GCN(
            in_channels=classifier_config["in_channels"],
            out_channels=classifier_config["out_channels"],
            hidden_channels=classifier_config.get("hidden_channels"),
            num_layers=classifier_config.get("num_layers", 2),
            drop_rate=classifier_config.get("drop_rate", 0.0),
            bias=classifier_config.get("bias", True),
            activation_fn=classifier_config.get("activation_fn"),
            activation_fn_kwargs=classifier_config.get("activation_fn_kwargs"),
            improved=classifier_config.get("improved", False),
            add_self_loops=classifier_config.get("add_self_loops", True),
            normalize=classifier_config.get("normalize", True),
            cached=classifier_config.get("cached", False),
        )

        super().__init__(
            classifier=classifier,
            loss_fn=loss_fn if loss_fn is not None else nn.CrossEntropyLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.classifier_config: GCNClassifierConfig = classifier_config
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
        # Reduce hypergraph to graph and remove self-loops
        reduced_edge_index = HyperedgeIndex(hyperedge_index).reduce(
            strategy=self.classifier_config.get(
                "graph_reduction_strategy",
                GraphReductionStrategyEnum.CLIQUE_EXPANSION,
            ),
            num_nodes=self.classifier_config.get("num_nodes"),
        )
        edge_index = EdgeIndex(reduced_edge_index).remove_selfloops().item

        return self.classifier(x, edge_index)

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
