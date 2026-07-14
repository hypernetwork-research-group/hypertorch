from torch import Tensor, nn, optim
from typing import Any, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hypertorch.models import MLP
from hypertorch.types import HData
from hypertorch.utils import ActivationFn, NormalizationFn, Stage

from hypertorch.nc.common import NcModule


class MLPClassifierConfig(TypedDict):
    """
    Configuration for the MLP classifier in ``MLPNcModule``.

    Attributes:
        in_channels: Number of input features per node.
        out_channels: Number of node classes.
        num_layers: Number of layers in the MLP classifier.
        hidden_channels: Optional number of hidden units per layer. If ``None``, no hidden layers
            are used and the classifier is a simple linear layer.
        activation_fn: Optional activation function class to use in the MLP classifier.
            If ``None``, no activation function is applied.
        activation_fn_kwargs: Optional dictionary of keyword arguments to pass to the activation
            function constructor.
        normalization_fn: Optional normalization function class to use in the MLP classifier.
            If ``None``, no normalization is applied.
        normalization_fn_kwargs: Optional dictionary of keyword arguments to pass to the
            normalization function constructor.
        bias: Whether to include bias terms in the MLP layers. Defaults to ``True``.
        drop_rate: Dropout rate to apply after each MLP layer except the last one.
            Defaults to ``0.0``.
    """

    in_channels: int
    out_channels: int
    num_layers: NotRequired[int]
    hidden_channels: NotRequired[int | None]
    activation_fn: NotRequired[ActivationFn | None]
    activation_fn_kwargs: NotRequired[dict | None]
    normalization_fn: NotRequired[NormalizationFn | None]
    normalization_fn_kwargs: NotRequired[dict | None]
    bias: NotRequired[bool]
    drop_rate: NotRequired[float]


class MLPNcModule(NcModule):
    """
    A LightningModule for MLP-based multiclass node classification.

    Uses an MLP classifier to map node features directly to per-node class logits.
    During training, validation, and testing, loss and metrics are computed on the
    supervised target nodes selected by ``HData.target_node_mask`` when present.

    Attributes:
        encoder: Optional encoder module inherited from ``NcModule``. Defaults to ``None``.
        classifier: MLP classifier module inherited from ``NcModule``.
        loss_fn: Loss function inherited from ``NcModule``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``NcModule``.
        train_metrics: Optional training metrics inherited from ``NcModule``.
        val_metrics: Optional validation metrics inherited from ``NcModule``.
        test_metrics: Optional test metrics inherited from ``NcModule``.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
    """

    def __init__(
        self,
        classifier_config: MLPClassifierConfig,
        loss_fn: nn.Module | None = None,
        lr: float = 0.001,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the MLP NC module.

        Args:
            classifier_config: Configuration for the MLP classifier.
            loss_fn: Optional loss function. Defaults to ``CrossEntropyLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.001``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior
                of ``torchmetrics``. Defaults to ``None``.
        """
        classifier = MLP(
            in_channels=classifier_config["in_channels"],
            hidden_channels=classifier_config.get("hidden_channels"),
            out_channels=classifier_config["out_channels"],
            num_layers=classifier_config.get("num_layers", 1),
            activation_fn=classifier_config.get("activation_fn"),
            activation_fn_kwargs=classifier_config.get("activation_fn_kwargs"),
            normalization_fn=classifier_config.get("normalization_fn"),
            normalization_fn_kwargs=classifier_config.get("normalization_fn_kwargs"),
            bias=classifier_config.get("bias", True),
            drop_rate=classifier_config.get("drop_rate", 0.0),
        )

        super().__init__(
            classifier=classifier,
            loss_fn=loss_fn if loss_fn is not None else nn.CrossEntropyLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.lr: float = lr

    def forward(self, x: Tensor) -> Tensor:
        """
        Predict node-class logits from node features.

        Examples:
            Given 4 nodes with 3 features each and 2 output classes:
                >>> x = [[0.1, 0.2, 0.3],   # node 0
                ...      [0.4, 0.5, 0.6],   # node 1
                ...      [0.7, 0.8, 0.9],   # node 2
                ...      [1.0, 1.1, 1.2]]   # node 3

            The forward pass maps each node to one row of class logits:
                >>> logits = model.forward(x)
                >>> logits.shape
                torch.Size([4, 2])

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.

        Returns:
            logits: Node-class logits of shape ``(num_nodes, num_classes)``.
        """
        return self.classifier(x)

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
        return self.forward(batch.x)

    def configure_optimizers(self) -> optim.Adam:
        """
        Configure the optimizer.

        Returns:
            optimizer: Adam optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.lr)

    def __eval_step(self, batch: HData, stage: Stage) -> Tensor:
        """
        Run shared evaluation logic for a stage.

        Args:
            batch: Input batch.
            stage: Current evaluation stage.

        Returns:
            loss: Computed loss.
        """
        logits = self.forward(batch.x)
        target_logits, target_labels = self._target_logits_and_labels(logits, batch)
        batch_size = int(target_labels.size(0))

        loss = self._compute_loss(target_logits, target_labels, batch_size, stage)
        self._compute_metrics(target_logits, target_labels, batch_size, stage)
        return loss
