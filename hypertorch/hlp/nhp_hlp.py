from torch import Tensor, nn, optim
from typing import Any, Literal, TypedDict
from torchmetrics import MetricCollection
from typing_extensions import NotRequired
from hypertorch.models import NHP
from hypertorch.nn import NHPRankingLoss
from hypertorch.types import HData
from hypertorch.utils import ActivationFn, Stage

from hypertorch.hlp.common import HlpModule


class NHPEncoderConfig(TypedDict):
    """
    Configuration for the NHP encoder/scorer to be used for hyperedge link prediction.

    Attributes:
        in_channels: Number of input features per node.
        hidden_channels: Number of hidden channels for incidence embeddings. Defaults to ``512``.
        activation_fn: Optional activation function. Defaults to ``None``.
        activation_fn_kwargs: Keyword arguments for the activation function. Defaults to ``None``.
        aggregation: Hyperedge scoring aggregation. ``"maxmin"`` uses the paper's
            element-wise range representation; ``"mean"`` uses mean pooling.
            Defaults to ``"maxmin"``.
        bias: Whether to include bias terms. Defaults to ``True``.
    """

    in_channels: int
    hidden_channels: NotRequired[int]
    activation_fn: NotRequired[ActivationFn | None]
    activation_fn_kwargs: NotRequired[dict | None]
    aggregation: NotRequired[Literal["mean", "maxmin"]]
    bias: NotRequired[bool]


class NHPHlpModule(HlpModule):
    """
    A LightningModule for undirected NHP hyperedge link prediction.

    NHP encodes and scores candidate hyperedges in a single pass.
    Unlike encoder wrappers that produce reusable global node embeddings,
    NHP builds candidate-specific incidence embeddings before pooling and scoring each hyperedge.

    Attributes:
        encoder: NHP scorer inherited from ``HlpModule``.
        decoder: Identity decoder inherited from ``HlpModule``.
        loss_fn: Loss function inherited from ``HlpModule``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``HlpModule``.
        train_metrics: Optional training metrics inherited from ``HlpModule``.
        val_metrics: Optional validation metrics inherited from ``HlpModule``.
        test_metrics: Optional test metrics inherited from ``HlpModule``.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
        weight_decay: L2 regularization. Defaults to ``5e-4``.
    """

    def __init__(
        self,
        encoder_config: NHPEncoderConfig,
        loss_fn: nn.Module | None = None,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the NHP HLP module.

        Args:
            encoder_config: Configuration for the NHP encoder/scorer.
            loss_fn: Optional loss function. Defaults to ``NHPRankingLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.001``.
            weight_decay: L2 regularization. Defaults to ``5e-4``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior
                of ``torchmetrics``. Defaults to ``None``.
        """
        encoder = NHP(
            in_channels=encoder_config["in_channels"],
            hidden_channels=encoder_config.get("hidden_channels", 512),
            activation_fn=encoder_config.get("activation_fn"),
            activation_fn_kwargs=encoder_config.get("activation_fn_kwargs"),
            aggregation=encoder_config.get("aggregation", "maxmin"),
            bias=encoder_config.get("bias", True),
        )

        super().__init__(
            encoder=encoder,
            decoder=nn.Identity(),
            loss_fn=loss_fn if loss_fn is not None else NHPRankingLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.lr: float = lr
        self.weight_decay: float = weight_decay

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Encode and score each candidate hyperedge.

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge connectivity of shape ``(2, num_incidences)``.

        Returns:
            scores: Scores of shape ``(num_hyperedges,)``.

        Raises:
            ValueError: If the encoder is not defined for this module.
        """
        if self.encoder is None:
            raise ValueError("Encoder is not defined for this HLP module.")
        return self.encoder(x, hyperedge_index)

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
        Predict hyperedge scores for a batch.

        Args:
            batch: Prediction batch.
            batch_idx: Batch index, unused.

        Returns:
            scores: Predicted hyperedge scores.
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
        scores = self.forward(batch.x, batch.hyperedge_index)
        labels = batch.y
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)
        return loss
