from typing import Any, Literal, TypedDict
from torch import Tensor, nn, optim
from torchmetrics import MetricCollection
from typing_extensions import NotRequired
from hypertorch.models import HNHN, SLP
from hypertorch.nn import HyperedgeAggregator
from hypertorch.types import HData
from hypertorch.utils import Stage

from hypertorch.hlp.common import HLPPredictor


class HNHNEncoderConfig(TypedDict):
    """
    Configuration for the HNHN encoder in HNHNPredictor.

    Attributes:
        in_channels: Number of input features per node.
        hidden_channels: Number of hidden units in the intermediate HNHN layer.
        out_channels: Number of output features (embedding size) per node.
        bias: Whether to include bias terms. Defaults to ``True``.
        use_batch_normalization: Whether to use batch normalization. Defaults to ``False``.
        drop_rate: Dropout rate. Defaults to ``0.5``.
    """

    in_channels: int
    hidden_channels: int
    out_channels: int
    bias: NotRequired[bool]
    use_batch_normalization: NotRequired[bool]
    drop_rate: NotRequired[float]


class HNHNPredictor(HLPPredictor):
    """
    A LightningModule for HNHN-based HLP predictor.

    Uses HNHN as an encoder to produce node embeddings through explicit
    hyperedge neurons, aggregates them per hyperedge, and scores each
    hyperedge with a linear decoder.

    Attributes:
        encoder: HNHN encoder module inherited from ``HLPPredictor``.
        decoder: SLP decoder module inherited from ``HLPPredictor``.
        loss_fn: Loss function inherited from ``HLPPredictor``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``HLPPredictor``.
        train_metrics: Optional training metrics inherited from ``HLPPredictor``.
        val_metrics: Optional validation metrics inherited from ``HLPPredictor``.
        test_metrics: Optional test metrics inherited from ``HLPPredictor``.
        aggregation: Method to aggregate node embeddings per hyperedge. Defaults to ``"mean"``.
        lr: Learning rate for the optimizer. Defaults to ``0.01``.
        weight_decay: L2 regularization. Defaults to ``5e-4``.
        scheduler_step_size: Step size for learning rate scheduler. Defaults to ``100``.
        scheduler_gamma: Multiplicative factor for learning rate decay. Defaults to ``0.51``.
    """

    def __init__(
        self,
        encoder_config: HNHNEncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = "mean",
        loss_fn: nn.Module | None = None,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        scheduler_step_size: int = 100,
        scheduler_gamma: float = 0.51,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the HNHN-based HLP predictor.

        Args:
            encoder_config: Configuration for the HNHN encoder.
            aggregation: Method used to aggregate node embeddings per hyperedge.
                Defaults to ``"mean"``.
            loss_fn: Optional loss function. Defaults to ``BCEWithLogitsLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.01``.
            weight_decay: L2 regularization. Defaults to ``5e-4``.
            scheduler_step_size: Step size for the learning rate scheduler. Defaults to ``100``.
            scheduler_gamma: Multiplicative factor for learning rate decay. Defaults to ``0.51``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior
                of ``torchmetrics``. Defaults to ``None``.
        """
        encoder = HNHN(
            in_channels=encoder_config["in_channels"],
            hidden_channels=encoder_config["hidden_channels"],
            num_classes=encoder_config["out_channels"],
            bias=encoder_config.get("bias", True),
            use_batch_normalization=encoder_config.get("use_batch_normalization", False),
            drop_rate=encoder_config.get("drop_rate", 0.5),
        )
        decoder = SLP(in_channels=encoder_config["out_channels"], out_channels=1)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss_fn=loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.aggregation: Literal["mean", "max", "min", "sum"] = aggregation
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.scheduler_step_size: int = scheduler_step_size
        self.scheduler_gamma: float = scheduler_gamma

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Run the full HNHN-based hyperedge link prediction pipeline.

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge connectivity of shape ``(2, num_incidences)``.

        Returns:
            scores: Logit scores of shape ``(num_hyperedges,)``.

        Raises:
            ValueError: If the encoder is not defined for this module.
        """
        if self.encoder is None:
            raise ValueError("Encoder is not defined for this HLP module.")

        node_embeddings: Tensor = self.encoder(x, hyperedge_index)
        hyperedge_embeddings = HyperedgeAggregator(hyperedge_index, node_embeddings).pool(
            self.aggregation
        )
        scores: Tensor = self.decoder(hyperedge_embeddings).squeeze(-1)
        return scores

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

    def configure_optimizers(self) -> tuple[list[optim.Adam], list[optim.lr_scheduler.StepLR]]:
        """
        Configure the optimizer and scheduler.

        Returns:
            optimizers_and_schedulers: Optimizer and scheduler lists.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]

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
        target_scores, target_labels = self._target_scores_and_labels(scores, batch)
        batch_size = target_labels.size(0)

        loss = self._compute_loss(target_scores, target_labels, batch_size, stage)
        self._compute_metrics(target_scores, target_labels, batch_size, stage)
        return loss
