from typing import Literal, Optional, TypedDict

from torch import Tensor, nn, optim
from torchmetrics import MetricCollection
from typing_extensions import NotRequired

from hyperbench.hlp.hlp import HlpModule
from hyperbench.models import HNHN, SLP
from hyperbench.nn import HyperedgeAggregator
from hyperbench.types import HData
from hyperbench.utils import Stage


class HNHNEncoderConfig(TypedDict):
    """
    Configuration for the HNHN encoder in HNHNHlpModule.

    Args:
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


class HNHNHlpModule(HlpModule):
    """
    A LightningModule for HNHN-based Hyperedge Link Prediction.

    Uses HNHN as an encoder to produce node embeddings through explicit
    hyperedge neurons, aggregates them per hyperedge, and scores each
    hyperedge with a linear decoder.

    Args:
        encoder_config: Configuration for the HNHN encoder.
        aggregation: Method to aggregate node embeddings per hyperedge. Defaults to ``"mean"``.
        loss_fn: Loss function. Defaults to ``BCEWithLogitsLoss``.
        lr: Learning rate for the optimizer. Defaults to ``0.01``.
        weight_decay: L2 regularization. Defaults to ``5e-4``.
        scheduler_step_size: Step size for learning rate scheduler. Defaults to ``100``.
        scheduler_gamma: Multiplicative factor for learning rate decay. Defaults to ``0.51``.
        metrics: Optional metric collection for evaluation.
    """

    def __init__(
        self,
        encoder_config: HNHNEncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = "mean",
        loss_fn: Optional[nn.Module] = None,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        scheduler_step_size: int = 100,
        scheduler_gamma: float = 0.51,
        metrics: Optional[MetricCollection] = None,
    ):
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
        )

        self.aggregation = aggregation
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Run the full HNHN-based hyperedge link prediction pipeline.

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge connectivity of shape ``(2, num_incidences)``.

        Returns:
            Logit scores of shape ``(num_hyperedges,)``.
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
        return self.__eval_step(batch, Stage.TRAIN)

    def validation_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.VAL)

    def test_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.TEST)

    def predict_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.forward(batch.x, batch.hyperedge_index)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]

    def __eval_step(self, batch: HData, stage: Stage) -> Tensor:
        scores = self.forward(batch.x, batch.hyperedge_index)
        labels = batch.y
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)
        return loss
