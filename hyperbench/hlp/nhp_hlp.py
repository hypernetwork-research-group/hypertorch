from torch import Tensor, nn, optim
from typing import Literal, TypedDict
from torchmetrics import MetricCollection
from typing_extensions import NotRequired
from hyperbench.models import NHP
from hyperbench.nn import NHPRankingLoss
from hyperbench.types import HData
from hyperbench.utils import ActivationFn, Stage

from hyperbench.hlp.common import HlpModule


class NHPEncoderConfig(TypedDict):
    """
    Configuration for the NHP encoder/scorer to be used for hyperedge link prediction.

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Number of hidden channels for incidence embeddings. Defaults to ``512``.
        aggregation: Hyperedge scoring aggregation. ``"maxmin"`` uses the paper's
            element-wise range representation; ``"mean"`` uses mean pooling.
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

    Args:
        encoder_config: Configuration for the NHP encoder/scorer.
        loss_fn: Loss function. Defaults to :class:`NHPRankingLoss`.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
        weight_decay: L2 regularization. Defaults to ``5e-4``.
        metrics: Optional metric collection for evaluation.
    """

    def __init__(
        self,
        encoder_config: NHPEncoderConfig,
        loss_fn: nn.Module | None = None,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        metrics: MetricCollection | None = None,
    ):
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
        )

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Encode and score each candidate hyperedge.

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge connectivity of shape ``(2, num_incidences)``.

        Returns:
            Scores of shape ``(num_hyperedges,)``.
        """
        if self.encoder is None:
            raise ValueError("Encoder is not defined for this HLP module.")
        return self.encoder(x, hyperedge_index)

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.TRAIN)

    def validation_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.VAL)

    def test_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.TEST)

    def predict_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.forward(batch.x, batch.hyperedge_index)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def __eval_step(self, batch: HData, stage: Stage) -> Tensor:
        scores = self.forward(batch.x, batch.hyperedge_index)
        labels = batch.y
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)
        return loss
