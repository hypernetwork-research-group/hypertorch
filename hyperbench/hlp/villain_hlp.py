from torch import Tensor, nn, optim
from typing import Literal, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hyperbench.models import SLP, VilLain
from hyperbench.nn import HyperedgeAggregator
from hyperbench.types import HData
from hyperbench.utils import Stage

from hyperbench.hlp.common import HlpModule


class VilLainEncoderConfig(TypedDict):
    """
    Configuration for ``VilLainHlpModule``.

    Attributes:
        num_nodes: Total number of trainable nodes.
        embedding_dim: Returned node and hyperedge embedding dimension. Defaults to ``128``.
        labels_per_subspace: Number of virtual labels per subspace. Defaults to ``2``.
        training_steps: Propagation steps used for VilLain loss. Defaults to ``4``.
        generation_steps: Propagation steps averaged by ``forward``. Defaults to ``100``.
        tau: Gumbel-Softmax temperature. Defaults to ``1.0``.
        eps: Numerical stability constant. Defaults to ``1e-10``.
        villain_loss_weight: Weight applied to VilLain self-supervision. Defaults to ``1.0``.

    """

    num_nodes: int
    embedding_dim: NotRequired[int]
    labels_per_subspace: NotRequired[int]
    training_steps: NotRequired[int]
    generation_steps: NotRequired[int]
    tau: NotRequired[float]
    eps: NotRequired[float]
    villain_loss_weight: NotRequired[float]


class VilLainHlpModule(HlpModule):
    """
    Feature-free VilLain Hyperedge Link Prediction module.

    Args:
        encoder_config: Configuration for the VilLain encoder.
        embedding_mode: Whether to return node or hyperedge embeddings from the VilLain encoder.
        aggregation: Aggregation method to pool node embeddings into hyperedge embeddings when ``embedding_mode="node"``.
            Ignored when ``embedding_mode="hyperedge"``. Defaults to ``maxmin``.
        loss_fn: Loss function for the HLP task. Defaults to ``nn.BCEWithLogitsLoss()``.
        lr: Learning rate for the optimizer. Defaults to ``0.01``.
        weight_decay: Weight decay for the optimizer. Defaults to ``0.0``.
        metrics: Metrics to compute during training and evaluation. Defaults to ``None``.

    """

    def __init__(
        self,
        encoder_config: VilLainEncoderConfig,
        embedding_mode: Literal["node", "hyperedge"] = "node",
        aggregation: Literal["mean", "max", "min", "maxmin", "sum"] = "maxmin",
        loss_fn: nn.Module | None = None,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        metrics: MetricCollection | None = None,
    ):
        self.embedding_dim = encoder_config.get("embedding_dim", 128)
        self.aggregation = aggregation
        self.lr = lr
        self.weight_decay = weight_decay
        self.villain_loss_weight = encoder_config.get("villain_loss_weight", 1.0)
        self.embedding_mode = embedding_mode

        encoder = VilLain(
            num_nodes=encoder_config["num_nodes"],
            embedding_dim=self.embedding_dim,
            labels_per_subspace=encoder_config.get("labels_per_subspace", 2),
            training_steps=encoder_config.get("training_steps", 4),
            generation_steps=encoder_config.get("generation_steps", 100),
            tau=encoder_config.get("tau", 1.0),
            eps=encoder_config.get("eps", 1e-10),
        )
        decoder = SLP(in_channels=self.embedding_dim, out_channels=1)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss_fn=loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics,
        )

    def forward(
        self,
        hyperedge_index: Tensor,
        global_node_ids: Tensor | None = None,
        num_hyperedges: int | None = None,
    ) -> Tensor:
        encoder = self.__to_villain_encoder()

        match self.embedding_mode:
            case "hyperedge":
                hyperedge_embeddings = encoder.hyperedge_embeddings(
                    hyperedge_index=hyperedge_index,
                    node_ids=global_node_ids,
                    num_hyperedges=num_hyperedges,
                )
            case _:
                node_embeddings = encoder.node_embeddings(
                    hyperedge_index=hyperedge_index,
                    node_ids=global_node_ids,
                    num_hyperedges=num_hyperedges,
                )
                hyperedge_embeddings = HyperedgeAggregator(
                    hyperedge_index=hyperedge_index,
                    node_embeddings=node_embeddings,
                    num_hyperedges=num_hyperedges,
                ).pool(self.aggregation)

        scores: Tensor = self.decoder(hyperedge_embeddings).squeeze(-1)
        return scores

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        scores = self.forward(
            hyperedge_index=batch.hyperedge_index,
            global_node_ids=batch.global_node_ids,
            num_hyperedges=batch.num_hyperedges,
        )

        labels = batch.y
        batch_size = batch.num_hyperedges

        hlp_loss = self.loss_fn(scores, labels)
        villain_loss, villain_loss_parts = self.__to_villain_encoder().loss(
            hyperedge_index=batch.hyperedge_index,
            node_ids=batch.global_node_ids,
            num_hyperedges=batch.num_hyperedges,
        )
        loss = hlp_loss + (self.villain_loss_weight * villain_loss)

        loss_prefix = Stage.TRAIN.value
        self.log(f"{loss_prefix}_hlp_loss", hlp_loss, prog_bar=True, batch_size=batch_size)
        self.log(
            f"{loss_prefix}_villain_loss",
            villain_loss,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{loss_prefix}_local_loss",
            villain_loss_parts["local_loss"],
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            f"{loss_prefix}_global_loss",
            villain_loss_parts["global_loss"],
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(f"{loss_prefix}_loss", loss, prog_bar=True, batch_size=batch_size)

        self._compute_metrics(scores, labels, batch_size, Stage.TRAIN)
        return loss

    def validation_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.VAL)

    def test_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.TEST)

    def predict_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.forward(
            hyperedge_index=batch.hyperedge_index,
            global_node_ids=batch.global_node_ids,
            num_hyperedges=batch.num_hyperedges,
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def __eval_step(self, batch: HData, stage: Stage) -> Tensor:
        scores = self.forward(
            hyperedge_index=batch.hyperedge_index,
            global_node_ids=batch.global_node_ids,
            num_hyperedges=batch.num_hyperedges,
        )
        labels = batch.y
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)
        return loss

    def __to_villain_encoder(self) -> VilLain:
        if self.encoder is None or not isinstance(self.encoder, VilLain):
            raise ValueError("VilLain requires a VilLain encoder, but none was provided.")
        return self.encoder
