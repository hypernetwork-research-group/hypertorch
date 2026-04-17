from torch import Tensor, nn, optim
from typing import Literal, Optional, TypedDict
from typing_extensions import NotRequired
from hyperbench.models import HyperGCN, SLP
from hyperbench.nn import HyperedgeAggregator
from hyperbench.types import HData
from torchmetrics import MetricCollection
from hyperbench.utils import Stage

from hyperbench.hlp.hlp import HlpModule


class HyperGCNEncoderConfig(TypedDict):
    """
    Configuration for the HyperGCN encoder in HyperGCNHlpModule.

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Number of hidden units in the intermediate HyperGCN layer.
        out_channels: Number of output features (embedding size) per node.
        bias: Whether to include bias terms. Defaults to ``True``.
        use_batch_normalization: Whether to use batch normalization. Defaults to ``False``.
        drop_rate: Dropout rate. Defaults to ``0.5``.
        use_mediator: Whether to use mediator nodes for hyperedge-to-edge conversion. Defaults to ``False``.
        fast: Whether to cache the graph structure after first computation. Defaults to ``True``.
    """

    in_channels: int
    hidden_channels: int
    out_channels: int
    bias: NotRequired[bool]
    use_batch_normalization: NotRequired[bool]
    drop_rate: NotRequired[float]
    use_mediator: NotRequired[bool]
    fast: NotRequired[bool]


class HyperGCNHlpModule(HlpModule):
    """
    A LightningModule for HyperGCN-based Hyperedge Link Prediction.

    Uses HyperGCN as an encoder to produce structure-aware node embeddings via
    graph convolution on the full hypergraph, aggregates them per hyperedge,
    and scores each hyperedge with a linear decoder.

    Args:
        encoder_config: Configuration for the HyperGCN encoder.
        aggregation: Method to aggregate node embeddings per hyperedge. Defaults to ``"mean"``.
        loss_fn: Loss function. Defaults to ``BCEWithLogitsLoss``.
        lr: Learning rate for the optimizer. Defaults to ``0.01``.
        weight_decay: L2 regularization. Defaults to ``5e-4``.
        metrics: Optional metric collection for evaluation.
    """

    def __init__(
        self,
        encoder_config: HyperGCNEncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = "mean",
        loss_fn: Optional[nn.Module] = None,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        metrics: Optional[MetricCollection] = None,
    ):
        encoder = HyperGCN(
            in_channels=encoder_config["in_channels"],
            hidden_channels=encoder_config["hidden_channels"],
            num_classes=encoder_config["out_channels"],
            bias=encoder_config.get("bias", True),
            use_batch_normalization=encoder_config.get("use_batch_normalization", False),
            drop_rate=encoder_config.get("drop_rate", 0.5),
            use_mediator=encoder_config.get("use_mediator", False),
            fast=encoder_config.get("fast", True),
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

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Encode node features via HyperGCN, aggregate per hyperedge, and score.

        Steps:
            1. Encode: HyperGCN builds a GCN Laplacian from ``hyperedge_index``
               and applies message passing to produce structure-aware node embeddings.
            2. Aggregate: For each hyperedge, aggregate its member nodes' embeddings
               using the configured pooling method (mean/max/min/sum).
            3. Decode: A linear layer scores each hyperedge embedding.

        Examples:
            Given 5 nodes with 3 features and 2 hyperedges::

                >>> x.shape  # (5, 3) — all nodes in the hypergraph
                >>> hyperedge_index = [[0, 1, 2, 3, 4],  # node IDs (global)
                ...                    [0, 0, 0, 1, 1]]  # hyperedge IDs

            The forward pass:
                1. HyperGCN encodes all 5 nodes using the full graph Laplacian.
                   ``node_embeddings.shape = (5, out_channels)``
                2. Aggregate per hyperedge:
                   - hyperedge 0: pool(emb[0], emb[1], emb[2])
                   - hyperedge 1: pool(emb[3], emb[4])
                3. Decode: one scalar score per hyperedge → ``scores.shape = (2,)``

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
                Must contain **all** nodes in the hypergraph.
            hyperedge_index: Hyperedge connectivity of shape ``(2, num_incidences)``
                with **global** node IDs.

        Returns:
            Scores of shape ``(num_hyperedges,)``.
        """
        if self.encoder is None:
            raise ValueError("Encoder is not defined for this HLP module.")

        # Encode: HyperGCN applies Laplacian-based message passing
        # Example: x: (num_nodes, in_channels)
        #          -> node_embeddings: (num_nodes, out_channels)
        node_embeddings: Tensor = self.encoder(x, hyperedge_index)

        # Aggregate: pool node embeddings per hyperedge
        # shape: (num_hyperedges, out_channels)
        hyperedge_embeddings = HyperedgeAggregator(hyperedge_index, node_embeddings).pool(
            self.aggregation
        )

        # Decode: linear projection to scalar score per hyperedge
        # shape: (num_hyperedges, 1) -> squeeze -> (num_hyperedges,)
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
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def __eval_step(self, batch: HData, stage: Stage) -> Tensor:
        scores = self.forward(batch.x, batch.hyperedge_index)
        labels = batch.y
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)
        return loss
