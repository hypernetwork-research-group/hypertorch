from torch import Tensor, nn, optim
from typing import Literal, Optional, TypedDict
from typing_extensions import NotRequired
from hyperbench.models import HGNN, SLP
from hyperbench.nn import HyperedgeAggregator
from hyperbench.types import HData
from torchmetrics import MetricCollection
from hyperbench.utils import Stage

from hyperbench.hlp.common import HlpModule


class HGNNEncoderConfig(TypedDict):
    """
    Configuration for the HGNN encoder in HGNNHlpModule.

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Number of hidden units in the intermediate HGNN layer.
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


class HGNNHlpModule(HlpModule):
    """
    A LightningModule for HGNN-based Hyperedge Link Prediction.

    Uses HGNN as an encoder to produce structure-aware node embeddings via
    spectral hypergraph convolution, aggregates them per hyperedge,
    and scores each hyperedge with a linear decoder.

    Args:
        encoder_config: Configuration for the HGNN encoder.
        aggregation: Method to aggregate node embeddings per hyperedge. Defaults to ``"mean"``.
        loss_fn: Loss function. Defaults to ``BCEWithLogitsLoss``.
        lr: Learning rate for the optimizer. Defaults to ``0.01``.
        weight_decay: L2 regularization. Defaults to ``5e-4``.
        metrics: Optional metric collection for evaluation.
    """

    def __init__(
        self,
        encoder_config: HGNNEncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = "mean",
        loss_fn: Optional[nn.Module] = None,
        lr: float = 0.001,
        weight_decay: float = 5e-4,
        metrics: Optional[MetricCollection] = None,
    ):
        encoder = HGNN(
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

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Run the full HGNN-based hyperedge link prediction pipeline.

        The pipeline has three stages:
        1. Encode: HGNN applies two rounds of ``D_n^{-1/2} H D_e^{-1} H^T D_n^{-1/2}``
           smoothing to propagate information through the hypergraph topology (nodes ->
           hyperedges -> nodes). The output is a structure-aware node embedding matrix of
           shape ``(num_nodes, out_channels)``.
        2. Aggregate: For each hyperedge being scored, pool the embeddings of its member
           nodes using the configured strategy (mean/max/min/sum). This produces a hyperedge
           embedding that summarizes the collective representation of the hyperedge's nodes.
           Shape: ``(num_hyperedges, out_channels)``.
        3. Decode: A single linear layer (SLP) projects each hyperedge embedding to a
           scalar score representing the likelihood that the hyperedge is a real (positive)
           hyperedge. Shape: ``(num_hyperedges,)``.

        Examples:
            Given 5 nodes with 8 features and 2 hyperedges::

                >>> x.shape  # (5, 8) - all nodes in the hypergraph
                >>> hyperedge_index = [[0, 1, 2, 3, 4],  # node IDs
                ...                    [0, 0, 0, 1, 1]]  # hyperedge IDs

            The forward pass:
                1. HGNN encodes all 5 nodes using the hypergraph Laplacian.
                   ``node_embeddings.shape = (5, out_channels)``
                2. Aggregate per hyperedge:
                   - hyperedge 0: pool(emb[0], emb[1], emb[2])
                   - hyperedge 1: pool(emb[3], emb[4])
                   ``hyperedge_embeddings.shape = (2, out_channels)``
                3. Decode: one scalar per hyperedge -> ``scores.shape = (2,)``

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
                Must contain **all** nodes referenced in ``hyperedge_index``.
            hyperedge_index: Hyperedge connectivity of shape ``(2, num_incidences)``,
                with row 0 containing global node IDs and row 1 hyperedge IDs.

        Returns:
            Logit scores of shape ``(num_hyperedges,)``. Pass through sigmoid to get
            probabilities, or use directly with ``BCEWithLogitsLoss``.
        """
        if self.encoder is None:
            raise ValueError("Encoder is not defined for this HLP module.")

        # Encode: two-hop HGNN smoothing (nodes -> hyperedges -> nodes), no graph reduction
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
