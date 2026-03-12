from torch import Tensor, nn, optim
from typing import Literal, Optional
from hyperbench.models import MLP, SLP
from hyperbench.nn import HyperedgeAggregator
from hyperbench.types import HData
from hyperbench.utils import Aggregation, ActivationFn, NamedMetricFnDict, Stage

from .hlp import HlpModule


class EncoderConfig:
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        num_layers: int = 1,
        hidden_channels: Optional[int] = None,
        activation_fn: Optional[ActivationFn] = None,
        drop_rate: float = 0.0,
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate


class MLPHlpModule(HlpModule):
    """
    A LightningModule for MLP-based Hyperedge Link Prediction.

    Uses an MLP encoder to produce node embeddings, aggregates them per hyperedge
    via mean pooling, and scores each hyperedge with a linear decoder.

    Args:
        in_channels: Number of input node features.
        hidden_channels: Number of hidden channels in the MLP encoder.
        out_channels: Dimensionality of node embeddings produced by the encoder.
        num_layers: Number of hidden layers in the MLP encoder.
        aggregation: Method to aggregate node embeddings per hyperedge.
        activation_fn: Activation function class for the MLP encoder. Defaults to ``ReLU``.
        loss_fn: Loss function. Defaults to ``BCEWithLogitsLoss``.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
        metrics: Optional dictionary of metric functions.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = Aggregation.MEAN,
        loss_fn: Optional[nn.Module] = None,
        lr: float = 0.001,
        metrics: Optional[NamedMetricFnDict] = None,
    ):
        # The encoder outputs node embeddings of shape (num_nodes, out_channels).
        encoder = MLP(
            in_channels=encoder_config.in_channels,
            hidden_channels=encoder_config.hidden_channels,
            out_channels=encoder_config.out_channels,
            num_layers=encoder_config.num_layers,
            activation_fn=encoder_config.activation_fn,
            drop_rate=encoder_config.drop_rate,
        )

        # The decoder takes in the aggregated hyperedge embeddings of shape (num_hyperedges, encoder_config.out_channels)
        # and produces a score for each hyperedge of shape (num_hyperedges, 1).
        decoder = SLP(in_channels=encoder_config.out_channels, out_channels=1)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss_fn=loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics,
        )

        self.aggregation = aggregation
        self.lr = lr

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Encode node features, aggregate per hyperedge via mean pooling, and score.

        Examples:
            Given 4 nodes with 3 features each and 2 hyperedges:
                >>> x = [[0.1, 0.2, 0.3],   # node 0
                ...      [0.4, 0.5, 0.6],   # node 1
                ...      [0.7, 0.8, 0.9],   # node 2
                ...      [1.0, 1.1, 1.2]]   # node 3

                >>> # hyperedge 0 = {node 0, node 1, node 2}
                >>> # hyperedge 1 = {node 2, node 3}
                >>> hyperedge_index = [[0, 1, 2, 2, 3],   # node ids
                ...                    [0, 0, 0, 1, 1]]   # hyperedge ids

            The forward pass:
                1. Encoder maps each node to an embedding vector.
                2. Aggregate embeddings by summing them per hyperedge:
                    - hyperedge 0: emb[0] + emb[1] + emb[2]
                    - hyperedge 1: emb[2] + emb[3]
                3. Sums are divided by the number of nodes per hyperedge (mean pooling):
                    - hyperedge 0: (emb[0] + emb[1] + emb[2]) / 3
                    - hyperedge 1: (emb[2] + emb[3]) / 2
                4. Decoder scores each hyperedge embedding, producing one scalar per hyperedge.

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge connectivity of shape ``(2, num_incidences)``.

        Returns:
            Scores of shape ``(num_hyperedges,)``.
        """
        if self.encoder is None:
            raise ValueError("Encoder is not defined for this HLP module.")

        # Encode: map each node raw features to an embedding vector.
        # x: (num_nodes, in_channels) -> node_embeddings: (num_nodes, out_channels)
        # Example: in_channels=3, out_channels=2
        #          -> node 0: [0.1, 0.2, 0.3] -> [e00, e01]
        #          -> node 1: [0.4, 0.5, 0.6] -> [e10, e11]
        #          -> node 2: [0.7, 0.8, 0.9] -> [e20, e21]
        #          -> node 3: [1.0, 1.1, 1.2] -> [e30, e31]
        node_embeddings: Tensor = self.encoder(x)

        # Aggregate: for each hyperedge, aggregate the embeddings of its member nodes.
        # Example::
        # - hyperedge 0 contains node 0, 1, 2 -> aggregate([e00, e01], [e10, e11], [e20, e21]) -> [pooled_0, pooled_1]
        # - hyperedge 1 contains node 2, 3 -> aggregate([e20, e21], [e30, e31]) -> [pooled_0, pooled_1]
        # shape: (num_hyperedges, out_channels)
        hyperedge_embeddings = HyperedgeAggregator(hyperedge_index, node_embeddings).pool(
            self.aggregation,
        )

        # Decode: score each hyperedge embedding, producing one scalar per hyperedge.
        # Example:
        # - hyperedge 0: [pooled_0, pooled_1] -> score_0
        # - hyperedge 1: [pooled_0, pooled_1] -> score_1
        # shape: (2,)
        scores: Tensor = self.decoder(hyperedge_embeddings).squeeze(-1)
        return scores

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        scores = self.forward(batch.x, batch.hyperedge_index)
        labels = batch.y
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, Stage.TRAIN)
        return loss

    def validation_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.VAL)

    def test_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.TEST)

    def predict_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.forward(batch.x, batch.hyperedge_index)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def __eval_step(self, batch: HData, stage: Stage) -> Tensor:
        scores = self.forward(batch.x, batch.hyperedge_index)
        labels = batch.y
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)
        return loss
