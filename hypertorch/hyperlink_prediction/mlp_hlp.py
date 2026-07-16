from torch import Tensor, nn, optim
from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired
from hypertorch.models import MLP, SLP
from hypertorch.nn import HyperedgeAggregator
from hypertorch.types import HData
from torchmetrics import MetricCollection
from hypertorch.utils import ActivationFn, NormalizationFn, Stage

from hypertorch.hyperlink_prediction.common import HLPPredictor


class MLPEncoderConfig(TypedDict):
    """
    Configuration for the MLP encoder in MLPPredictor.

    Attributes:
        in_channels: Number of input features per node.
        out_channels: Number of output features (embedding size) per node.
        num_layers: Number of layers in the MLP encoder.
        hidden_channels: Optional number of hidden units per layer. If ``None``, no hidden layers
            are used and the encoder is a simple linear layer.
        activation_fn: Optional activation function class to use in the MLP encoder.
            If ``None``, no activation function is applied.
        activation_fn_kwargs: Optional dictionary of keyword arguments to pass to the activation
            function constructor.
        normalization_fn: Optional normalization function class to use in the MLP encoder.
            If ``None``, no normalization is applied.
        normalization_fn_kwargs: Optional dictionary of keyword arguments to pass to the
            normalization function constructor.
        bias: Whether to include bias terms in the MLP layers. Defaults to ``True``.
        drop_rate: Dropout rate to apply after each MLP layer (except the last one).
            Defaults to ``0.0`` (no dropout).
    """

    in_channels: int
    out_channels: NotRequired[int]
    num_layers: NotRequired[int]
    hidden_channels: NotRequired[int | None]
    activation_fn: NotRequired[ActivationFn | None]
    activation_fn_kwargs: NotRequired[dict | None]
    normalization_fn: NotRequired[NormalizationFn | None]
    normalization_fn_kwargs: NotRequired[dict | None]
    bias: NotRequired[bool]
    drop_rate: NotRequired[float]


class MLPPredictor(HLPPredictor):
    """
    A LightningModule for MLP-based HLP predictor.

    Uses an MLP encoder to produce node embeddings, aggregates them per hyperedge
    via mean pooling, and scores each hyperedge with a linear decoder.

    Attributes:
        encoder: MLP encoder module inherited from ``HLPPredictor``.
        decoder: SLP decoder module inherited from ``HLPPredictor``.
        loss_fn: Loss function inherited from ``HLPPredictor``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``HLPPredictor``.
        train_metrics: Optional training metrics inherited from ``HLPPredictor``.
        val_metrics: Optional validation metrics inherited from ``HLPPredictor``.
        test_metrics: Optional test metrics inherited from ``HLPPredictor``.
        aggregation: Method to aggregate node embeddings per hyperedge. Defaults to ``"mean"``.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
    """

    def __init__(
        self,
        encoder_config: MLPEncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = "mean",
        loss_fn: nn.Module | None = None,
        lr: float = 0.001,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the MLP-based HLP predictor.

        Args:
            encoder_config: Configuration for the MLP encoder.
            aggregation: Method used to aggregate node embeddings per hyperedge.
                Defaults to ``"mean"``.
            loss_fn: Optional loss function. Defaults to ``BCEWithLogitsLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.001``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior
                of ``torchmetrics``. Defaults to ``None``.
        """
        # The encoder outputs node embeddings of shape (num_nodes, out_channels).
        encoder = MLP(
            in_channels=encoder_config["in_channels"],
            hidden_channels=encoder_config.get("hidden_channels"),
            out_channels=encoder_config.get("out_channels", 1),
            num_layers=encoder_config.get("num_layers", 1),
            activation_fn=encoder_config.get("activation_fn"),
            activation_fn_kwargs=encoder_config.get("activation_fn_kwargs"),
            normalization_fn=encoder_config.get("normalization_fn"),
            normalization_fn_kwargs=encoder_config.get("normalization_fn_kwargs"),
            bias=encoder_config.get("bias", True),
            drop_rate=encoder_config.get("drop_rate", 0.0),
        )

        # The decoder takes in the aggregated hyperedge embeddings of shape
        # (num_hyperedges, encoder_config.out_channels)
        # and produces a score for each hyperedge of shape (num_hyperedges, 1).
        decoder = SLP(in_channels=encoder_config.get("out_channels", 1), out_channels=1)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss_fn=loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.aggregation: Literal["mean", "max", "min", "sum"] = aggregation
        self.lr: float = lr

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

                >>> Encoder maps each node to an embedding vector.
                >>> Aggregate embeddings by summing them per hyperedge:
                ...   - hyperedge 0: emb[0] + emb[1] + emb[2]
                ...   - hyperedge 1: emb[2] + emb[3]
                >>> Sums are divided by the number of nodes per hyperedge (mean pooling):
                ...   - hyperedge 0: (emb[0] + emb[1] + emb[2]) / 3
                ...   - hyperedge 1: (emb[2] + emb[3]) / 2
                >>> Decoder scores each hyperedge embedding, producing one scalar per hyperedge.

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
        # - hyperedge 0 contains node 0, 1, 2 -> aggregate([e00, e01], [e10, e11], [e20, e21])
        #                                         -> [pooled_0, pooled_1]
        # - hyperedge 1 contains node 2, 3 -> aggregate([e20, e21], [e30, e31])
        #                                  -> [pooled_0, pooled_1]
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
        """
        Run a training step.

        Args:
            batch: Training batch.
            batch_idx: Batch index, unused.

        Returns:
            loss: Training loss.
        """
        scores = self.forward(batch.x, batch.hyperedge_index)
        target_scores, target_labels = self._target_scores_and_labels(scores, batch)
        batch_size = target_labels.size(0)

        loss = self._compute_loss(target_scores, target_labels, batch_size, Stage.TRAIN)
        self._compute_metrics(target_scores, target_labels, batch_size, Stage.TRAIN)
        return loss

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
        scores = self.forward(batch.x, batch.hyperedge_index)
        target_scores, target_labels = self._target_scores_and_labels(scores, batch)
        batch_size = target_labels.size(0)

        loss = self._compute_loss(target_scores, target_labels, batch_size, stage)
        self._compute_metrics(target_scores, target_labels, batch_size, stage)
        return loss
