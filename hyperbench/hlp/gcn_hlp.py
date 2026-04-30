from torch import Tensor, nn, optim
from typing import Dict, Literal, Optional, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hyperbench.models import GCN, SLP
from hyperbench.nn import HyperedgeAggregator
from hyperbench.types import EdgeIndex, HData, HyperedgeIndex
from hyperbench.utils import ActivationFn, Stage

from hyperbench.hlp.common import HlpModule


class GCNEncoderConfig(TypedDict):
    """
    Configuration for the GCN encoder in GCNHlpModule.

    Args:
        in_channels: Number of input features per node.
        out_channels: Number of output features (embedding size) per node.
        hidden_channels: Number of hidden units in the intermediate GCN layers.
        num_layers: Number of GCN layers. Defaults to ``2``.
        drop_rate: Dropout rate applied after each hidden GCN layer. Defaults to ``0.0``.
        bias: Whether to include bias terms. Defaults to ``True``.
        improved: Whether to use the improved GCN normalization. Defaults to ``False``.
        add_self_loops: Whether to add self-loops before convolution. Defaults to ``True``.
        normalize: Whether to normalize the adjacency matrix in ``GCNConv``. Defaults to ``True``.
        cached: Whether to cache the normalized graph in ``GCNConv``. Defaults to ``False``.
        graph_reduction_strategy: Strategy for reducing the hypergraph to a graph. Defaults to ``"clique_expansion"``.
        activation_fn: Activation function to use after each hidden layer. Defaults to ``nn.ReLU``.
        activation_fn_kwargs: Keyword arguments for the activation function. Defaults to empty dict.
    """

    in_channels: int
    out_channels: int
    hidden_channels: NotRequired[int]
    num_layers: NotRequired[int]
    drop_rate: NotRequired[float]
    bias: NotRequired[bool]
    improved: NotRequired[bool]
    add_self_loops: NotRequired[bool]
    normalize: NotRequired[bool]
    cached: NotRequired[bool]
    graph_reduction_strategy: NotRequired[Literal["clique_expansion"]]
    activation_fn: NotRequired[ActivationFn]
    activation_fn_kwargs: NotRequired[Dict]


class GCNHlpModule(HlpModule):
    """
    A LightningModule for GCN-based HLP.

    Uses a graph reduction of the input hypergraph to run GCN over nodes,
    aggregates node embeddings per hyperedge, and scores each hyperedge with a linear decoder.

    Args:
        encoder_config: Configuration for the GCN encoder.
        aggregation: Method to aggregate node embeddings per hyperedge. Defaults to ``"mean"``.
        loss_fn: Loss function. Defaults to ``BCEWithLogitsLoss``.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
        weight_decay: L2 regularization. Defaults to ``0.0``.
        metrics: Optional metric collection for evaluation.
    """

    def __init__(
        self,
        encoder_config: GCNEncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = "mean",
        loss_fn: Optional[nn.Module] = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        metrics: Optional[MetricCollection] = None,
    ):
        encoder = GCN(
            in_channels=encoder_config["in_channels"],
            out_channels=encoder_config["out_channels"],
            hidden_channels=encoder_config.get("hidden_channels"),
            num_layers=encoder_config.get("num_layers", 2),
            drop_rate=encoder_config.get("drop_rate", 0.0),
            bias=encoder_config.get("bias", True),
            activation_fn=encoder_config.get("activation_fn"),
            activation_fn_kwargs=encoder_config.get("activation_fn_kwargs"),
            improved=encoder_config.get("improved", False),
            add_self_loops=encoder_config.get("add_self_loops", True),
            normalize=encoder_config.get("normalize", True),
            cached=encoder_config.get("cached", False),
        )
        decoder = SLP(in_channels=encoder_config["out_channels"], out_channels=1)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss_fn=loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics,
        )

        self.encoder_config = encoder_config
        self.aggregation = aggregation
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Reduce the hypergraph to a graph, encode nodes with GCN, aggregate per hyperedge, and score.

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge connectivity of shape ``(2, num_incidences)``.

        Returns:
            Logit scores of shape ``(num_hyperedges,)``.
        """
        if self.encoder is None:
            raise ValueError("Encoder is not defined for this HLP module.")

        # Reduce hypergraph to graph and remove self-loops
        reduced_edge_index = HyperedgeIndex(hyperedge_index).reduce(
            strategy=self.encoder_config.get("graph_reduction_strategy", "clique_expansion")
        )
        edge_index = EdgeIndex(reduced_edge_index).remove_selfloops().item

        # Encode nodes with GCN
        node_embeddings: Tensor = self.encoder(x, edge_index)

        # Aggregate node embeddings per hyperedge
        hyperedge_embeddings = HyperedgeAggregator(hyperedge_index, node_embeddings).pool(
            self.aggregation
        )

        return self.decoder(hyperedge_embeddings).squeeze(-1)

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
