from torch import Tensor, nn, optim
from typing import Literal, Optional, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hyperbench.models import Node2Vec, SLP
from hyperbench.types import HData
from hyperbench.utils import Stage
from hyperbench.nn import HyperedgeAggregator

from hyperbench.hlp.common import HlpModule
from hyperbench.hlp.node2vec_common import (
    NODE2VEC_JOINT_MODE,
    Node2VecHlpConfig,
    Node2VecMode,
    Node2VecWalkLoaderState,
    _next_walk_batch,
    _to_node2vec_edge_index,
    _to_node2vec_encoder,
    _validate_global_node_ids,
    _validate_walk_length_and_context_size,
)


class Node2VecSLPEncoderConfig(TypedDict):
    """
    Configuration for the Node2Vec encoder in ``Node2VecSLPHlpModule``.

    Args:
        mode: Whether to use precomputed node embeddings from ``x`` or train a Node2Vec encoder jointly inside the module.
        num_features: Dimension of the node embeddings consumed by the decoder.
        node2vec_config: Shared Node2Vec configuration used in joint mode, or metadata for validating precomputed embeddings.
    """

    mode: NotRequired[Node2VecMode]
    num_features: int
    node2vec_config: Node2VecHlpConfig


class Node2VecSLPHlpModule(HlpModule):
    """
    A LightningModule for Node2Vec-based Hyperedge Link Prediction.

    Supports two modes:
    - ``precomputed``: use node embeddings already stored in ``batch.x``.
    - ``joint``: train a Node2Vec encoder jointly with the hyperedge decoder.

    Args:
        encoder_config: Configuration for the Node2Vec encoder.
        aggregation: Method to aggregate node embeddings per hyperedge.
        loss_fn: Loss function. Defaults to ``BCEWithLogitsLoss``.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
        weight_decay: Weight decay (L2 regularization) for the optimizer. Defaults to ``0.0`` (no weight decay).
        metrics: Optional dictionary of metric functions.
    """

    def __init__(
        self,
        encoder_config: Node2VecSLPEncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = "mean",
        loss_fn: Optional[nn.Module] = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        metrics: Optional[MetricCollection] = None,
    ):
        self.mode = encoder_config.get("mode", NODE2VEC_JOINT_MODE)
        self.embedding_dim = encoder_config["num_features"]
        node2vec_config = encoder_config["node2vec_config"]

        encoder = (
            self.__build_node2vec_encoder(self.embedding_dim, node2vec_config, self.mode)
            if self.mode == NODE2VEC_JOINT_MODE
            else None
        )

        decoder = SLP(in_channels=self.embedding_dim, out_channels=1)

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            loss_fn=loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics,
        )

        self.aggregation = aggregation
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_walk_batch_size = node2vec_config.get("random_walk_batch_size", 128)
        self.node2vec_loss_weight = node2vec_config.get("node2vec_loss_weight", 1.0)

        self.__walk_loader_state = Node2VecWalkLoaderState()

    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        global_node_ids: Optional[Tensor] = None,
    ) -> Tensor:
        # Encode: get node embeddings from precomputation or joint encoder
        if self.mode == NODE2VEC_JOINT_MODE:
            encoder = _to_node2vec_encoder(self.encoder, self.mode)
            _validate_global_node_ids(encoder.num_embeddings, global_node_ids, self.mode)
            node_embeddings = encoder(batch=global_node_ids)
        else:
            if x.size(1) != self.embedding_dim:
                raise ValueError(
                    f"Expected precomputed node embeddings with dimension "
                    f"{self.embedding_dim}, got {x.size(1)}."
                )
            node_embeddings = x

        # Aggregate: pool node embeddings per hyperedge
        # shape: (num_hyperedges, embedding_dim)
        hyperedge_embeddings = HyperedgeAggregator(hyperedge_index, node_embeddings).pool(
            self.aggregation
        )

        # Decode: linear projection to scalar score per hyperedge
        # shape: (num_hyperedges, 1) -> squeeze -> (num_hyperedges,)
        scores: Tensor = self.decoder(hyperedge_embeddings).squeeze(-1)
        return scores

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        scores = self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)
        labels = batch.y
        batch_size = batch.num_hyperedges

        if self.mode == NODE2VEC_JOINT_MODE:
            # Node2Vec.loss() is already a stochastic objective over sampled walks,
            # so one walk batch is a standard SGD estimate, not a logically different loss,
            # meaning we can optimize training by using a single walk batch per training step,
            # instead of averaging over multiple walk batches.
            positive_random_walk, negative_random_walk = _next_walk_batch(
                mode=self.mode,
                encoder=self.encoder,
                batch_size=self.random_walk_batch_size,
                state=self.__walk_loader_state,
            )
            positive_random_walk = positive_random_walk.to(self.device)
            negative_random_walk = negative_random_walk.to(self.device)

            hlp_loss = self.loss_fn(scores, labels)
            node2vec_loss = _to_node2vec_encoder(self.encoder, self.mode).loss(
                positive_random_walk,
                negative_random_walk,
            )
            loss = hlp_loss + (self.node2vec_loss_weight * node2vec_loss)

            loss_prefix = Stage.TRAIN.value
            self.log(f"{loss_prefix}_hlp_loss", hlp_loss, prog_bar=True, batch_size=batch_size)
            self.log(
                f"{loss_prefix}_node2vec_loss", node2vec_loss, prog_bar=True, batch_size=batch_size
            )
            self.log(f"{loss_prefix}_loss", loss, prog_bar=True, batch_size=batch_size)
        else:
            loss = self._compute_loss(scores, labels, batch_size, Stage.TRAIN)

        self._compute_metrics(scores, labels, batch_size, Stage.TRAIN)
        return loss

    def validation_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.VAL)

    def test_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.__eval_step(batch, Stage.TEST)

    def predict_step(self, batch: HData, batch_idx: int) -> Tensor:
        return self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def __build_node2vec_encoder(
        self,
        embedding_dim: int,
        node2vec_config: Node2VecHlpConfig,
        mode: Node2VecMode,
    ) -> Node2Vec:
        _validate_walk_length_and_context_size(
            walk_length=node2vec_config.get("walk_length", 20),
            context_size=node2vec_config.get("context_size", 10),
        )

        edge_index, num_nodes = _to_node2vec_edge_index(node2vec_config, mode)

        return Node2Vec(
            edge_index=edge_index,
            embedding_dim=embedding_dim,
            walk_length=node2vec_config.get("walk_length", 20),
            context_size=node2vec_config.get("context_size", 10),
            num_walks_per_node=node2vec_config.get("num_walks_per_node", 10),
            p=node2vec_config.get("p", 1.0),
            q=node2vec_config.get("q", 1.0),
            num_negative_samples=node2vec_config.get("num_negative_samples", 1),
            num_nodes=num_nodes,
            sparse=node2vec_config.get("sparse", False),
        )

    def __eval_step(self, batch: HData, stage: Stage) -> Tensor:
        scores = self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)
        labels = batch.y
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)
        return loss
