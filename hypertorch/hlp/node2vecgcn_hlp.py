from torch import Tensor, nn, optim
from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hypertorch.models import GCN, Node2VecGCN, Node2VecConfig, SLP
from hypertorch.types import EdgeIndex, HData, HyperedgeIndex, GraphReductionStrategyEnum
from hypertorch.nn import HyperedgeAggregator
from hypertorch.utils import Stage

from hypertorch.hlp.common import HlpModule, stage_metric_name
from hypertorch.hlp.node2vec_common import (
    NODE2VEC_JOINT_MODE,
    NODE2VEC_PRECOMPUTED_MODE,
    Node2VecGCNHlpConfig,
    Node2VecHlpConfig,
    Node2VecMode,
    Node2VecWalkLoaderState,
    _next_walk_batch,
    _to_gcn_config,
    _to_node2vec_encoder,
    _to_node2vec_edge_index,
    _validate_global_node_ids,
    _validate_walk_length_and_context_size,
)


class Node2VecGCNEncoderConfig(TypedDict):
    """
    Configuration for the Node2Vec encoder in ``Node2VecGCNHlpModule``.

    Attributes:
        mode: Whether to use precomputed node embeddings from ``x`` or train a Node2Vec encoder
            jointly inside the module.
        num_features: Dimension of the node embeddings consumed by the decoder.
        node2vec_config: Shared Node2Vec configuration used in joint mode, or metadata for
            validating precomputed embeddings.
        gcn_config: Configuration for the GCN layers.
    """

    mode: NotRequired[Node2VecMode]
    num_features: int
    node2vec_config: Node2VecHlpConfig
    gcn_config: Node2VecGCNHlpConfig


class Node2VecGCNHlpModule(HlpModule):
    """
    A LightningModule for Node2Vec-based Hyperedge Link Prediction with GCN encoder.

    Supports two modes:
        - ``precomputed``: use node embeddings already stored in ``batch.x``.
        - ``joint``: train a Node2Vec encoder jointly with the GCN layers and hyperedge decoder.

    Attributes:
        encoder: Optional Node2Vec-GCN encoder inherited from ``HlpModule``.
        decoder: SLP decoder module inherited from ``HlpModule``.
        loss_fn: Loss function inherited from ``HlpModule``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``HlpModule``.
        train_metrics: Optional training metrics inherited from ``HlpModule``.
        val_metrics: Optional validation metrics inherited from ``HlpModule``.
        test_metrics: Optional test metrics inherited from ``HlpModule``.
        mode: Whether to use precomputed or joint Node2Vec embeddings.
        embedding_dim: Node embedding dimension consumed by the GCN stack.
        node2vec_hlp_config: Node2Vec HLP configuration.
        gcn_hlp_config: GCN HLP configuration.
        precomputed_gcn_encoder: GCN encoder used in precomputed mode.
        aggregation: Method to aggregate node embeddings per hyperedge. Defaults to ``"mean"``.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
        weight_decay: Weight decay for the optimizer. Defaults to ``0.0``.
        random_walk_batch_size: Batch size used for Node2Vec walk loss in joint mode.
            Defaults to ``128``.
        node2vec_loss_weight: Weight applied to Node2Vec walk loss in joint mode.
            Defaults to ``1.0``.
    """

    def __init__(
        self,
        encoder_config: Node2VecGCNEncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = "mean",
        loss_fn: nn.Module | None = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the Node2Vec-GCN HLP module.

        Args:
            encoder_config: Configuration for Node2Vec embeddings and GCN layers.
            aggregation: Method used to aggregate node embeddings per hyperedge.
                Defaults to ``mean``.
            loss_fn: Optional HLP loss function. Defaults to ``BCEWithLogitsLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.001``.
            weight_decay: Weight decay for the optimizer. Defaults to ``0.0``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior of
                ``torchmetrics``. Defaults to ``None``.
        """
        self.mode: Node2VecMode = encoder_config.get("mode", NODE2VEC_JOINT_MODE)
        self.embedding_dim: int = encoder_config["num_features"]

        self.node2vec_hlp_config: Node2VecHlpConfig = encoder_config["node2vec_config"]
        self.gcn_hlp_config: Node2VecGCNHlpConfig = encoder_config["gcn_config"]

        node2vecgcn_encoder = (
            self.__build_node2vecgcn_encoder(
                embedding_dim=self.embedding_dim,
                node2vec_config=self.node2vec_hlp_config,
                gcn_config=self.gcn_hlp_config,
                mode=self.mode,
            )
            if self.mode == NODE2VEC_JOINT_MODE
            else None
        )

        decoder = SLP(in_channels=self.gcn_hlp_config["out_channels"], out_channels=1)

        super().__init__(
            encoder=node2vecgcn_encoder,
            decoder=decoder,
            loss_fn=loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.precomputed_gcn_encoder: GCN | None = (
            self.__build_gcn_encoder(self.embedding_dim, self.gcn_hlp_config)
            if self.mode == NODE2VEC_PRECOMPUTED_MODE
            else None
        )

        self.aggregation: Literal["mean", "max", "min", "sum"] = aggregation
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.random_walk_batch_size: int = self.node2vec_hlp_config.get(
            "random_walk_batch_size", 128
        )
        self.node2vec_loss_weight: float = self.node2vec_hlp_config.get("node2vec_loss_weight", 1.0)

        self.__walk_loader_state: Node2VecWalkLoaderState = Node2VecWalkLoaderState()

    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        global_node_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Score hyperedges from Node2Vec embeddings refined by GCN.

        Args:
            x: Node feature or precomputed embedding matrix.
            hyperedge_index: Hyperedge incidence tensor.
            global_node_ids: Optional global node IDs for joint Node2Vec lookup.
                Defaults to ``None``.

        Returns:
            scores: Predicted hyperedge scores.

        Raises:
            ValueError: If the configured mode cannot supply node embeddings.
        """
        gcn_edge_index = self.__to_gcn_edge_index(hyperedge_index)

        if self.mode == NODE2VEC_JOINT_MODE:
            encoder = _to_node2vec_encoder(self.encoder, self.mode)
            _validate_global_node_ids(encoder.num_embeddings, global_node_ids, self.mode)
            node_embeddings = encoder(batch=global_node_ids, edge_index=gcn_edge_index)
        else:
            if x.size(1) != self.embedding_dim:
                raise ValueError(
                    f"Expected precomputed node embeddings with dimension "
                    f"{self.embedding_dim}, got {x.size(1)}."
                )
            if self.precomputed_gcn_encoder is None:
                raise ValueError("Precomputed GCN encoder is not initialized.")
            node_embeddings = self.precomputed_gcn_encoder(x, gcn_edge_index)

        hyperedge_embeddings = HyperedgeAggregator(
            hyperedge_index,
            node_embeddings,
        ).pool(self.aggregation)

        return self.decoder(hyperedge_embeddings).squeeze(-1)

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Run a training step.

        In joint mode this combines HLP loss with one stochastic Node2Vec walk loss batch.

        Args:
            batch: Training batch.
            batch_idx: Batch index, unused.

        Returns:
            loss: Training loss.
        """
        scores = self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)
        labels = batch.y
        batch_size = batch.num_hyperedges

        if self.mode == NODE2VEC_JOINT_MODE:
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
                positive_random_walk, negative_random_walk
            )
            loss = hlp_loss + (self.node2vec_loss_weight * node2vec_loss)

            self.log(
                stage_metric_name(Stage.TRAIN, "hlp_loss"),
                hlp_loss,
                prog_bar=True,
                batch_size=batch_size,
                **self.metrics_log_kwargs,
            )
            self.log(
                stage_metric_name(Stage.TRAIN, "node2vec_loss"),
                node2vec_loss,
                prog_bar=True,
                batch_size=batch_size,
                **self.metrics_log_kwargs,
            )
            self.log(
                stage_metric_name(Stage.TRAIN, "loss"),
                loss,
                prog_bar=True,
                batch_size=batch_size,
                **self.metrics_log_kwargs,
            )
        else:
            loss = self._compute_loss(scores, labels, batch_size, Stage.TRAIN)

        self._compute_metrics(scores, labels, batch_size, Stage.TRAIN)
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
        return self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)

    def configure_optimizers(self) -> optim.Adam:
        """
        Configure the optimizer.

        Returns:
            optimizer: Adam optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def __build_gcn_encoder(self, embedding_dim: int, gcn_config: Node2VecGCNHlpConfig) -> GCN:
        """
        Build a GCN encoder for precomputed mode.

        Args:
            embedding_dim: Input embedding dimension.
            gcn_config: HLP-side GCN configuration.

        Returns:
            encoder: GCN encoder.
        """
        return GCN(**_to_gcn_config(embedding_dim, gcn_config))

    def __build_node2vecgcn_encoder(
        self,
        embedding_dim: int,
        node2vec_config: Node2VecHlpConfig,
        gcn_config: Node2VecGCNHlpConfig,
        mode: Node2VecMode,
    ) -> Node2VecGCN:
        """
        Build the joint Node2Vec-GCN encoder.

        Args:
            embedding_dim: Node2Vec embedding dimension.
            node2vec_config: Node2Vec HLP configuration.
            gcn_config: HLP-side GCN configuration.
            mode: Node2Vec training mode used in validation errors.

        Returns:
            encoder: Joint Node2Vec-GCN encoder.
        """
        _validate_walk_length_and_context_size(
            walk_length=node2vec_config.get("walk_length", 20),
            context_size=node2vec_config.get("context_size", 10),
        )

        edge_index, num_nodes = _to_node2vec_edge_index(node2vec_config, mode)

        model_node2vec_config: Node2VecConfig = {
            "edge_index": edge_index,
            "embedding_dim": embedding_dim,
            "walk_length": node2vec_config.get("walk_length", 20),
            "context_size": node2vec_config.get("context_size", 10),
            "num_walks_per_node": node2vec_config.get("num_walks_per_node", 10),
            "p": node2vec_config.get("p", 1.0),
            "q": node2vec_config.get("q", 1.0),
            "num_negative_samples": node2vec_config.get("num_negative_samples", 1),
            "num_nodes": num_nodes,
            "sparse": node2vec_config.get("sparse", False),
        }

        return Node2VecGCN(
            node2vec_config=model_node2vec_config,
            gcn_config=_to_gcn_config(embedding_dim, gcn_config),
        )

    def __eval_step(self, batch: HData, stage: Stage) -> Tensor:
        """
        Run shared evaluation logic for a stage.

        Args:
            batch: Input batch.
            stage: Current evaluation stage.

        Returns:
            loss: Computed loss.
        """
        scores = self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)
        labels = batch.y
        batch_size = batch.num_hyperedges

        loss = self._compute_loss(scores, labels, batch_size, stage)
        self._compute_metrics(scores, labels, batch_size, stage)
        return loss

    def __to_gcn_edge_index(self, hyperedge_index: Tensor) -> Tensor:
        """
        Reduce a hyperedge index to the graph edge index used by GCN.

        Args:
            hyperedge_index: Hyperedge incidence tensor.

        Returns:
            edge_index: Reduced graph edge index without self-loops.
        """
        graph_reduction_strategy = self.gcn_hlp_config.get(
            "graph_reduction_strategy", GraphReductionStrategyEnum.CLIQUE_EXPANSION
        )
        reduced_gcn_edge_index = HyperedgeIndex(hyperedge_index).reduce(
            strategy=graph_reduction_strategy,
            num_nodes=self.gcn_hlp_config.get("num_nodes"),
        )
        return EdgeIndex(reduced_gcn_edge_index).remove_selfloops().item
