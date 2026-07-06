from torch import Tensor, nn, optim
from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hypertorch.models import Node2Vec, SLP
from hypertorch.types import HData
from hypertorch.utils import Stage
from hypertorch.nn import HyperedgeAggregator

from hypertorch.hlp.common import HlpModule, stage_metric_name
from hypertorch.hlp.node2vec_common import (
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

    Attributes:
        mode: Whether to use precomputed node embeddings from ``x`` or train a Node2Vec encoder
            jointly inside the module. Defaults to ``"joint"``.
        num_features: Dimension of the node embeddings consumed by the decoder.
        node2vec_config: Shared Node2Vec configuration used in joint mode, or metadata for
            validating precomputed embeddings.
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

    Attributes:
        encoder: Optional Node2Vec encoder inherited from ``HlpModule``.
        decoder: SLP decoder module inherited from ``HlpModule``.
        loss_fn: Loss function inherited from ``HlpModule``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``HlpModule``.
        train_metrics: Optional training metrics inherited from ``HlpModule``.
        val_metrics: Optional validation metrics inherited from ``HlpModule``.
        test_metrics: Optional test metrics inherited from ``HlpModule``.
        mode: Whether to use precomputed or joint Node2Vec embeddings.
        embedding_dim: Node embedding dimension consumed by the decoder.
        aggregation: Method to aggregate node embeddings per hyperedge.
            Defaults to ``mean``.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
        weight_decay: Weight decay for the optimizer. Defaults to ``0.0``.
        random_walk_batch_size: Batch size used for Node2Vec walk loss in joint mode.
            Defaults to ``128``.
        node2vec_loss_weight: Weight applied to Node2Vec walk loss in joint mode.
            Defaults to ``1.0``.
    """

    def __init__(
        self,
        encoder_config: Node2VecSLPEncoderConfig,
        aggregation: Literal["mean", "max", "min", "sum"] = "mean",
        loss_fn: nn.Module | None = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the Node2Vec-SLP HLP module.

        Args:
            encoder_config: Configuration for Node2Vec embeddings.
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
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.aggregation: Literal["mean", "max", "min", "sum"] = aggregation
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.random_walk_batch_size: int = node2vec_config.get("random_walk_batch_size", 128)
        self.node2vec_loss_weight: float = node2vec_config.get("node2vec_loss_weight", 1.0)

        self.__walk_loader_state: Node2VecWalkLoaderState = Node2VecWalkLoaderState()

    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        global_node_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Score hyperedges from precomputed or jointly trained Node2Vec embeddings.

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
        target_scores, target_labels = self._target_scores_and_labels(scores, batch)
        batch_size = target_labels.size(0)

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

            hlp_loss = self.loss_fn(target_scores, target_labels)
            node2vec_loss = _to_node2vec_encoder(self.encoder, self.mode).loss(
                positive_random_walk,
                negative_random_walk,
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
        return self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)

    def configure_optimizers(self) -> optim.Adam:
        """
        Configure the optimizer.

        Returns:
            optimizer: Adam optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def __build_node2vec_encoder(
        self,
        embedding_dim: int,
        node2vec_config: Node2VecHlpConfig,
        mode: Node2VecMode,
    ) -> Node2Vec:
        """
        Build the joint-mode Node2Vec encoder.

        Args:
            embedding_dim: Node2Vec embedding dimension.
            node2vec_config: Node2Vec HLP configuration.
            mode: Node2Vec training mode used in validation errors.

        Returns:
            encoder: Node2Vec encoder.
        """
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
        """
        Run shared evaluation logic for a stage.

        Args:
            batch: Input batch.
            stage: Current evaluation stage.

        Returns:
            loss: Computed loss.
        """
        scores = self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)
        target_scores, target_labels = self._target_scores_and_labels(scores, batch)
        batch_size = target_labels.size(0)

        loss = self._compute_loss(target_scores, target_labels, batch_size, stage)
        self._compute_metrics(target_scores, target_labels, batch_size, stage)
        return loss
