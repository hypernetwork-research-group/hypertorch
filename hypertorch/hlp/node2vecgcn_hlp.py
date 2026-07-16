from torch import Tensor, nn, optim
from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hypertorch.models import SLP
from hypertorch.types import HData
from hypertorch.nn import HyperedgeAggregator
from hypertorch.utils import Stage

from hypertorch.hlp.common import HLPPredictor, stage_metric_name
from hypertorch.models.node2vec_common import (
    NODE2VEC_JOINT_MODE,
    NODE2VEC_PRECOMPUTED_MODE,
    Node2VecGCNEncoderConfig as Node2VecGCNHLPConfig,
    Node2VecEncoderConfig as Node2VecHLPConfig,
    Node2VecMode,
    Node2VecWalkLoaderState,
    build_gcn_encoder,
    build_node2vecgcn_encoder,
    next_walk_batch,
    to_gcn_edge_index,
    to_node2vec_encoder,
    to_node2vecgcn_encoder,
    validate_global_node_ids,
)


class Node2VecGCNEncoderConfig(TypedDict):
    """
    Configuration for the Node2VecGCN encoder in ``Node2VecGCNPredictor``.

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
    node2vec_config: Node2VecHLPConfig
    gcn_config: Node2VecGCNHLPConfig


class Node2VecGCNPredictor(HLPPredictor):
    """
    A LightningModule for Node2VecGCN-based HLP predictor.

    Supports two modes:
        - ``precomputed``: use node embeddings already stored in ``batch.x``.
        - ``joint``: train a Node2Vec encoder jointly with the GCN layers and hyperedge decoder.

    Attributes:
        encoder: Optional Node2Vec-GCN encoder inherited from ``HLPPredictor``.
        decoder: SLP decoder module inherited from ``HLPPredictor``.
        loss_fn: Loss function inherited from ``HLPPredictor``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``HLPPredictor``.
        train_metrics: Optional training metrics inherited from ``HLPPredictor``.
        val_metrics: Optional validation metrics inherited from ``HLPPredictor``.
        test_metrics: Optional test metrics inherited from ``HLPPredictor``.
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
        Initialize the Node2VecGCN-based HLP predictor.

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

        self.node2vec_hlp_config: Node2VecHLPConfig = encoder_config["node2vec_config"]
        self.gcn_hlp_config: Node2VecGCNHLPConfig = encoder_config["gcn_config"]

        node2vecgcn_encoder = (
            build_node2vecgcn_encoder(
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

        self.precomputed_gcn_encoder: nn.Module | None = (
            build_gcn_encoder(self.embedding_dim, self.gcn_hlp_config)
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
        gcn_edge_index = to_gcn_edge_index(hyperedge_index, self.gcn_hlp_config)

        if self.mode == NODE2VEC_JOINT_MODE:
            encoder = to_node2vecgcn_encoder(self.encoder, self.mode)
            validate_global_node_ids(encoder.num_embeddings, global_node_ids, self.mode)
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
        target_scores, target_labels = self._target_scores_and_labels(scores, batch)
        batch_size = target_labels.size(0)

        if self.mode == NODE2VEC_JOINT_MODE:
            positive_random_walk, negative_random_walk = next_walk_batch(
                mode=self.mode,
                encoder=self.encoder,
                batch_size=self.random_walk_batch_size,
                state=self.__walk_loader_state,
            )
            positive_random_walk = positive_random_walk.to(self.device)
            negative_random_walk = negative_random_walk.to(self.device)

            hlp_loss = self.loss_fn(target_scores, target_labels)
            node2vec_loss = to_node2vec_encoder(self.encoder, self.mode).loss(
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
