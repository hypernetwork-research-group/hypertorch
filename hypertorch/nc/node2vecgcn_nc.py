from torch import Tensor, nn, optim
from typing import Any, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection

from hypertorch.hlp.common import stage_metric_name
from hypertorch.models.node2vec_common import (
    NODE2VEC_JOINT_MODE,
    NODE2VEC_PRECOMPUTED_MODE,
    Node2VecGCNEncoderConfig as Node2VecGCNNcConfig,
    Node2VecEncoderConfig as Node2VecNcConfig,
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
from hypertorch.nc.common import NcModule
from hypertorch.types import HData
from hypertorch.utils import Stage


class Node2VecGCNClassifierConfig(TypedDict):
    """
    Configuration for the Node2Vec-GCN node classification module.

    Attributes:
        mode: Whether to use precomputed node embeddings from ``x`` or train a Node2Vec
            encoder jointly inside the module. Defaults to ``"joint"``.
        num_features: Dimension of the Node2Vec embeddings, which is also used by the GCN.
        node2vec_config: Shared Node2Vec configuration used in joint mode. It is ignored
            in precomputed mode. So, it can be provided as an empty dictionary: ``{}``.
        gcn_config: Configuration for the GCN classifier.
    """

    mode: NotRequired[Node2VecMode]
    num_features: int
    node2vec_config: Node2VecNcConfig
    gcn_config: Node2VecGCNNcConfig


class Node2VecGCNNcModule(NcModule):
    """
    A LightningModule for Node2Vec-GCN multiclass node classification.

    Supports two modes:
        - ``precomputed``: use node embeddings already stored in ``batch.x``.
        - ``joint``: train a Node2Vec encoder jointly with the GCN classifier.

    Attributes:
        encoder: Optional Node2Vec-GCN encoder inherited from ``NcModule``.
        classifier: GCN classifier inherited from ``NcModule`` in precomputed mode.
        loss_fn: Loss function inherited from ``NcModule``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``NcModule``.
        train_metrics: Optional training metrics inherited from ``NcModule``.
        val_metrics: Optional validation metrics inherited from ``NcModule``.
        test_metrics: Optional test metrics inherited from ``NcModule``.
        mode: Whether to use precomputed or joint Node2Vec embeddings.
        embedding_dim: Node embedding dimension consumed by GCN.
        node2vec_config: Node2Vec configuration.
        gcn_config: GCN configuration.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
        weight_decay: Weight decay for the optimizer. Defaults to ``0.0``.
        random_walk_batch_size: Batch size used for Node2Vec walk loss in joint mode.
            Defaults to ``128``.
        node2vec_loss_weight: Weight applied to Node2Vec walk loss in joint mode.
            Defaults to ``1.0``.
    """

    def __init__(
        self,
        classifier_config: Node2VecGCNClassifierConfig,
        loss_fn: nn.Module | None = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the Node2Vec-GCN NC module.

        Args:
            classifier_config: Configuration for Node2Vec embeddings and GCN layers.
            loss_fn: Optional NC loss function. Defaults to ``CrossEntropyLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.001``.
            weight_decay: Weight decay for the optimizer. Defaults to ``0.0``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior of ``torchmetrics``.
                Defaults to ``None``.
        """
        self.mode: Node2VecMode = classifier_config.get("mode", NODE2VEC_JOINT_MODE)
        self.embedding_dim: int = classifier_config["num_features"]
        self.node2vec_config: Node2VecNcConfig = classifier_config["node2vec_config"]
        self.gcn_config: Node2VecGCNNcConfig = classifier_config["gcn_config"]

        node2vecgcn_encoder = (
            build_node2vecgcn_encoder(
                embedding_dim=self.embedding_dim,
                node2vec_config=self.node2vec_config,
                gcn_config=self.gcn_config,
                mode=self.mode,
            )
            if self.mode == NODE2VEC_JOINT_MODE
            else None
        )

        classifier = (
            build_gcn_encoder(self.embedding_dim, self.gcn_config)
            if self.mode == NODE2VEC_PRECOMPUTED_MODE
            else nn.Identity()
        )

        super().__init__(
            encoder=node2vecgcn_encoder,
            classifier=classifier,
            loss_fn=loss_fn if loss_fn is not None else nn.CrossEntropyLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.random_walk_batch_size: int = self.node2vec_config.get("random_walk_batch_size", 128)
        self.node2vec_loss_weight: float = self.node2vec_config.get("node2vec_loss_weight", 1.0)

        self.__walk_loader_state: Node2VecWalkLoaderState = Node2VecWalkLoaderState()

    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        global_node_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Predict node-class logits from Node2Vec embeddings refined by GCN.

        Args:
            x: Node feature or precomputed embedding matrix.
            hyperedge_index: Hyperedge incidence tensor.
            global_node_ids: Optional global node IDs for joint Node2Vec lookup.
                Defaults to ``None``.

        Returns:
            logits: Node-class logits.

        Raises:
            ValueError: If the configured mode cannot supply node embeddings.
        """
        gcn_edge_index = to_gcn_edge_index(hyperedge_index, self.gcn_config)

        if self.mode == NODE2VEC_JOINT_MODE:
            encoder = to_node2vecgcn_encoder(self.encoder, self.mode)
            validate_global_node_ids(encoder.num_embeddings, global_node_ids, self.mode)
            logits = encoder(batch=global_node_ids, edge_index=gcn_edge_index)
        else:
            if x.size(1) != self.embedding_dim:
                raise ValueError(
                    f"Expected precomputed node embeddings with dimension "
                    f"{self.embedding_dim}, got {x.size(1)}."
                )
            logits = self.classifier(x, gcn_edge_index)

        return logits

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Run a training step.

        In joint mode this combines NC loss with one stochastic Node2Vec walk loss batch.

        Args:
            batch: Training batch.
            batch_idx: Batch index, unused.

        Returns:
            loss: Training loss.
        """
        logits = self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)
        target_logits, target_labels = self._target_logits_and_labels(logits, batch)
        batch_size = int(target_labels.size(0))

        if self.mode == NODE2VEC_JOINT_MODE:
            positive_random_walk, negative_random_walk = next_walk_batch(
                mode=self.mode,
                encoder=self.encoder,
                batch_size=self.random_walk_batch_size,
                state=self.__walk_loader_state,
            )
            positive_random_walk = positive_random_walk.to(self.device)
            negative_random_walk = negative_random_walk.to(self.device)

            nc_loss = self.loss_fn(target_logits, target_labels)
            node2vec_loss = to_node2vec_encoder(self.encoder, self.mode).loss(
                positive_random_walk,
                negative_random_walk,
            )
            loss = nc_loss + (self.node2vec_loss_weight * node2vec_loss)

            self.log(
                stage_metric_name(Stage.TRAIN, "nc_loss"),
                nc_loss,
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
            loss = self._compute_loss(target_logits, target_labels, batch_size, Stage.TRAIN)

        self._compute_metrics(target_logits, target_labels, batch_size, Stage.TRAIN)
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
        Predict node-class logits for a batch.

        Args:
            batch: Prediction batch.
            batch_idx: Batch index, unused.

        Returns:
            logits: Predicted node-class logits.
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
        logits = self.forward(batch.x, batch.hyperedge_index, batch.global_node_ids)
        target_logits, target_labels = self._target_logits_and_labels(logits, batch)
        batch_size = int(target_labels.size(0))

        loss = self._compute_loss(target_logits, target_labels, batch_size, stage)
        self._compute_metrics(target_logits, target_labels, batch_size, stage)
        return loss
