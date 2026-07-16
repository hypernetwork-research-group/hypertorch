from torch import Tensor, nn, optim
from typing import Any, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection

from hypertorch.hyperlink_prediction.common import stage_metric_name
from hypertorch.models.node2vec_common import (
    NODE2VEC_JOINT_MODE,
    Node2VecEncoderConfig as Node2VecNCConfig,
    Node2VecMode,
    Node2VecWalkLoaderState,
    build_node2vec_encoder,
    next_walk_batch,
    to_node2vec_encoder,
    validate_global_node_ids,
)
from hypertorch.models import SLP
from hypertorch.nc.common import NCClassifier
from hypertorch.types import HData
from hypertorch.utils import Stage


class Node2VecEncoderConfig(TypedDict):
    """
    Configuration for the Node2Vec encoder in ``Node2VecClassifier``.

    Attributes:
        mode: Whether to use precomputed node embeddings from ``x`` or train a Node2Vec
            encoder jointly inside the module. Defaults to ``"joint"``.
        num_features: Dimension of the Node2Vec embeddings, which is also
            the input dimension of the classifier.
        node2vec_config: Shared Node2Vec configuration used in joint mode, or metadata for
            validating precomputed embeddings.
    """

    mode: NotRequired[Node2VecMode]
    num_features: int
    node2vec_config: Node2VecNCConfig


class Node2VecClassifierConfig(TypedDict):
    """
    Configuration for the Node2Vec classifier in ``Node2VecClassifier``.

    Attributes:
        out_channels: Number of node classes.
    """

    out_channels: int


class Node2VecClassifier(NCClassifier):
    """
    A LightningModule for Node2Vec-based NC classifier.

    Supports two modes:
        - ``precomputed``: use node embeddings already stored in ``batch.x``.
        - ``joint``: train a Node2Vec encoder jointly with the node classifier.

    Attributes:
        encoder: Optional Node2Vec encoder inherited from ``NCClassifier``.
        classifier: Classifier inherited from ``NCClassifier``.
        loss_fn: Loss function inherited from ``NCClassifier``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``NCClassifier``.
        train_metrics: Optional training metrics inherited from ``NCClassifier``.
        val_metrics: Optional validation metrics inherited from ``NCClassifier``.
        test_metrics: Optional test metrics inherited from ``NCClassifier``.
        mode: Whether to use precomputed or joint Node2Vec embeddings.
        embedding_dim: Node embedding dimension consumed by the classifier.
        lr: Learning rate for the optimizer. Defaults to ``0.001``.
        weight_decay: Weight decay for the optimizer. Defaults to ``0.0``.
        random_walk_batch_size: Batch size used for Node2Vec walk loss in joint mode.
            Defaults to ``128``.
        node2vec_loss_weight: Weight applied to Node2Vec walk loss in joint mode.
            Defaults to ``1.0``.
    """

    def __init__(
        self,
        encoder_config: Node2VecEncoderConfig,
        classifier_config: Node2VecClassifierConfig,
        loss_fn: nn.Module | None = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the Node2Vec-based NC classifier.

        Args:
            encoder_config: Configuration for the Node2Vec encoder.
            classifier_config: Configuration for the classifier.
            loss_fn: Optional NC loss function. Defaults to ``CrossEntropyLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.001``.
            weight_decay: Weight decay for the optimizer. Defaults to ``0.0``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior of ``torchmetrics``.
                Defaults to ``None``.
        """
        self.mode: Node2VecMode = encoder_config.get("mode", NODE2VEC_JOINT_MODE)
        self.embedding_dim: int = encoder_config["num_features"]
        node2vec_config = encoder_config["node2vec_config"]

        encoder = (
            build_node2vec_encoder(self.embedding_dim, node2vec_config, self.mode)
            if self.mode == NODE2VEC_JOINT_MODE
            else None
        )

        classifier = SLP(
            in_channels=self.embedding_dim,
            out_channels=classifier_config["out_channels"],
        )

        super().__init__(
            encoder=encoder,
            classifier=classifier,
            loss_fn=loss_fn if loss_fn is not None else nn.CrossEntropyLoss(),
            metrics=metrics,
            metrics_log_kwargs=metrics_log_kwargs,
        )

        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.random_walk_batch_size: int = node2vec_config.get("random_walk_batch_size", 128)
        self.node2vec_loss_weight: float = node2vec_config.get("node2vec_loss_weight", 1.0)

        self.__walk_loader_state: Node2VecWalkLoaderState = Node2VecWalkLoaderState()

    def forward(
        self,
        x: Tensor,
        global_node_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Predict node-class logits from precomputed or jointly trained Node2Vec embeddings.

        Args:
            x: Node feature or precomputed embedding matrix.
            global_node_ids: Optional global node IDs for joint Node2Vec lookup.
                Defaults to ``None``.

        Returns:
            logits: Node-class logits.

        Raises:
            ValueError: If the configured mode cannot supply node embeddings.
        """
        # Encode: get node embeddings from precomputation or joint encoder
        if self.mode == NODE2VEC_JOINT_MODE:
            encoder = to_node2vec_encoder(self.encoder, self.mode)
            validate_global_node_ids(encoder.num_embeddings, global_node_ids, self.mode)
            node_embeddings = encoder(batch=global_node_ids)
        else:
            if x.size(1) != self.embedding_dim:
                raise ValueError(
                    f"Expected precomputed node embeddings with dimension "
                    f"{self.embedding_dim}, got {x.size(1)}."
                )
            node_embeddings = x

        # Decode: linear projection to scalar score per node class
        # shape: (num_nodes, out_channels)
        logits: Tensor = self.classifier(node_embeddings)
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
        logits = self.forward(batch.x, batch.global_node_ids)
        target_logits, target_labels = self._target_logits_and_labels(logits, batch)
        batch_size = int(target_labels.size(0))

        if self.mode == NODE2VEC_JOINT_MODE:
            # Node2Vec.loss() is already a stochastic objective over sampled walks,
            # so one walk batch is a standard SGD estimate, not a logically different loss,
            # meaning we can optimize training by using a single walk batch per training step,
            # instead of averaging over multiple walk batches.
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
        return self.forward(batch.x, batch.global_node_ids)

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
        logits = self.forward(batch.x, batch.global_node_ids)
        target_logits, target_labels = self._target_logits_and_labels(logits, batch)
        batch_size = int(target_labels.size(0))

        loss = self._compute_loss(target_logits, target_labels, batch_size, stage)
        self._compute_metrics(target_logits, target_labels, batch_size, stage)
        return loss
