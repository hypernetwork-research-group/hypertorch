from torch import Tensor, nn, optim
from typing import Any, TypedDict
from typing_extensions import NotRequired
from torchmetrics import MetricCollection
from hypertorch.hyperlink_prediction.common import stage_metric_name
from hypertorch.models import SLP, VilLain
from hypertorch.types import HData
from hypertorch.utils import Stage

from hypertorch.node_classification.common import NCClassifier


class VilLainEncoderConfig(TypedDict):
    """
    Configuration for the VilLain encoder in ``VilLainClassifier``.

    Attributes:
        num_nodes: Total number of trainable nodes.
        embedding_dim: Returned node embedding dimension. Defaults to ``128``.
        labels_per_subspace: Number of virtual labels per subspace. Defaults to ``2``.
        training_steps: Propagation steps used for VilLain loss. Defaults to ``4``.
        generation_steps: Propagation steps averaged by ``forward``. Defaults to ``100``.
        tau: Gumbel-Softmax temperature. Defaults to ``1.0``.
        eps: Numerical stability constant. Defaults to ``1e-10``.
        villain_loss_weight: Weight applied to VilLain self-supervision. Defaults to ``1.0``.
    """

    num_nodes: int
    embedding_dim: NotRequired[int]
    labels_per_subspace: NotRequired[int]
    training_steps: NotRequired[int]
    generation_steps: NotRequired[int]
    tau: NotRequired[float]
    eps: NotRequired[float]
    villain_loss_weight: NotRequired[float]


class VilLainClassifierConfig(TypedDict):
    """
    Configuration for the classifier in ``VilLainClassifier``.

    Attributes:
        out_channels: Number of node classes.
    """

    out_channels: int


class VilLainClassifier(NCClassifier):
    """
    Feature-free VilLain-based NC classifier.

    The module learns transductive node embeddings from the hypergraph structure with
    ``VilLain`` and classifies the generated embeddings with an MLP.

    Attributes:
        encoder: VilLain encoder module inherited from ``NCClassifier``.
        classifier: Classifier module inherited from ``NCClassifier``.
        loss_fn: Loss function inherited from ``NCClassifier``.
        metrics_log_kwargs: Metric logging keyword arguments inherited from ``NCClassifier``.
        train_metrics: Optional training metrics inherited from ``NCClassifier``.
        val_metrics: Optional validation metrics inherited from ``NCClassifier``.
        test_metrics: Optional test metrics inherited from ``NCClassifier``.
        embedding_dim: VilLain node embedding dimension.
        lr: Learning rate for the optimizer. Defaults to ``0.01``.
        weight_decay: L2 regularization. Defaults to ``0.0``.
        villain_loss_weight: Weight applied to VilLain self-supervision. Defaults to ``1.0``.
    """

    def __init__(
        self,
        encoder_config: VilLainEncoderConfig,
        classifier_config: VilLainClassifierConfig,
        loss_fn: nn.Module | None = None,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        metrics: MetricCollection | None = None,
        metrics_log_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the VilLain-based NC classifier.

        Args:
            encoder_config: Configuration for the VilLain encoder.
            classifier_config: Configuration for the classifier.
            loss_fn: Optional NC loss function. Defaults to ``CrossEntropyLoss``.
            lr: Learning rate for the optimizer. Defaults to ``0.01``.
            weight_decay: L2 regularization. Defaults to ``0.0``.
            metrics: Optional metric collection for evaluation. Defaults to ``None``.
            metrics_log_kwargs: Additional keyword arguments passed to metric log calls.
                Useful for configuring distributed synchronization behavior
                of ``torchmetrics``. Defaults to ``None``.
        """
        self.embedding_dim: int = encoder_config.get("embedding_dim", 128)
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.villain_loss_weight: float = encoder_config.get("villain_loss_weight", 1.0)

        encoder = VilLain(
            num_nodes=encoder_config["num_nodes"],
            embedding_dim=self.embedding_dim,
            labels_per_subspace=encoder_config.get("labels_per_subspace", 2),
            training_steps=encoder_config.get("training_steps", 4),
            generation_steps=encoder_config.get("generation_steps", 100),
            tau=encoder_config.get("tau", 1.0),
            eps=encoder_config.get("eps", 1e-10),
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

    def forward(
        self,
        hyperedge_index: Tensor,
        global_node_ids: Tensor | None = None,
        num_hyperedges: int | None = None,
    ) -> Tensor:
        """
        Predict node-class logits from VilLain-generated node embeddings.

        Args:
            hyperedge_index: Hyperedge incidence tensor.
            global_node_ids: Optional global node IDs for transductive embedding lookup.
                Defaults to ``None``.
            num_hyperedges: Optional explicit number of hyperedges. Defaults to ``None``.

        Returns:
            logits: Node-class logits of shape ``(num_nodes, num_classes)``.
        """
        node_embeddings = self.__to_villain_encoder().node_embeddings(
            hyperedge_index=hyperedge_index,
            node_ids=global_node_ids,
            num_hyperedges=num_hyperedges,
        )
        logits: Tensor = self.classifier(node_embeddings)
        return logits

    def training_step(self, batch: HData, batch_idx: int) -> Tensor:
        """
        Run a training step with NC and VilLain self-supervised losses.

        Args:
            batch: Training batch.
            batch_idx: Batch index, unused.

        Returns:
            loss: Combined training loss.
        """
        logits = self.forward(
            hyperedge_index=batch.hyperedge_index,
            global_node_ids=batch.global_node_ids,
            num_hyperedges=batch.num_hyperedges,
        )
        target_logits, target_labels = self._target_logits_and_labels(logits, batch)
        batch_size = target_labels.size(0)

        nc_loss = self.loss_fn(target_logits, target_labels)
        villain_loss, villain_loss_parts = self.__to_villain_encoder().loss(
            hyperedge_index=batch.hyperedge_index,
            node_ids=batch.global_node_ids,
            num_hyperedges=batch.num_hyperedges,
        )
        loss = nc_loss + (self.villain_loss_weight * villain_loss)

        self.log(
            stage_metric_name(Stage.TRAIN, "nc_loss"),
            nc_loss,
            prog_bar=True,
            batch_size=batch_size,
            **self.metrics_log_kwargs,
        )
        self.log(
            stage_metric_name(Stage.TRAIN, "villain_loss"),
            villain_loss,
            prog_bar=True,
            batch_size=batch_size,
            **self.metrics_log_kwargs,
        )
        self.log(
            stage_metric_name(Stage.TRAIN, "local_loss"),
            villain_loss_parts["local_loss"],
            prog_bar=False,
            batch_size=batch_size,
            **self.metrics_log_kwargs,
        )
        self.log(
            stage_metric_name(Stage.TRAIN, "global_loss"),
            villain_loss_parts["global_loss"],
            prog_bar=False,
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
        return self.forward(
            hyperedge_index=batch.hyperedge_index,
            global_node_ids=batch.global_node_ids,
            num_hyperedges=batch.num_hyperedges,
        )

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
        logits = self.forward(
            hyperedge_index=batch.hyperedge_index,
            global_node_ids=batch.global_node_ids,
            num_hyperedges=batch.num_hyperedges,
        )
        target_logits, target_labels = self._target_logits_and_labels(logits, batch)
        batch_size = target_labels.size(0)

        loss = self._compute_loss(target_logits, target_labels, batch_size, stage)
        self._compute_metrics(target_logits, target_labels, batch_size, stage)
        return loss

    def __to_villain_encoder(self) -> VilLain:
        """
        Return the configured VilLain encoder.

        Returns:
            encoder: VilLain encoder instance.

        Raises:
            ValueError: If the configured encoder is missing or has the wrong type.
        """
        if self.encoder is None or not isinstance(self.encoder, VilLain):
            raise ValueError("VilLain requires a VilLain encoder, but none was provided.")
        return self.encoder
